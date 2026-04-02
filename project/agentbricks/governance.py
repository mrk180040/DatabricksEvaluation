"""
AgentBricks Governance — Databricks-integrated safety layer.

Combines our custom policy engine with the Databricks serving endpoint
guardrails interface:

  1. Pre-request check (input)
       → PII detection/redaction
       → Prompt injection detection
       → Secret detection
       → Restricted topic check
       → Decision: allow / redact / block

  2. Post-response check (output)
       → PII detection/redaction
       → Secret leakage detection
       → Decision: allow / redact / block

  3. Guardrails metadata envelope
       → Formatted for Databricks serving endpoint trace UI
       → Logged to audit.jsonl

  4. Databricks Safety endpoint (optional)
       → When DATABRICKS_HOST + DATABRICKS_TOKEN are set and
         ``use_databricks_safety_endpoint=True``, the safety check
         can be routed to a Databricks-hosted safety model endpoint
         (e.g., ``databricks-llama-guard-3-8b``) for LLM-graded safety.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

from project.governance.audit import write_audit_event
from project.governance.pii import detect_pii, redact_pii
from project.governance.policy import GovernanceDecision, GovernancePolicyConfig
from project.governance.safety import detect_prompt_injection, detect_restricted_topics, detect_secrets
from project.utils.logger import get_logger

logger = get_logger("agentbricks.governance")


@dataclass
class AgentBricksGovernanceConfig:
    """
    Extended config that layers Databricks platform controls on top of
    the existing GovernancePolicyConfig logic.
    """
    # ---------- inherit base policy controls ----------
    redact_input_pii: bool = True
    redact_output_pii: bool = True
    block_if_input_has_pii: bool = False
    block_prompt_injection: bool = True
    block_if_input_has_secrets: bool = True
    block_restricted_topics: bool = True
    restricted_topics: list[str] = field(
        default_factory=lambda: ["self-harm", "malware generation", "credential theft"]
    )
    enable_audit_log: bool = True
    audit_log_path: str = "governance/logs/audit.jsonl"

    # ---------- Databricks-specific controls ----------
    # Route safety check to a Databricks-hosted LlamaGuard endpoint
    use_databricks_safety_endpoint: bool = False
    databricks_safety_model: str = "databricks-llama-guard-3-8b"

    # Severity threshold — "safe" | "unsafe"
    safety_endpoint_block_on: str = "unsafe"

    # Include the full guardrails envelope in the mlflow trace
    emit_guardrails_trace: bool = True


@dataclass
class AgentBricksGuardrailsResult:
    """Richer result that carries all Databricks-style guardrail fields."""

    # Core decision
    allow: bool
    reason: str
    text: str  # possibly redacted

    # Count fields (for eval KPIs)
    pii_count: int = 0
    injection_count: int = 0
    secret_count: int = 0
    restricted_topic_count: int = 0

    # Databricks safety endpoint result
    safety_endpoint_verdict: str = "not_checked"  # "safe" | "unsafe" | "not_checked"
    safety_endpoint_score: float = 0.0

    # Full guardrails envelope for trace / serving UI
    guardrails_metadata: dict[str, Any] = field(default_factory=dict)

    def to_governance_decision(self) -> GovernanceDecision:
        """Compatibility shim so orchestrator code that uses GovernanceDecision still works."""
        return GovernanceDecision(
            allow=self.allow,
            reason=self.reason,
            text=self.text,
            pii_count=self.pii_count,
            injection_count=self.injection_count,
            secret_count=self.secret_count,
            restricted_topic_count=self.restricted_topic_count,
        )


class AgentBricksGovernance:
    """
    Drop-in replacement for GovernancePolicy that adds Databricks
    platform safety endpoint support and richer trace metadata.
    """

    def __init__(self, config: AgentBricksGovernanceConfig | None = None) -> None:
        self.config = config or AgentBricksGovernanceConfig()
        self._safety_client = self._build_safety_client()

    # ------------------------------------------------------------------
    # Public check methods
    # ------------------------------------------------------------------

    def assess_input(self, text: str) -> AgentBricksGuardrailsResult:
        """
        Run all input checks in sequence.  Returns a guardrails result
        with the (possibly redacted) text and a block/allow decision.
        """
        pii_matches = detect_pii(text)
        injection_hits = detect_prompt_injection(text)
        secret_hits = detect_secrets(text)
        restricted_hits = detect_restricted_topics(text, self.config.restricted_topics)

        allow = True
        reason = "allowed"
        processed = text

        if pii_matches and self.config.redact_input_pii:
            processed = redact_pii(processed)
            reason = "input_pii_redacted"

        if pii_matches and self.config.block_if_input_has_pii:
            allow, reason = False, "blocked_input_pii"

        if injection_hits and self.config.block_prompt_injection:
            allow, reason = False, "blocked_prompt_injection"

        if secret_hits and self.config.block_if_input_has_secrets:
            allow, reason = False, "blocked_input_secret"

        if restricted_hits and self.config.block_restricted_topics:
            allow, reason = False, "blocked_restricted_topic"

        # -- Databricks safety endpoint (optional deep check) --
        safety_verdict = "not_checked"
        safety_score = 0.0
        if allow and self.config.use_databricks_safety_endpoint:
            safety_verdict, safety_score = self._call_safety_endpoint(processed)
            if safety_verdict == self.config.safety_endpoint_block_on:
                allow = False
                reason = f"blocked_by_databricks_safety_endpoint:{safety_verdict}"

        result = AgentBricksGuardrailsResult(
            allow=allow,
            reason=reason,
            text=processed,
            pii_count=len(pii_matches),
            injection_count=len(injection_hits),
            secret_count=len(secret_hits),
            restricted_topic_count=len(restricted_hits),
            safety_endpoint_verdict=safety_verdict,
            safety_endpoint_score=safety_score,
            guardrails_metadata=self._build_guardrails_envelope(
                stage="input",
                allow=allow,
                reason=reason,
                pii_count=len(pii_matches),
                injection_count=len(injection_hits),
                secret_count=len(secret_hits),
                restricted_count=len(restricted_hits),
                safety_verdict=safety_verdict,
            ),
        )
        self._audit("input", result)
        return result

    def assess_output(self, text: str) -> AgentBricksGuardrailsResult:
        """Run output safety checks."""
        pii_matches = detect_pii(text)
        secret_hits = detect_secrets(text)
        processed = text
        allow = True
        reason = "allowed"

        if pii_matches and self.config.redact_output_pii:
            processed = redact_pii(processed)
            reason = "output_pii_redacted"

        if secret_hits:
            allow, reason = False, "blocked_output_secret"

        safety_verdict = "not_checked"
        safety_score = 0.0
        if allow and self.config.use_databricks_safety_endpoint:
            safety_verdict, safety_score = self._call_safety_endpoint(processed)
            if safety_verdict == self.config.safety_endpoint_block_on:
                allow = False
                reason = f"blocked_output_by_databricks_safety_endpoint:{safety_verdict}"

        result = AgentBricksGuardrailsResult(
            allow=allow,
            reason=reason,
            text=processed,
            pii_count=len(pii_matches),
            secret_count=len(secret_hits),
            safety_endpoint_verdict=safety_verdict,
            safety_endpoint_score=safety_score,
            guardrails_metadata=self._build_guardrails_envelope(
                stage="output",
                allow=allow,
                reason=reason,
                pii_count=len(pii_matches),
                secret_count=len(secret_hits),
                safety_verdict=safety_verdict,
            ),
        )
        self._audit("output", result)
        return result

    # ------------------------------------------------------------------
    # Databricks Safety Endpoint integration
    # ------------------------------------------------------------------

    def _build_safety_client(self) -> Any:
        if not (
            self.config.use_databricks_safety_endpoint
            and OpenAI is not None
        ):
            return None

        host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
        token = os.getenv("DATABRICKS_TOKEN", "")
        if not host or not token:
            logger.warning(
                "DATABRICKS_HOST/TOKEN not set — safety endpoint check disabled."
            )
            return None

        return OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")

    def _call_safety_endpoint(self, text: str) -> tuple[str, float]:
        """
        Call the Databricks LlamaGuard (or any safety-classify endpoint).
        Expects the model to return a short string starting with "safe" or "unsafe".
        """
        if self._safety_client is None:
            return "not_checked", 0.0

        try:
            response = self._safety_client.chat.completions.create(
                model=self.config.databricks_safety_model,
                messages=[{"role": "user", "content": text}],
                max_tokens=10,
                temperature=0.0,
            )
            verdict_raw = (response.choices[0].message.content or "").strip().lower()
            verdict = "unsafe" if verdict_raw.startswith("unsafe") else "safe"
            score = 1.0 if verdict == "unsafe" else 0.0
            logger.info("Databricks safety endpoint verdict: %s", verdict)
            return verdict, score
        except Exception as exc:
            logger.warning("Safety endpoint call failed: %s", exc)
            return "not_checked", 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_guardrails_envelope(
        *,
        stage: str,
        allow: bool,
        reason: str,
        pii_count: int = 0,
        injection_count: int = 0,
        secret_count: int = 0,
        restricted_count: int = 0,
        safety_verdict: str = "not_checked",
    ) -> dict[str, Any]:
        """
        Build the guardrails dict that Databricks serving trace UI displays.
        Follows the pattern used by Databricks Review App guardrail blocks.
        """
        return {
            "stage": stage,
            "decision": "block" if not allow else "allow",
            "reason": reason,
            "checks": {
                "pii": {"count": pii_count, "triggered": pii_count > 0},
                "prompt_injection": {"count": injection_count, "triggered": injection_count > 0},
                "secrets": {"count": secret_count, "triggered": secret_count > 0},
                "restricted_topics": {"count": restricted_count, "triggered": restricted_count > 0},
                "databricks_safety_endpoint": {"verdict": safety_verdict},
            },
        }

    def _audit(self, stage: str, result: AgentBricksGuardrailsResult) -> None:
        if not self.config.enable_audit_log:
            return
        event: dict[str, Any] = {
            "stage": stage,
            "allow": result.allow,
            "reason": result.reason,
            "pii_count": result.pii_count,
            "injection_count": result.injection_count,
            "secret_count": result.secret_count,
            "restricted_topic_count": result.restricted_topic_count,
            "safety_endpoint_verdict": result.safety_endpoint_verdict,
        }
        if self.config.emit_guardrails_trace:
            event["guardrails"] = result.guardrails_metadata
        write_audit_event(event, self.config.audit_log_path)
