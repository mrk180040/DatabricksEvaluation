from __future__ import annotations

import os
from dataclasses import dataclass, field

from project.governance.audit import write_audit_event
from project.governance.pii import detect_pii, redact_pii
from project.governance.safety import detect_prompt_injection, detect_restricted_topics, detect_secrets

# Default audit log path.  In production, override GOVERNANCE_AUDIT_LOG_PATH
# to a durable location such as a Unity Catalog volume:
#   /Volumes/<catalog>/<schema>/<volume>/audit/audit.jsonl
_DEFAULT_AUDIT_LOG_PATH = os.getenv("GOVERNANCE_AUDIT_LOG_PATH", "governance/logs/audit.jsonl")


@dataclass
class GovernancePolicyConfig:
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
    audit_log_path: str = field(default_factory=lambda: _DEFAULT_AUDIT_LOG_PATH)


@dataclass
class GovernanceDecision:
    allow: bool
    reason: str
    text: str
    pii_count: int = 0
    injection_count: int = 0
    secret_count: int = 0
    restricted_topic_count: int = 0


class GovernancePolicy:
    def __init__(self, config: GovernancePolicyConfig | None = None):
        self.config = config or GovernancePolicyConfig()

    def assess_input(self, text: str) -> GovernanceDecision:
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
            allow = False
            reason = "blocked_input_pii"

        if injection_hits and self.config.block_prompt_injection:
            allow = False
            reason = "blocked_prompt_injection"

        if secret_hits and self.config.block_if_input_has_secrets:
            allow = False
            reason = "blocked_input_secret"

        if restricted_hits and self.config.block_restricted_topics:
            allow = False
            reason = "blocked_restricted_topic"

        decision = GovernanceDecision(
            allow=allow,
            reason=reason,
            text=processed,
            pii_count=len(pii_matches),
            injection_count=len(injection_hits),
            secret_count=len(secret_hits),
            restricted_topic_count=len(restricted_hits),
        )
        self._audit("input", decision)
        return decision

    def assess_output(self, text: str) -> GovernanceDecision:
        pii_matches = detect_pii(text)
        secret_hits = detect_secrets(text)
        processed = text
        allow = True
        reason = "allowed"

        if pii_matches and self.config.redact_output_pii:
            processed = redact_pii(processed)
            reason = "output_pii_redacted"

        if secret_hits:
            allow = False
            reason = "blocked_output_secret"

        decision = GovernanceDecision(
            allow=allow,
            reason=reason,
            text=processed,
            pii_count=len(pii_matches),
            secret_count=len(secret_hits),
        )
        self._audit("output", decision)
        return decision

    def _audit(self, stage: str, decision: GovernanceDecision) -> None:
        if not self.config.enable_audit_log:
            return
        write_audit_event(
            {
                "stage": stage,
                "allow": decision.allow,
                "reason": decision.reason,
                "pii_count": decision.pii_count,
                "injection_count": decision.injection_count,
                "secret_count": decision.secret_count,
                "restricted_topic_count": decision.restricted_topic_count,
            },
            self.config.audit_log_path,
        )
