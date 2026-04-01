from __future__ import annotations

from dataclasses import dataclass
import re

from project.configs.prompts import SUPERVISOR_PROMPT
from project.utils.llm_client import LLMClient


@dataclass
class SupervisorAgent:
    llm_client: LLMClient

    def run(self, query: str) -> dict:
        fallback = self._fallback_route(query)
        payload = self.llm_client.json_completion(
            system_prompt=SUPERVISOR_PROMPT,
            user_prompt=f"Route this user query:\n{query}",
            fallback=fallback,
        )

        selected = payload.get("selected_agent")
        if selected not in {"job_log_agent", "databricks_add_agent", "unity_catalog_agent"}:
            return fallback

        confidence = str(payload.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"

        return {
            "selected_agent": selected,
            "reason": str(payload.get("reason", fallback["reason"])),
            "confidence": confidence,
        }

    @staticmethod
    def _fallback_route(query: str) -> dict:
        q = query.lower()
        def contains_phrase(phrase: str) -> bool:
            return re.search(rf"\b{re.escape(phrase)}\b", q) is not None

        if any(
            contains_phrase(token)
            for token in ["job failed", "error", "stacktrace", "exception", "run failed", "logs", "failed"]
        ):
            return {
                "selected_agent": "job_log_agent",
                "reason": "Detected failure/log troubleshooting pattern.",
                "confidence": "medium",
            }
        if any(
            contains_phrase(token)
            for token in [
                "create cluster",
                "add user",
                "grant workspace",
                "schedule job",
                "create job",
                "new job",
                "provision",
                "add",
            ]
        ):
            return {
                "selected_agent": "databricks_add_agent",
                "reason": "Detected platform action/provisioning request.",
                "confidence": "medium",
            }
        return {
            "selected_agent": "unity_catalog_agent",
            "reason": "Defaulted to data governance/metadata support.",
            "confidence": "low",
        }
