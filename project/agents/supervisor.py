from __future__ import annotations

from dataclasses import dataclass

from project.configs.prompts import SUPERVISOR_PROMPT
from project.utils.llm_client import LLMClient, LLMResponseFormatError


@dataclass
class SupervisorAgent:
    llm_client: LLMClient

    def run(self, query: str) -> dict:
        payload = self.llm_client.json_completion(
            system_prompt=SUPERVISOR_PROMPT,
            user_prompt=f"Route this user query:\n{query}",
        )

        selected = payload.get("selected_agent")
        if selected not in {"job_log_agent", "databricks_add_agent", "unity_catalog_agent"}:
            raise LLMResponseFormatError("supervisor response missing valid 'selected_agent'.")

        confidence = str(payload.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            raise LLMResponseFormatError("supervisor response missing valid 'confidence'.")

        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise LLMResponseFormatError("supervisor response missing non-empty 'reason'.")

        return {
            "selected_agent": selected,
            "reason": reason,
            "confidence": confidence,
        }
