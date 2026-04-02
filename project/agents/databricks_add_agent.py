from __future__ import annotations

from dataclasses import dataclass

from project.configs.prompts import DATABRICKS_ADD_AGENT_PROMPT
from project.utils.llm_client import LLMClient, LLMResponseFormatError


@dataclass
class DatabricksAddAgent:
    llm_client: LLMClient

    def run(self, query: str) -> dict:
        payload = self.llm_client.json_completion(
            system_prompt=DATABRICKS_ADD_AGENT_PROMPT,
            user_prompt=f"Create a safe execution plan for:\n{query}",
        )

        action = payload.get("action")
        status = payload.get("status")
        parameters = payload.get("parameters")

        if not isinstance(action, str) or not action.strip():
            raise LLMResponseFormatError("databricks_add_agent response missing non-empty 'action'.")
        if not isinstance(status, str) or status != "planned":
            raise LLMResponseFormatError("databricks_add_agent response must include status='planned'.")
        if not isinstance(parameters, dict):
            raise LLMResponseFormatError("databricks_add_agent response missing object 'parameters'.")

        return {
            "action": action,
            "parameters": parameters,
            "status": status,
        }
