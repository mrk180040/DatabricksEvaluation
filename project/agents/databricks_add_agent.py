from __future__ import annotations

from dataclasses import dataclass

from project.configs.prompts import DATABRICKS_ADD_AGENT_PROMPT
from project.utils.llm_client import LLMClient


@dataclass
class DatabricksAddAgent:
    llm_client: LLMClient

    def run(self, query: str) -> dict:
        fallback = {
            "action": "plan_databricks_change",
            "parameters": {
                "requested_change": query,
                "requires_approval": True,
                "target_environment": "dev",
            },
            "status": "planned",
        }
        payload = self.llm_client.json_completion(
            system_prompt=DATABRICKS_ADD_AGENT_PROMPT,
            user_prompt=f"Create a safe execution plan for:\n{query}",
            fallback=fallback,
        )

        status = str(payload.get("status", "planned"))
        if status != "planned":
            status = "planned"
        parameters = payload.get("parameters", fallback["parameters"])
        if not isinstance(parameters, dict):
            parameters = fallback["parameters"]

        return {
            "action": str(payload.get("action", fallback["action"])),
            "parameters": parameters,
            "status": status,
        }
