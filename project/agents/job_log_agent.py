from __future__ import annotations

from dataclasses import dataclass

from project.configs.prompts import JOB_LOG_AGENT_PROMPT
from project.utils.llm_client import LLMClient


@dataclass
class JobLogAgent:
    llm_client: LLMClient

    def run(self, query: str) -> dict:
        fallback = {
            "analysis": "The job appears to have failed during execution and requires log-level diagnostics.",
            "possible_root_cause": "Transient infrastructure issue or code/runtime dependency mismatch.",
            "next_steps": [
                "Inspect failed run output and cluster event logs.",
                "Validate task dependencies and runtime version compatibility.",
                "Rerun with debug logging enabled and compare with last successful run.",
            ],
        }
        payload = self.llm_client.json_completion(
            system_prompt=JOB_LOG_AGENT_PROMPT,
            user_prompt=f"Analyze this Databricks job issue:\n{query}",
            fallback=fallback,
        )

        return {
            "analysis": str(payload.get("analysis", fallback["analysis"])),
            "possible_root_cause": str(payload.get("possible_root_cause", fallback["possible_root_cause"])),
            "next_steps": payload.get("next_steps", fallback["next_steps"]),
        }
