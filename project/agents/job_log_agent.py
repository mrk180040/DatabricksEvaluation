from __future__ import annotations

from dataclasses import dataclass

from project.configs.prompts import JOB_LOG_AGENT_PROMPT
from project.utils.llm_client import LLMClient, LLMResponseFormatError


@dataclass
class JobLogAgent:
    llm_client: LLMClient

    def run(self, query: str) -> dict:
        payload = self.llm_client.json_completion(
            system_prompt=JOB_LOG_AGENT_PROMPT,
            user_prompt=f"Analyze this Databricks job issue:\n{query}",
        )

        analysis = payload.get("analysis")
        possible_root_cause = payload.get("possible_root_cause")
        next_steps = payload.get("next_steps")

        if not isinstance(analysis, str) or not analysis.strip():
            raise LLMResponseFormatError("job_log_agent response missing non-empty 'analysis'.")
        if not isinstance(possible_root_cause, str) or not possible_root_cause.strip():
            raise LLMResponseFormatError("job_log_agent response missing non-empty 'possible_root_cause'.")
        if not isinstance(next_steps, list) or not all(isinstance(item, str) for item in next_steps):
            raise LLMResponseFormatError("job_log_agent response missing string list 'next_steps'.")

        return {
            "analysis": analysis,
            "possible_root_cause": possible_root_cause,
            "next_steps": next_steps,
        }
