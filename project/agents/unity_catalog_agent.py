from __future__ import annotations

from dataclasses import dataclass

from project.configs.prompts import UNITY_CATALOG_AGENT_PROMPT
from project.utils.llm_client import LLMClient, LLMResponseFormatError


@dataclass
class UnityCatalogAgent:
    llm_client: LLMClient

    def run(self, query: str) -> dict:
        payload = self.llm_client.json_completion(
            system_prompt=UNITY_CATALOG_AGENT_PROMPT,
            user_prompt=f"Answer this Unity Catalog query:\n{query}",
        )

        answer = payload.get("answer")
        entities = payload.get("entities")
        confidence = str(payload.get("confidence", "")).lower()

        if not isinstance(answer, str) or not answer.strip():
            raise LLMResponseFormatError("unity_catalog_agent response missing non-empty 'answer'.")
        if confidence not in {"high", "medium", "low"}:
            raise LLMResponseFormatError("unity_catalog_agent response missing valid 'confidence'.")
        if not isinstance(entities, list):
            raise LLMResponseFormatError("unity_catalog_agent response missing list 'entities'.")

        return {
            "answer": answer,
            "entities": entities,
            "confidence": confidence,
        }
