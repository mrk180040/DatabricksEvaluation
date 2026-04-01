from __future__ import annotations

from dataclasses import dataclass

from project.configs.prompts import UNITY_CATALOG_AGENT_PROMPT
from project.utils.llm_client import LLMClient


@dataclass
class UnityCatalogAgent:
    llm_client: LLMClient

    def run(self, query: str) -> dict:
        fallback = {
            "answer": "This appears to be a Unity Catalog governance question. Confirm catalog, schema, and privilege scope.",
            "entities": ["catalog", "schema", "table", "privileges"],
            "confidence": "medium",
        }
        payload = self.llm_client.json_completion(
            system_prompt=UNITY_CATALOG_AGENT_PROMPT,
            user_prompt=f"Answer this Unity Catalog query:\n{query}",
            fallback=fallback,
        )

        confidence = str(payload.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        entities = payload.get("entities", fallback["entities"])
        if not isinstance(entities, list):
            entities = fallback["entities"]

        return {
            "answer": str(payload.get("answer", fallback["answer"])),
            "entities": entities,
            "confidence": confidence,
        }
