from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from project.configs.prompts import UNITY_CATALOG_AGENT_PROMPT
from project.utils.llm_client import LLMResponseFormatError


class UnityCatalogAgent:
    """
    Answers Unity Catalog governance and metadata questions using an LCEL chain.

    Chain: ``ChatPromptTemplate | ChatDatabricks | JsonOutputParser``
    """

    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model
        self._chain = (
            ChatPromptTemplate.from_messages([
                ("system", UNITY_CATALOG_AGENT_PROMPT),
                ("human", "Answer this Unity Catalog query:\n{query}"),
            ])
            | self.chat_model
            | JsonOutputParser()
        )

    def run(self, query: str) -> dict:
        try:
            payload = self._chain.invoke({"query": query})
        except Exception as exc:
            raise LLMResponseFormatError(f"unity_catalog_agent chain failed: {exc}") from exc

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
