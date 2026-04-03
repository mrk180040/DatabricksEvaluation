from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from project.configs.prompts import SUPERVISOR_PROMPT
from project.utils.llm_client import LLMResponseFormatError


class SupervisorAgent:
    """
    Routes a user query to the appropriate specialist agent using an LCEL chain.

    Chain: ``ChatPromptTemplate | ChatDatabricks | JsonOutputParser``
    """

    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model
        self._chain = (
            ChatPromptTemplate.from_messages([
                ("system", SUPERVISOR_PROMPT),
                ("human", "Route this user query:\n{query}"),
            ])
            | self.chat_model
            | JsonOutputParser()
        )

    def run(self, query: str) -> dict:
        try:
            payload = self._chain.invoke({"query": query})
        except Exception as exc:
            raise LLMResponseFormatError(f"supervisor chain failed: {exc}") from exc

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

    # Backward-compat shim used by orchestrator health checks.
    @property
    def llm_client(self):
        return self.chat_model
