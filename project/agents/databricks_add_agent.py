from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from project.configs.prompts import DATABRICKS_ADD_AGENT_PROMPT
from project.utils.llm_client import LLMResponseFormatError


class DatabricksAddAgent:
    """
    Plans Databricks platform actions using an LCEL chain.

    Chain: ``ChatPromptTemplate | ChatDatabricks | JsonOutputParser``
    """

    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model
        self._chain = (
            ChatPromptTemplate.from_messages([
                ("system", DATABRICKS_ADD_AGENT_PROMPT),
                ("human", "Create a safe execution plan for:\n{query}"),
            ])
            | self.chat_model
            | JsonOutputParser()
        )

    def run(self, query: str) -> dict:
        try:
            payload = self._chain.invoke({"query": query})
        except Exception as exc:
            raise LLMResponseFormatError(f"databricks_add_agent chain failed: {exc}") from exc

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
