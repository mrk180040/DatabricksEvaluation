from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from project.configs.prompts import JOB_LOG_AGENT_PROMPT
from project.utils.llm_client import LLMResponseFormatError


class JobLogAgent:
    """
    Diagnoses Databricks job failures using an LCEL chain.

    Chain: ``ChatPromptTemplate | ChatDatabricks | JsonOutputParser``
    """

    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model
        self._chain = (
            ChatPromptTemplate.from_messages([
                ("system", JOB_LOG_AGENT_PROMPT),
                ("human", "Analyze this Databricks job issue:\n{query}"),
            ])
            | self.chat_model
            | JsonOutputParser()
        )

    def run(self, query: str) -> dict:
        try:
            payload = self._chain.invoke({"query": query})
        except Exception as exc:
            raise LLMResponseFormatError(f"job_log_agent chain failed: {exc}") from exc

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
