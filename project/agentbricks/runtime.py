"""
AgentBricks Runtime — real MLflow PythonModel adapter.

This module wraps AgentBricksOrchestrator in the exact contract that
Databricks Model Serving / Agent Framework expects:

INPUT  (from serving endpoint or mlflow.evaluate):
  pd.DataFrame with a `messages` column, where each cell is a list of
  OpenAI-format message dicts:
    [{"role": "user", "content": "..."}, ...]

OUTPUT (StringResponse-compatible):
  list[str] or str, where each value is the assistant response text.

Deployment lifecycle:
  1. Instantiate AgentBricksModel and log it via mlflow.pyfunc.log_model().
  2. Register in Unity Catalog via AgentBricksDeployer.register().
  3. Deploy with databricks.agents.deploy() via AgentBricksDeployer.deploy().
"""
from __future__ import annotations

import logging
import os
from typing import Any

try:
    import mlflow
    import mlflow.pyfunc
    import pandas as pd
except ImportError:
    mlflow = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]

from project.governance.policy import GovernancePolicy, GovernancePolicyConfig
from project.orchestration.orchestrator import AgentBricksConfig, AgentBricksOrchestrator
from project.utils.llm_client import LLMClient, LLMConfig

logger = logging.getLogger("agentbricks.runtime")


class AgentBricksModel(mlflow.pyfunc.PythonModel if mlflow else object):  # type: ignore[misc]
    """
    MLflow PythonModel implementation that satisfies the Databricks
    Agent Framework serving contract.
    """

    def __init__(
        self,
        orchestrator_config: AgentBricksConfig | None = None,
        governance_config: GovernancePolicyConfig | None = None,
    ) -> None:
        self._orchestrator_config = orchestrator_config or AgentBricksConfig()
        self._governance_config = governance_config or GovernancePolicyConfig()
        self._orchestrator: AgentBricksOrchestrator | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Keep the model pickle-safe for MLflow by dropping live runtime objects."""
        state = self.__dict__.copy()
        state["_orchestrator"] = None
        return state

    # ------------------------------------------------------------------
    # MLflow PythonModel lifecycle
    # ------------------------------------------------------------------

    def load_context(self, context: Any) -> None:  # noqa: ARG002
        """Called by MLflow when loading the model from a serving endpoint."""
        self._build_orchestrator()

    def _build_orchestrator(self) -> None:
        from dotenv import load_dotenv

        load_dotenv()
        llm_client = LLMClient(LLMConfig())
        self._orchestrator = AgentBricksOrchestrator(
            llm_client=llm_client,
            config=self._orchestrator_config,
        )

    # ------------------------------------------------------------------
    # Prediction — the core serving contract
    # ------------------------------------------------------------------

    def predict(
        self,
        context: Any,  # mlflow.pyfunc.PythonModelContext
        model_input,   # pd.DataFrame | dict | str — no type hint avoids MLflow schema warning
        params: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> Any:  # pd.DataFrame | dict
        """
        Transform OpenAI-format messages into a ChatCompletionResponse.

        Databricks Model Serving sends a DataFrame with a ``messages``
        column. We also accept a plain dict for notebook/CLI use.
        
        Returns a response dict compatible with ChatCompletionResponse schema:
          {
            "choices": [
              {
                "message": {"role": "assistant", "content": "..."},
                "finish_reason": "stop"
              }
            ]
          }
        """
        if self._orchestrator is None:
            self._build_orchestrator()

        rows = self._parse_input(model_input)
        results = [self._run_one(row) for row in rows]

        if pd is not None and not isinstance(model_input, dict):
            # Return as DataFrame for serving endpoint
            if results and isinstance(results[0], dict):
                return pd.DataFrame(results)
            # Wrap string results in proper format
            return pd.DataFrame([
                {
                    "choices": [{"message": {"role": "assistant", "content": r}, "finish_reason": "stop"}]
                }
                for r in results
            ])

        # Single-row dict shortcut (notebook / test usage)
        if len(results) == 1:
            result = results[0]
            return result if isinstance(result, dict) else {
                "choices": [{"message": {"role": "assistant", "content": str(result)}, "finish_reason": "stop"}]
            }
        return results

    # ------------------------------------------------------------------
    # Per-row execution helper
    # ------------------------------------------------------------------

    def _run_one(self, query: str) -> dict[str, Any]:
        """
        Execute orchestrator and return ChatCompletionResponse-compatible dict.
        """
        assert self._orchestrator is not None

        result = self._orchestrator.run(query=query, track_with_mlflow=False)
        answer = result.get("final_answer", "No answer found")
        
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": str(answer)},
                    "finish_reason": "stop"
                }
            ]
        }

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_input(model_input: Any) -> list[str]:
        """
        Accept:
          • pd.DataFrame with a ``messages`` column
          • dict  {"messages": [...]} or {"query": "..."}
          • plain str  (convenience)
        Returns a list of query strings.
        """
        if isinstance(model_input, str):
            return [model_input]

        if isinstance(model_input, dict):
            if "query" in model_input:
                return [str(model_input["query"])]
            messages = model_input.get("messages", [])
            return [_extract_last_user_message(messages)]

        if pd is not None and isinstance(model_input, pd.DataFrame):
            queries: list[str] = []
            for _, row in model_input.iterrows():
                messages = row.get("messages", [])
                if isinstance(messages, str):
                    queries.append(messages)
                else:
                    queries.append(_extract_last_user_message(messages))
            return queries

        raise TypeError(f"Unsupported model_input type: {type(model_input)}")


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _extract_last_user_message(messages: list[dict[str, str]]) -> str:
    """Return the content of the last user message in an OpenAI messages list."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    # Fallback: concatenate all content
    return " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))


def _build_guardrails_metadata(governance: dict[str, Any]) -> dict[str, Any]:
    """
    Format governance decisions into the Databricks serving guardrails
    envelope.  This is surfaced in the MLflow review app and trace UI.
    """
    blocked = governance.get("blocked", False)
    reason = governance.get("reason", "allowed")

    return {
        "blocked": blocked,
        "block_reason": reason if blocked else None,
        "input_checks": {
            "pii_detected": governance.get("input_pii_count", 0) > 0,
            "injection_detected": governance.get("prompt_injection_count", 0) > 0,
            "secret_detected": governance.get("secret_input_count", 0) > 0,
            "restricted_topic": governance.get("restricted_topic_count", 0) > 0,
        },
        "output_checks": {
            "pii_detected": governance.get("output_pii_count", 0) > 0,
            "secret_detected": governance.get("secret_output_count", 0) > 0,
        },
    }


# ------------------------------------------------------------------
# Model logging convenience helper
# ------------------------------------------------------------------

def log_agentbricks_model(
    *,
    model: AgentBricksModel,
    artifact_path: str = "agentbricks_agent",
    pip_requirements: list[str] | None = None,
) -> str:
    """
    Log an AgentBricksModel with mlflow.pyfunc.log_model() using Databricks-compatible
    ChatCompletionRequest/Response signatures for Agent Framework deployment.

    Usage
    -----
    with mlflow.start_run():
        uri = log_agentbricks_model(model=AgentBricksModel())
        deployer = AgentBricksDeployer(model_uri=uri, uc_model_name="main.agents.my_agent")
        version = deployer.register()
        deployer.deploy(version)
    """
    if mlflow is None:
        raise RuntimeError("mlflow is required but not installed.")

    default_pip = [
        "mlflow>=2.14.0",
        "openai>=1.51.0",
        "python-dotenv>=1.0.1",
        "databricks-agents>=0.5.0",
    ]
    requirements = pip_requirements or default_pip

    # Use Databricks' own signature classes (currently in rag_signatures, will move to llm)
    # These are what databricks.agents.deploy() validates against
    from mlflow.models.rag_signatures import ChatCompletionRequest, ChatCompletionResponse

    # Create request/response examples that match the schema
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Job failed with OOM error"}]
    )
    
    response = ChatCompletionResponse(
        choices=[{"message": {"role": "assistant", "content": "The job failed due to insufficient memory."}, "finish_reason": "stop"}]
    )

    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=model,
        pip_requirements=requirements,
        input_example=request,
    )

    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "unknown"
    return f"runs:/{run_id}/{artifact_path}"
