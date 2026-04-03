from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any

try:
    import mlflow
except Exception:
    mlflow = None  # type: ignore[assignment]

from langchain_core.messages import HumanMessage

from project.agents.graph import DatabricksAgentState, build_databricks_agent_graph
from project.governance import GovernancePolicy, GovernancePolicyConfig
from project.utils.databricks_llm import make_chat_model
from project.utils.llm_client import LLMClient, LLMNotConfiguredError
from project.utils.logger import get_logger, log_step, write_json


@dataclass
class OrchestratorConfig:
    confidence_threshold: str = "medium"
    fallback_agent: str = "unity_catalog_agent"
    mlflow_experiment: str = "databricks-multi-agent"
    enable_governance: bool = True


class MultiAgentOrchestrator:
    """
    Databricks LangChain/LangGraph multi-agent orchestrator.

    Architecture
    ------------
    Each agent node is a proper LangChain LCEL chain backed by ``ChatDatabricks``
    (``langchain-databricks`` package), assembled into a LangGraph ``StateGraph``::

        ChatPromptTemplate | ChatDatabricks | JsonOutputParser

    A supervisor node receives the user query, routes to one of three specialist
    workers (job_log, databricks_add, unity_catalog), and the worker's answer is
    returned via shared ``DatabricksAgentState``.

    Governance checks (PII, prompt injection, secret detection) and MLflow
    tracking are applied as pre/post-processing layers around the graph.
    """

    # Maps confidence label to a numeric rank for threshold comparison.
    _CONFIDENCE_RANK: dict[str, int] = {"low": 1, "medium": 2, "high": 3}
    # Numeric rank used when the confidence label is absent or unrecognised.
    _DEFAULT_CONFIDENCE_RANK: int = 2  # equivalent to "medium"

    def __init__(self, llm_client: LLMClient, config: OrchestratorConfig | None = None):
        self.logger = get_logger("orchestrator")
        self.config = config or OrchestratorConfig()
        # LLMClient is kept for Streamlit auth health checks (available(), auth_source()).
        self.llm_client = llm_client
        self.confidence_rank = self._CONFIDENCE_RANK
        self.governance = GovernancePolicy(GovernancePolicyConfig()) if self.config.enable_governance else None

        # Backward-compat shim so Streamlit can call orchestrator.supervisor.llm_client.available()
        class _SupervisorProxy:
            def __init__(self, client: LLMClient) -> None:
                self.llm_client = client

        self.supervisor = _SupervisorProxy(llm_client)

        # The LangGraph graph is lazy-initialised so that OBO tokens injected
        # after construction (e.g. Databricks App runtime) are picked up.
        self._graph = None

    def _get_graph(self):
        if self._graph is None:
            # Build a ChatDatabricks model and wire it into all agent LCEL chains.
            chat_model = make_chat_model()
            self._graph = build_databricks_agent_graph(chat_model)
        return self._graph

    def run(self, query: str, expected_agent: str | None = None, track_with_mlflow: bool = True) -> dict[str, Any]:
        start = time.perf_counter()
        use_mlflow = bool(track_with_mlflow and mlflow is not None)
        auth_source = self.llm_client.auth_source()
        failed_stage = "input_governance"
        if use_mlflow:
            mlflow.set_experiment(self.config.mlflow_experiment)
        run_ctx = mlflow.start_run(run_name="orchestrator_run") if use_mlflow else None
        try:
            input_decision = None
            if self.governance is not None:
                input_decision = self.governance.assess_input(query)
                if not input_decision.allow:
                    latency_ms = (time.perf_counter() - start) * 1000
                    trace = {
                        "query": query,
                        "auth_source": auth_source,
                        "selected_agent": "blocked",
                        "reason": input_decision.reason,
                        "agent_response": {},
                        "latency_ms": round(latency_ms, 2),
                        "status": "blocked",
                    }
                    governance = {
                        "blocked": True,
                        "reason": input_decision.reason,
                        "input_pii_count": input_decision.pii_count,
                        "prompt_injection_count": input_decision.injection_count,
                        "secret_input_count": input_decision.secret_count,
                        "restricted_topic_count": input_decision.restricted_topic_count,
                        "output_pii_count": 0,
                        "secret_output_count": 0,
                    }
                    return {
                        "final_answer": "Request blocked by governance policy.",
                        "trace": trace,
                        "governance": governance,
                    }
                query = input_decision.text

            log_step(self.logger, "received_query", query=query)
            failed_stage = "configuration"
            llm_available = self.llm_client.available()
            log_step(
                self.logger,
                "configuration_check",
                llm_available=llm_available,
                provider=self.llm_client.config.provider,
                auth_source=auth_source,
                host_set=bool(os.getenv("DATABRICKS_HOST")),
                env_obo_token_set=bool(os.getenv("DATABRICKS_OBO_TOKEN")),
                access_token_override_provided=self.llm_client.has_access_token_override(),
            )
            if not llm_available:
                raise LLMNotConfiguredError(
                    f"LLM client is not configured for provider={self.llm_client.config.provider} "
                    f"auth_source={auth_source}. "
                    f"Ensure DATABRICKS_OBO_TOKEN is set via agent-obo-scope/obo-token in app.yaml."
                )

            # --- LangGraph invocation ---------------------------------------
            failed_stage = "supervisor"
            graph = self._get_graph()
            initial: DatabricksAgentState = {
                "query": query,
                "messages": [HumanMessage(content=query)],
                "selected_agent": "",
                "confidence": "",
                "reason": "",
                "agent_response": {},
                "final_answer": "",
            }
            graph_result: DatabricksAgentState = graph.invoke(initial)

            selected_agent = graph_result.get("selected_agent", self.config.fallback_agent)
            reason = graph_result.get("reason", "")
            confidence = graph_result.get("confidence", "low")
            agent_response = graph_result.get("agent_response", {})
            final_answer = graph_result.get("final_answer", "")

            # Apply confidence-threshold fallback (supervisor already chose;
            # if confidence is below threshold we flag it in the trace).
            if self.confidence_rank.get(confidence, 0) < self.confidence_rank.get(self.config.confidence_threshold, self._DEFAULT_CONFIDENCE_RANK):
                reason = f"Fallback applied due to low confidence ({confidence}). Original: {reason}"
                selected_agent = self.config.fallback_agent

            log_step(
                self.logger,
                "routing_decision",
                selected_agent=selected_agent,
                reason=reason,
                confidence=confidence,
            )
            # ----------------------------------------------------------------

            latency_ms = (time.perf_counter() - start) * 1000

            if not final_answer:
                final_answer = self._to_final_answer(selected_agent, agent_response)

            output_decision = None
            if self.governance is not None:
                failed_stage = "output_governance"
                output_decision = self.governance.assess_output(final_answer)
                if output_decision.allow:
                    final_answer = output_decision.text
                else:
                    final_answer = "Response blocked by governance policy."

            trace = {
                "query": query,
                "auth_source": auth_source,
                "selected_agent": selected_agent,
                "reason": reason,
                "agent_response": agent_response,
                "latency_ms": round(latency_ms, 2),
                "status": "success",
            }
            governance = {
                "blocked": False,
                "reason": "allowed",
                "input_pii_count": input_decision.pii_count if input_decision else 0,
                "prompt_injection_count": input_decision.injection_count if input_decision else 0,
                "secret_input_count": input_decision.secret_count if input_decision else 0,
                "restricted_topic_count": input_decision.restricted_topic_count if input_decision else 0,
                "output_pii_count": output_decision.pii_count if output_decision else 0,
                "secret_output_count": output_decision.secret_count if output_decision else 0,
            }
            if output_decision is not None and not output_decision.allow:
                governance["blocked"] = True
                governance["reason"] = output_decision.reason

            final_output = {
                "final_answer": final_answer,
                "trace": trace,
                "governance": governance,
            }

            if use_mlflow:
                self._log_to_mlflow(
                    query=query,
                    selected_agent=selected_agent,
                    latency_ms=latency_ms,
                    expected_agent=expected_agent,
                    trace=trace,
                )

            log_step(self.logger, "orchestration_completed", latency_ms=round(latency_ms, 2), status="success")
            return final_output

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            error_type = type(exc).__name__
            trace = {
                "query": query,
                "auth_source": auth_source,
                "selected_agent": "none",
                "reason": f"orchestration_exception: {exc}",
                "agent_response": {},
                "failed_stage": failed_stage,
                "error_type": error_type,
                "error_message": str(exc),
                "latency_ms": round(latency_ms, 2),
                "status": "failure",
            }
            if use_mlflow:
                mlflow.log_metric("latency", latency_ms)
                mlflow.log_param("user_query", query)
                mlflow.log_param("selected_agent", "none")
                mlflow.log_param("status", "failure")
                mlflow.log_param("failed_stage", failed_stage)
                mlflow.log_param("error_type", error_type)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    trace_path = f"{tmp_dir}/trace.json"
                    write_json(trace_path, trace)
                    mlflow.log_artifact(trace_path)

            log_step(
                self.logger,
                "orchestration_failed",
                error=str(exc),
                error_type=error_type,
                failed_stage=failed_stage,
                auth_source=auth_source,
                latency_ms=round(latency_ms, 2),
            )
            return {
                "final_answer": "",
                "error": {
                    "type": error_type,
                    "message": str(exc),
                    "stage": failed_stage,
                    "auth_source": auth_source,
                },
                "trace": trace,
            }
        finally:
            if run_ctx is not None:
                mlflow.end_run()

    def _log_to_mlflow(
        self,
        *,
        query: str,
        selected_agent: str,
        latency_ms: float,
        expected_agent: str | None,
        trace: dict[str, Any],
    ) -> None:
        mlflow.log_param("user_query", query)
        mlflow.log_param("selected_agent", selected_agent)
        mlflow.log_metric("latency", latency_ms)

        if expected_agent:
            routing_accuracy = 1.0 if selected_agent == expected_agent else 0.0
            mlflow.log_metric("routing_accuracy", routing_accuracy)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = f"{tmp_dir}/trace.json"
            write_json(trace_path, trace)
            mlflow.log_artifact(trace_path)

    @staticmethod
    def _to_final_answer(selected_agent: str, response: dict[str, Any]) -> str:
        if selected_agent == "job_log_agent":
            next_steps = response.get("next_steps", [])
            return (
                f"Analysis: {response.get('analysis', '')}\n"
                f"Possible root cause: {response.get('possible_root_cause', '')}\n"
                f"Next steps: {', '.join(next_steps) if isinstance(next_steps, list) else next_steps}"
            )

        if selected_agent == "databricks_add_agent":
            return (
                f"Planned action: {response.get('action', '')}; "
                f"Status: {response.get('status', 'planned')}; "
                f"Parameters: {response.get('parameters', {})}"
            )

        return str(response.get("answer", response))



@dataclass
class OrchestratorConfig:
    confidence_threshold: str = "medium"
    fallback_agent: str = "unity_catalog_agent"
    mlflow_experiment: str = "databricks-multi-agent"
    enable_governance: bool = True


class MultiAgentOrchestrator:
    def __init__(self, llm_client: LLMClient, config: OrchestratorConfig | None = None):
        self.logger = get_logger("orchestrator")
        self.config = config or OrchestratorConfig()

        self.supervisor = SupervisorAgent(llm_client=llm_client)
        self.sub_agents = {
            "job_log_agent": JobLogAgent(llm_client=llm_client),
            "databricks_add_agent": DatabricksAddAgent(llm_client=llm_client),
            "unity_catalog_agent": UnityCatalogAgent(llm_client=llm_client),
        }
        self.confidence_rank = {"low": 1, "medium": 2, "high": 3}
        self.governance = GovernancePolicy(GovernancePolicyConfig()) if self.config.enable_governance else None

    def run(self, query: str, expected_agent: str | None = None, track_with_mlflow: bool = True) -> dict[str, Any]:
        start = time.perf_counter()
        use_mlflow = bool(track_with_mlflow and mlflow is not None)
        auth_source = self.supervisor.llm_client.auth_source()
        failed_stage = "input_governance"
        if use_mlflow:
            mlflow.set_experiment(self.config.mlflow_experiment)
        run_ctx = mlflow.start_run(run_name="orchestrator_run") if use_mlflow else None
        try:
            input_decision = None
            if self.governance is not None:
                input_decision = self.governance.assess_input(query)
                if not input_decision.allow:
                    latency_ms = (time.perf_counter() - start) * 1000
                    trace = {
                        "query": query,
                        "auth_source": auth_source,
                        "selected_agent": "blocked",
                        "reason": input_decision.reason,
                        "agent_response": {},
                        "latency_ms": round(latency_ms, 2),
                        "status": "blocked",
                    }
                    governance = {
                        "blocked": True,
                        "reason": input_decision.reason,
                        "input_pii_count": input_decision.pii_count,
                        "prompt_injection_count": input_decision.injection_count,
                        "secret_input_count": input_decision.secret_count,
                        "restricted_topic_count": input_decision.restricted_topic_count,
                        "output_pii_count": 0,
                        "secret_output_count": 0,
                    }
                    return {
                        "final_answer": "Request blocked by governance policy.",
                        "trace": trace,
                        "governance": governance,
                    }
                query = input_decision.text

            log_step(self.logger, "received_query", query=query)
            failed_stage = "configuration"
            llm_available = self.supervisor.llm_client.available()
            log_step(
                self.logger,
                "configuration_check",
                llm_available=llm_available,
                provider=self.supervisor.llm_client.config.provider,
                auth_source=auth_source,
                host_set=bool(os.getenv("DATABRICKS_HOST")),
                env_obo_token_set=bool(os.getenv("DATABRICKS_OBO_TOKEN")),
                access_token_override_provided=self.supervisor.llm_client.has_access_token_override(),
            )
            if not llm_available:
                raise LLMNotConfiguredError(
                    f"LLM client is not configured for provider={self.supervisor.llm_client.config.provider} "
                    f"auth_source={auth_source}. "
                    f"Ensure DATABRICKS_OBO_TOKEN is set via agent-obo-scope/obo-token in app.yaml."
                )
            failed_stage = "supervisor"
            supervisor_response = self.supervisor.run(query)
            selected_agent = str(supervisor_response["selected_agent"])
            reason = str(supervisor_response["reason"])
            confidence = str(supervisor_response["confidence"])

            if self.confidence_rank.get(confidence, 0) < self.confidence_rank.get(self.config.confidence_threshold, 2):
                selected_agent = self.config.fallback_agent
                reason = f"Fallback applied due to low confidence ({confidence})."

            agent = self.sub_agents.get(selected_agent, self.sub_agents[self.config.fallback_agent])

            log_step(
                self.logger,
                "routing_decision",
                selected_agent=selected_agent,
                reason=reason,
                confidence=confidence,
            )
            failed_stage = selected_agent
            agent_response = agent.run(query)

            latency_ms = (time.perf_counter() - start) * 1000
            final_answer = self._to_final_answer(selected_agent, agent_response)

            output_decision = None
            if self.governance is not None:
                failed_stage = "output_governance"
                output_decision = self.governance.assess_output(final_answer)
                if output_decision.allow:
                    final_answer = output_decision.text
                else:
                    final_answer = "Response blocked by governance policy."

            trace = {
                "query": query,
                "auth_source": auth_source,
                "selected_agent": selected_agent,
                "reason": reason,
                "agent_response": agent_response,
                "latency_ms": round(latency_ms, 2),
                "status": "success",
            }
            governance = {
                "blocked": False,
                "reason": "allowed",
                "input_pii_count": input_decision.pii_count if input_decision else 0,
                "prompt_injection_count": input_decision.injection_count if input_decision else 0,
                "secret_input_count": input_decision.secret_count if input_decision else 0,
                "restricted_topic_count": input_decision.restricted_topic_count if input_decision else 0,
                "output_pii_count": output_decision.pii_count if output_decision else 0,
                "secret_output_count": output_decision.secret_count if output_decision else 0,
            }
            if output_decision is not None and not output_decision.allow:
                governance["blocked"] = True
                governance["reason"] = output_decision.reason

            final_output = {
                "final_answer": final_answer,
                "trace": trace,
                "governance": governance,
            }

            if use_mlflow:
                self._log_to_mlflow(
                    query=query,
                    selected_agent=selected_agent,
                    latency_ms=latency_ms,
                    expected_agent=expected_agent,
                    trace=trace,
                )

            log_step(self.logger, "orchestration_completed", latency_ms=round(latency_ms, 2), status="success")
            return final_output

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            error_type = type(exc).__name__
            trace = {
                "query": query,
                "auth_source": auth_source,
                "selected_agent": "none",
                "reason": f"orchestration_exception: {exc}",
                "agent_response": {},
                "failed_stage": failed_stage,
                "error_type": error_type,
                "error_message": str(exc),
                "latency_ms": round(latency_ms, 2),
                "status": "failure",
            }
            if use_mlflow:
                mlflow.log_metric("latency", latency_ms)
                mlflow.log_param("user_query", query)
                mlflow.log_param("selected_agent", "none")
                mlflow.log_param("status", "failure")
                mlflow.log_param("failed_stage", failed_stage)
                mlflow.log_param("error_type", error_type)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    trace_path = f"{tmp_dir}/trace.json"
                    write_json(trace_path, trace)
                    mlflow.log_artifact(trace_path)

            log_step(
                self.logger,
                "orchestration_failed",
                error=str(exc),
                error_type=error_type,
                failed_stage=failed_stage,
                auth_source=auth_source,
                latency_ms=round(latency_ms, 2),
            )
            return {
                "final_answer": "",
                "error": {
                    "type": error_type,
                    "message": str(exc),
                    "stage": failed_stage,
                    "auth_source": auth_source,
                },
                "trace": trace,
            }
        finally:
            if run_ctx is not None:
                mlflow.end_run()

    def _log_to_mlflow(
        self,
        *,
        query: str,
        selected_agent: str,
        latency_ms: float,
        expected_agent: str | None,
        trace: dict[str, Any],
    ) -> None:
        mlflow.log_param("user_query", query)
        mlflow.log_param("selected_agent", selected_agent)
        mlflow.log_metric("latency", latency_ms)

        if expected_agent:
            routing_accuracy = 1.0 if selected_agent == expected_agent else 0.0
            mlflow.log_metric("routing_accuracy", routing_accuracy)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = f"{tmp_dir}/trace.json"
            write_json(trace_path, trace)
            mlflow.log_artifact(trace_path)

    @staticmethod
    def _to_final_answer(selected_agent: str, response: dict[str, Any]) -> str:
        if selected_agent == "job_log_agent":
            next_steps = response.get("next_steps", [])
            return (
                f"Analysis: {response.get('analysis', '')}\n"
                f"Possible root cause: {response.get('possible_root_cause', '')}\n"
                f"Next steps: {', '.join(next_steps) if isinstance(next_steps, list) else next_steps}"
            )

        if selected_agent == "databricks_add_agent":
            return (
                f"Planned action: {response.get('action', '')}; "
                f"Status: {response.get('status', 'planned')}; "
                f"Parameters: {response.get('parameters', {})}"
            )

        return str(response.get("answer", response))
