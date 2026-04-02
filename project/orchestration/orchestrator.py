from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from typing import Any

try:
    import mlflow
except Exception:
    mlflow = None  # type: ignore[assignment]

from project.agents import DatabricksAddAgent, JobLogAgent, SupervisorAgent, UnityCatalogAgent
from project.agentbricks.governance import AgentBricksGovernance, AgentBricksGovernanceConfig
from project.utils.llm_client import LLMClient
from project.utils.logger import get_logger, log_step, write_json


@dataclass
class AgentBricksConfig:
    confidence_threshold: str = "medium"
    fallback_agent: str = "unity_catalog_agent"
    mlflow_experiment: str = "databricks-agentbricks"
    enable_governance: bool = True
    # When True, use AgentBricksGovernance (Databricks safety endpoint + guardrails trace)
    # When False, governance is disabled entirely
    use_databricks_safety_endpoint: bool = False


class AgentBricksOrchestrator:
    def __init__(self, llm_client: LLMClient, config: AgentBricksConfig | None = None):
        self.logger = get_logger("orchestrator")
        self.config = config or AgentBricksConfig()

        self.supervisor = SupervisorAgent(llm_client=llm_client)
        self.sub_agents = {
            "job_log_agent": JobLogAgent(llm_client=llm_client),
            "databricks_add_agent": DatabricksAddAgent(llm_client=llm_client),
            "unity_catalog_agent": UnityCatalogAgent(llm_client=llm_client),
        }
        self.confidence_rank = {"low": 1, "medium": 2, "high": 3}
        gov_cfg = AgentBricksGovernanceConfig(
            use_databricks_safety_endpoint=self.config.use_databricks_safety_endpoint,
        )
        self.governance = AgentBricksGovernance(gov_cfg) if self.config.enable_governance else None

    def run(self, query: str, expected_agent: str | None = None, track_with_mlflow: bool = True) -> dict[str, Any]:
        start = time.perf_counter()
        use_mlflow = bool(track_with_mlflow and mlflow is not None)
        if use_mlflow:
            mlflow.set_experiment(self.config.mlflow_experiment)
        run_ctx = mlflow.start_run(run_name="orchestrator_run") if use_mlflow else None
        try:
            input_decision = None
            if self.governance is not None:
                _raw_input = self.governance.assess_input(query)
                input_decision = _raw_input.to_governance_decision()
                _input_guardrails = _raw_input.guardrails_metadata
                if not input_decision.allow:
                    latency_ms = (time.perf_counter() - start) * 1000
                    latency_ms = (time.perf_counter() - start) * 1000
                    trace = {
                        "query": query,
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
                        "guardrails": _input_guardrails,
                    }
                    return {
                        "final_answer": "Request blocked by governance policy.",
                        "trace": trace,
                        "governance": governance,
                    }
                query = input_decision.text

            log_step(self.logger, "received_query", query=query)
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
            agent_response = agent.run(query)

            latency_ms = (time.perf_counter() - start) * 1000
            final_answer = self._to_final_answer(selected_agent, agent_response)

            output_decision = None
            _output_guardrails: dict[str, Any] = {}
            if self.governance is not None:
                _raw_output = self.governance.assess_output(final_answer)
                output_decision = _raw_output.to_governance_decision()
                _output_guardrails = _raw_output.guardrails_metadata
                if output_decision.allow:
                    final_answer = output_decision.text
                else:
                    final_answer = "Response blocked by governance policy."

            trace = {
                "query": query,
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
                # Databricks guardrails trace metadata
                "guardrails": {
                    "input": _input_guardrails if input_decision else {},
                    "output": _output_guardrails,
                },
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
            trace = {
                "query": query,
                "selected_agent": "none",
                "reason": f"orchestration_exception: {exc}",
                "agent_response": {},
                "latency_ms": round(latency_ms, 2),
                "status": "failure",
            }
            if use_mlflow:
                mlflow.log_metric("latency", latency_ms)
                mlflow.log_param("user_query", query)
                mlflow.log_param("selected_agent", "none")
                with tempfile.TemporaryDirectory() as tmp_dir:
                    trace_path = f"{tmp_dir}/trace.json"
                    write_json(trace_path, trace)
                    mlflow.log_artifact(trace_path)

            log_step(self.logger, "orchestration_failed", error=str(exc), latency_ms=round(latency_ms, 2))
            return {
                "final_answer": "Request processing failed.",
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


OrchestratorConfig = AgentBricksConfig
MultiAgentOrchestrator = AgentBricksOrchestrator
