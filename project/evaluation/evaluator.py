from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any

try:
    import mlflow
except Exception:
    mlflow = None  # type: ignore[assignment]

from project.evaluation.dataset import load_dataset
from project.evaluation.metrics import average, keyword_match_score, response_quality_score_placeholder, routing_accuracy
from project.orchestration.orchestrator import MultiAgentOrchestrator
from project.utils.logger import get_logger, log_step, write_json


@dataclass
class EvaluationConfig:
    experiment_name: str = "databricks-multi-agent-evaluation"


class Evaluator:
    def __init__(self, orchestrator: MultiAgentOrchestrator, config: EvaluationConfig | None = None):
        self.orchestrator = orchestrator
        self.config = config or EvaluationConfig()
        self.logger = get_logger("evaluator")

    def evaluate(self, dataset_path: str) -> dict[str, Any]:
        if mlflow is None:
            raise RuntimeError("mlflow is required for evaluation mode but is not installed in this runtime.")
        dataset = load_dataset(dataset_path)
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run(run_name="evaluation_run"):
            routing_scores: list[float] = []
            keyword_scores: list[float] = []
            latency_scores: list[float] = []
            quality_scores: list[float] = []
            blocked_scores: list[float] = []
            pii_leak_scores: list[float] = []
            injection_detect_scores: list[float] = []
            secret_output_scores: list[float] = []
            rows: list[dict[str, Any]] = []

            for item in dataset:
                query = str(item["query"])
                expected_agent = str(item["expected_agent"])
                expected_keywords = item.get("expected_keywords", [])

                result = self.orchestrator.run(
                    query=query,
                    expected_agent=expected_agent,
                    track_with_mlflow=False,
                )
                trace = result["trace"]
                governance = result.get("governance", {})
                predicted_agent = str(trace["selected_agent"])
                final_answer = str(result["final_answer"])

                route_score = routing_accuracy(predicted_agent, expected_agent)
                keyword_score = keyword_match_score(final_answer, expected_keywords)
                quality_score = response_quality_score_placeholder()

                routing_scores.append(route_score)
                keyword_scores.append(keyword_score)
                latency_scores.append(float(trace["latency_ms"]))
                quality_scores.append(quality_score)
                blocked_scores.append(1.0 if governance.get("blocked", False) else 0.0)
                pii_leak_scores.append(1.0 if governance.get("output_pii_count", 0) > 0 else 0.0)
                injection_detect_scores.append(1.0 if governance.get("prompt_injection_count", 0) > 0 else 0.0)
                secret_output_scores.append(1.0 if governance.get("secret_output_count", 0) > 0 else 0.0)

                rows.append(
                    {
                        "query": query,
                        "expected_agent": expected_agent,
                        "predicted_agent": predicted_agent,
                        "routing_accuracy": route_score,
                        "keyword_match_score": keyword_score,
                        "response_quality_score": quality_score,
                        "governance_blocked": governance.get("blocked", False),
                        "governance_reason": governance.get("reason", "allowed"),
                        "input_pii_count": governance.get("input_pii_count", 0),
                        "prompt_injection_count": governance.get("prompt_injection_count", 0),
                        "secret_input_count": governance.get("secret_input_count", 0),
                        "output_pii_count": governance.get("output_pii_count", 0),
                        "secret_output_count": governance.get("secret_output_count", 0),
                        "trace": trace,
                    }
                )

            summary = {
                "dataset_size": len(rows),
                "routing_accuracy": average(routing_scores),
                "keyword_match_score": average(keyword_scores),
                "response_quality_score": average(quality_scores),
                "latency_ms_avg": average(latency_scores),
                "blocked_rate": average(blocked_scores),
                "pii_leak_rate": average(pii_leak_scores),
                "prompt_injection_detection_rate": average(injection_detect_scores),
                "secret_output_rate": average(secret_output_scores),
                "results": rows,
            }

            mlflow.log_metric("routing_accuracy", summary["routing_accuracy"])
            mlflow.log_metric("keyword_match_score", summary["keyword_match_score"])
            mlflow.log_metric("response_quality_score", summary["response_quality_score"])
            mlflow.log_metric("latency", summary["latency_ms_avg"])
            mlflow.log_metric("blocked_rate", summary["blocked_rate"])
            mlflow.log_metric("pii_leak_rate", summary["pii_leak_rate"])
            mlflow.log_metric("prompt_injection_detection_rate", summary["prompt_injection_detection_rate"])
            mlflow.log_metric("secret_output_rate", summary["secret_output_rate"])
            mlflow.log_param("dataset_path", dataset_path)

            with tempfile.TemporaryDirectory() as tmp_dir:
                evaluation_path = f"{tmp_dir}/evaluation_results.json"
                write_json(evaluation_path, summary)
                mlflow.log_artifact(evaluation_path)

            log_step(
                self.logger,
                "evaluation_completed",
                routing_accuracy=summary["routing_accuracy"],
                keyword_match_score=summary["keyword_match_score"],
                dataset_size=summary["dataset_size"],
            )
            return summary
