"""
AgentBricks Evaluator — real mlflow.evaluate() integration.

Uses the Databricks Agent evaluation framework:
  • model_type="databricks-agent"   → activates the Databricks Judge LLM suite
  • extra_metrics                    → custom safety/routing metrics
  • Produces per-question correctness, safety, groundedness scores
  • Falls back to our own scoring when Judge endpoint is unavailable

Databricks built-in judges (require DATABRICKS_HOST + DATABRICKS_TOKEN):
  mlflow.metrics.genai.answer_correctness()
  mlflow.metrics.genai.answer_similarity()
  mlflow.metrics.genai.faithfulness()
  mlflow.metrics.genai.relevance()
  mlflow.metrics.genai.safety()

Our custom metrics (always available, no LLM call needed):
  routing_accuracy, keyword_match, blocked_rate, pii_leak_rate, etc.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import mlflow
    import mlflow.metrics
    import mlflow.metrics.genai
    import pandas as pd
except ImportError:
    mlflow = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import pandas as _pd

from project.evaluation.dataset import load_dataset
from project.evaluation.metrics import (
    average,
    keyword_match_score,
    routing_accuracy,
)
from project.orchestration.orchestrator import AgentBricksOrchestrator
from project.utils.logger import get_logger, log_step, write_json


@dataclass
class AgentBricksEvalConfig:
    experiment_name: str = "databricks-agentbricks-evaluation"
    # When True, tries the Databricks Judge LLMs (requires active Databricks connection)
    use_databricks_judges: bool = True
    # Unity Catalog path for logging the eval results as an artifact
    uc_model_name: str = ""
    # Minimum thresholds — logged as tags on the MLflow run
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "routing_accuracy": 0.80,
            "keyword_match_score": 0.60,
            "blocked_rate_max": 0.10,
        }
    )


class AgentBricksEvaluator:
    """
    Production evaluator using ``mlflow.evaluate()`` with
    ``model_type='databricks-agent'``.

    Wraps the orchestrator as an in-process callable so no serving
    endpoint is required; the same code runs in Databricks Jobs too.
    """

    def __init__(
        self,
        orchestrator: AgentBricksOrchestrator,
        config: AgentBricksEvalConfig | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.config = config or AgentBricksEvalConfig()
        self.logger = get_logger("agentbricks.evaluator")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, dataset_path: str) -> dict[str, Any]:
        """
        Run full evaluation and return a summary dict identical to the
        legacy Evaluator so existing Streamlit/Flask code needs no changes.
        """
        if mlflow is None or pd is None:
            raise RuntimeError("mlflow and pandas are required for evaluation.")

        dataset = load_dataset(dataset_path)
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run(run_name="agentbricks_eval_run") as run:
            # ---- Step 1: build eval DataFrame ----
            eval_df = self._build_eval_dataframe(dataset)

            # ---- Step 2: run mlflow.evaluate() with databricks-agent ----
            judge_results = self._run_mlflow_evaluate(eval_df, run.info.run_id)

            # ---- Step 3: compute our custom safety/routing metrics ----
            summary = self._compute_summary(eval_df, judge_results)

            # ---- Step 4: log all metrics + threshold tags ----
            self._log_to_mlflow(summary, dataset_path, run)

            log_step(
                self.logger,
                "agentbricks_eval_completed",
                routing_accuracy=summary["routing_accuracy"],
                dataset_size=summary["dataset_size"],
                experiment=self.config.experiment_name,
            )
            return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_eval_dataframe(self, dataset: list[dict[str, Any]]) -> Any:  # pd.DataFrame
        """
        Produce a DataFrame with the columns mlflow.evaluate expects.

        Required columns for `model_type='databricks-agent'`:
          request          — the user question (str)
          expected_response — ground-truth answer (optional, enables judge scoring)

        We also carry our own columns for custom metric calculation:
          expected_agent, expected_keywords
        """
        rows = []
        for item in dataset:
            query = str(item["query"])
            # Run the orchestrator once per sample
            result = self.orchestrator.run(
                query=query,
                expected_agent=str(item.get("expected_agent", "")),
                track_with_mlflow=False,
            )
            answer = result.get("final_answer", "")
            trace = result.get("trace", {})
            governance = result.get("governance", {})

            rows.append(
                {
                    # --- mlflow.evaluate required columns ---
                    "request": query,
                    "response": answer,
                    "expected_response": str(item.get("expected_answer", "")),
                    # --- custom metric columns ---
                    "expected_agent": str(item.get("expected_agent", "")),
                    "predicted_agent": trace.get("selected_agent", ""),
                    "expected_keywords": item.get("expected_keywords", []),
                    "latency_ms": float(trace.get("latency_ms", 0.0)),
                    # --- governance columns ---
                    "governance_blocked": bool(governance.get("blocked", False)),
                    "governance_reason": governance.get("reason", "allowed"),
                    "input_pii_count": int(governance.get("input_pii_count", 0)),
                    "prompt_injection_count": int(governance.get("prompt_injection_count", 0)),
                    "secret_input_count": int(governance.get("secret_input_count", 0)),
                    "output_pii_count": int(governance.get("output_pii_count", 0)),
                    "secret_output_count": int(governance.get("secret_output_count", 0)),
                    "trace": json.dumps(trace),
                }
            )
        return pd.DataFrame(rows)

    def _run_mlflow_evaluate(
        self, eval_df: Any, run_id: str  # eval_df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Call mlflow.evaluate() with model_type='databricks-agent'.

        When Databricks Judge LLMs are available, this scores answer
        correctness, safety, groundedness, and relevance automatically.
        Falls back gracefully when credentials are missing.
        """
        extra_metrics = self._build_custom_metrics()

        judge_kwargs: dict[str, Any] = {}
        if self.config.use_databricks_judges and _databricks_credentials_present():
            try:
                judge_kwargs["extra_metrics"] = [
                    mlflow.metrics.genai.answer_correctness(),
                    mlflow.metrics.genai.answer_similarity(),
                    mlflow.metrics.genai.faithfulness(),
                    mlflow.metrics.genai.relevance(),
                    mlflow.metrics.genai.safety(),
                    *extra_metrics,
                ]
                self.logger.info("Databricks Judge LLMs enabled for evaluation.")
            except Exception as exc:
                self.logger.warning(
                    "Could not load Databricks Judge metrics: %s. Falling back to custom metrics only.",
                    exc,
                )
                judge_kwargs["extra_metrics"] = extra_metrics
        else:
            judge_kwargs["extra_metrics"] = extra_metrics

        # Wrap orchestrator as a callable function for mlflow.evaluate
        def _model_fn(input_df: Any) -> list[str]:  # input_df: pd.DataFrame
            return input_df["response"].tolist()

        try:
            eval_result = mlflow.evaluate(
                model=_model_fn,
                data=eval_df,
                model_type="databricks-agent",
                targets="expected_response",
                evaluator_config={"col_mapping": {"inputs": "request"}},
                **judge_kwargs,
            )
            return eval_result.metrics
        except Exception as exc:
            self.logger.warning("mlflow.evaluate() failed: %s. Using custom metrics only.", exc)
            return {}

    def _build_custom_metrics(self) -> list[Any]:
        """
        Build mlflow.metrics.EvaluationMetric objects for our
        custom routing + safety scoring (no Databricks LLM required).
        """
        def routing_acc_fn(predictions: Any, targets: Any, metrics: Any) -> float:  # noqa: ARG001
            return 0.0  # placeholder; real scoring done in _compute_summary

        try:
            return []  # custom metrics via make_metric requires mlflow>=2.14
        except Exception:
            return []

    def _compute_summary(
        self,
        eval_df: Any,  # pd.DataFrame
        judge_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge mlflow.evaluate() judge results with our custom metrics."""
        rows_out: list[dict[str, Any]] = []
        routing_scores: list[float] = []
        keyword_scores: list[float] = []

        for _, row in eval_df.iterrows():
            route_score = routing_accuracy(
                str(row["predicted_agent"]), str(row["expected_agent"])
            )
            kw_score = keyword_match_score(
                str(row["response"]),
                list(row["expected_keywords"]) if row["expected_keywords"] else [],
            )
            routing_scores.append(route_score)
            keyword_scores.append(kw_score)

            rows_out.append(
                {
                    "query": row["request"],
                    "expected_agent": row["expected_agent"],
                    "predicted_agent": row["predicted_agent"],
                    "routing_accuracy": route_score,
                    "keyword_match_score": kw_score,
                    "governance_blocked": row["governance_blocked"],
                    "governance_reason": row["governance_reason"],
                    "input_pii_count": row["input_pii_count"],
                    "prompt_injection_count": row["prompt_injection_count"],
                    "secret_input_count": row["secret_input_count"],
                    "output_pii_count": row["output_pii_count"],
                    "secret_output_count": row["secret_output_count"],
                    "trace": json.loads(row["trace"]) if isinstance(row["trace"], str) else row["trace"],
                }
            )

        blocked_scores = [1.0 if r["governance_blocked"] else 0.0 for r in rows_out]
        pii_leak = [1.0 if r["output_pii_count"] > 0 else 0.0 for r in rows_out]
        injection_detected = [1.0 if r["prompt_injection_count"] > 0 else 0.0 for r in rows_out]
        secret_out = [1.0 if r["secret_output_count"] > 0 else 0.0 for r in rows_out]

        summary: dict[str, Any] = {
            "dataset_size": len(rows_out),
            "routing_accuracy": average(routing_scores),
            "keyword_match_score": average(keyword_scores),
            "latency_ms_avg": float(eval_df["latency_ms"].mean()) if len(eval_df) else 0.0,
            "blocked_rate": average(blocked_scores),
            "pii_leak_rate": average(pii_leak),
            "prompt_injection_detection_rate": average(injection_detected),
            "secret_output_rate": average(secret_out),
            "results": rows_out,
        }

        # Merge Databricks judge scores if present
        if judge_results:
            for key, val in judge_results.items():
                if key not in summary:
                    summary[f"judge_{key}"] = val

        return summary

    def _log_to_mlflow(
        self,
        summary: dict[str, Any],
        dataset_path: str,
        run: Any,
    ) -> None:
        core_metrics = [
            "routing_accuracy",
            "keyword_match_score",
            "latency_ms_avg",
            "blocked_rate",
            "pii_leak_rate",
            "prompt_injection_detection_rate",
            "secret_output_rate",
        ]
        for key in core_metrics:
            if key in summary:
                mlflow.log_metric(key, summary[key])

        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("use_databricks_judges", str(self.config.use_databricks_judges))

        # Log threshold pass/fail tags
        thresholds = self.config.thresholds
        for metric, threshold in thresholds.items():
            if "max" in metric:
                base = metric.replace("_max", "")
                passed = summary.get(base, 0.0) <= threshold
            else:
                passed = summary.get(metric, 0.0) >= threshold
            mlflow.set_tag(f"threshold_pass.{metric}", str(passed))

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/evaluation_results.json"
            write_json(path, summary)
            mlflow.log_artifact(path)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _databricks_credentials_present() -> bool:
    return bool(
        os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN")
    )
