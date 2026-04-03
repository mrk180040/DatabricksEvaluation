"""
Evaluation helpers for the multi-agent framework.

This module provides utilities to:
1. Load an evaluation dataset from a Unity Catalog Delta table.
2. Wrap the multi-agent graph so ``mlflow.evaluate`` can call it.
3. Run ``mlflow.evaluate`` and log the results back to the active MLflow run.

Typical Unity Catalog table schema
-----------------------------------
The evaluation table is expected to contain at least the following columns:

  • ``question``        (string) – the input question / task
  • ``ground_truth``    (string) – the expected / reference answer
  • ``context``         (string, optional) – retrieved context for RAG evaluation

All other columns are carried through unchanged.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import mlflow
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# Load evaluation dataset from Unity Catalog
# ---------------------------------------------------------------------------

def load_evaluation_dataset(
    catalog: str,
    schema: str,
    table: str,
    spark=None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load an evaluation dataset from a Unity Catalog Delta table.

    The function tries to use the ``spark`` session you pass in.  If no session
    is supplied it falls back to ``pyspark.sql.SparkSession.getActiveSession()``.

    Parameters
    ----------
    catalog : Unity Catalog catalog name (e.g. ``"main"``).
    schema  : Schema / database name (e.g. ``"eval_datasets"``).
    table   : Table name (e.g. ``"qa_gold_set"``).
    spark   : An active ``SparkSession`` (optional).
    limit   : If given, only the first *limit* rows are loaded.

    Returns
    -------
    A ``pandas.DataFrame`` with at least a ``question`` column.

    Raises
    ------
    RuntimeError
        When no active SparkSession is available and none was supplied.
    """
    if spark is None:
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
        except Exception as exc:
            raise RuntimeError(
                "No active SparkSession found.  Pass a SparkSession via the "
                "'spark' parameter or run inside a Databricks notebook."
            ) from exc

    full_table_name = f"`{catalog}`.`{schema}`.`{table}`"
    query = f"SELECT * FROM {full_table_name}"
    if limit is not None:
        query += f" LIMIT {limit}"

    sdf = spark.sql(query)
    return sdf.toPandas()


# ---------------------------------------------------------------------------
# MLflow-compatible model function
# ---------------------------------------------------------------------------

def _build_predict_fn(graph):
    """
    Return a callable that accepts a ``pandas.DataFrame`` with a ``question``
    column and returns a ``pandas.DataFrame`` with an ``answer`` column.

    This signature is what ``mlflow.evaluate`` expects from the ``model``
    argument when ``model`` is a plain Python callable.
    """

    def predict(model_input: pd.DataFrame) -> pd.DataFrame:
        answers = []
        for question in model_input["question"].tolist():
            result = graph.invoke(
                {"messages": [HumanMessage(content=str(question))]}
            )
            answers.append(result["messages"][-1].content)
        return pd.DataFrame({"answer": answers})

    return predict


# ---------------------------------------------------------------------------
# Run mlflow.evaluate
# ---------------------------------------------------------------------------

def run_mlflow_evaluation(
    graph,
    eval_df: pd.DataFrame,
    run_name: str = "multi_agent_evaluation",
    extra_metrics: Optional[list] = None,
    evaluators: str = "default",
) -> mlflow.models.EvaluationResult:
    """
    Evaluate the multi-agent graph using ``mlflow.evaluate``.

    Parameters
    ----------
    graph        : A compiled LangGraph graph (from ``build_multi_agent_graph()``).
    eval_df      : Evaluation dataset as a DataFrame.  Must have a ``question``
                   column; an optional ``ground_truth`` column enables answer
                   correctness metrics.
    run_name     : MLflow run name.
    extra_metrics: Additional ``mlflow.metrics`` objects to compute.
    evaluators   : Evaluator name(s) forwarded to ``mlflow.evaluate``.

    Returns
    -------
    ``mlflow.models.EvaluationResult`` – contains the aggregate metrics dict
    and a per-row results DataFrame.
    """
    predict_fn = _build_predict_fn(graph)

    # Build the MLflow evaluation dataset
    targets_col = "ground_truth" if "ground_truth" in eval_df.columns else None
    mlflow_dataset = mlflow.data.from_pandas(
        eval_df,
        name="unity_catalog_eval_dataset",
        targets=targets_col,
    )

    with mlflow.start_run(run_name=run_name, nested=True):
        eval_result = mlflow.evaluate(
            model=predict_fn,
            data=mlflow_dataset,
            model_type="text",
            targets=targets_col,
            extra_metrics=extra_metrics or [],
            evaluators=evaluators,
        )

    return eval_result


# ---------------------------------------------------------------------------
# Log evaluation results
# ---------------------------------------------------------------------------

def log_evaluation_results(
    eval_result: mlflow.models.EvaluationResult,
    artifact_prefix: str = "eval",
) -> None:
    """
    Log aggregate metrics and the per-row results table to the active MLflow run.

    Parameters
    ----------
    eval_result    : The result returned by ``run_mlflow_evaluation``.
    artifact_prefix: Prefix for logged artifact file names.
    """
    with mlflow.start_run(nested=True):
        # Log aggregate scalars
        for metric_name, value in eval_result.metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{artifact_prefix}/{metric_name}", value)

        # Log the per-row results table
        results_df: pd.DataFrame = eval_result.tables.get("eval_results_table")
        if results_df is not None:
            mlflow.log_table(results_df, artifact_file=f"{artifact_prefix}/results.json")
