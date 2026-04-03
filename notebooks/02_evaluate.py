# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 2 — Evaluate the Multi-Agent Framework Against Unity Catalog
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads an evaluation dataset from a Unity Catalog Delta table
# MAGIC 2. Runs each query through the LangGraph multi-agent framework
# MAGIC 3. Computes routing accuracy, keyword match, and latency metrics via `mlflow.evaluate`
# MAGIC 4. Persists per-row results and aggregate metrics back to a Unity Catalog Delta table

# COMMAND ----------
# MAGIC %md ## 1 · Install Dependencies

# COMMAND ----------

# MAGIC %pip install -r requirements.txt -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ## 2 · Configuration

# COMMAND ----------

import os
import sys
sys.path.insert(0, "/Workspace/Repos/<your-repo-path>/DatabricksEvaluation")  # adjust to your repo path

# ---------------------------------------------------------------------------
# Unity Catalog settings
# ---------------------------------------------------------------------------
EVAL_CATALOG   = "main"                     # catalog containing evaluation data
EVAL_SCHEMA    = "eval_datasets"            # schema
EVAL_TABLE     = "qa_gold_set"             # gold-set evaluation table
RESULTS_TABLE  = "multi_agent_eval_results" # where results are written back

# Expected schema of the evaluation table:
#   question      STRING  — the input query
#   expected_agent STRING — ground-truth routing label (job_log_agent | ...)
#   expected_keywords ARRAY<STRING> — keywords expected in the answer (optional)

# Auth
try:
    OBO_TOKEN = dbutils.secrets.get(scope="agent-obo-scope", key="obo-token")
    os.environ["DATABRICKS_OBO_TOKEN"] = OBO_TOKEN
    DATABRICKS_HOST = spark.conf.get("spark.databricks.workspaceUrl", "")
    os.environ["DATABRICKS_HOST"] = f"https://{DATABRICKS_HOST}" if not DATABRICKS_HOST.startswith("http") else DATABRICKS_HOST
    print("✓ Credentials loaded from Databricks Secrets")
except Exception:
    print("⚠ Falling back to env vars for credentials")

print(f"Evaluating: {EVAL_CATALOG}.{EVAL_SCHEMA}.{EVAL_TABLE}")

# COMMAND ----------
# MAGIC %md ## 3 · Load Evaluation Dataset from Unity Catalog

# COMMAND ----------

import pandas as pd

eval_sdf = spark.table(f"`{EVAL_CATALOG}`.`{EVAL_SCHEMA}`.`{EVAL_TABLE}`")
eval_df: pd.DataFrame = eval_sdf.toPandas()

print(f"Loaded {len(eval_df)} rows from {EVAL_CATALOG}.{EVAL_SCHEMA}.{EVAL_TABLE}")
display(eval_df.head())

# COMMAND ----------
# MAGIC %md ## 4 · Run Evaluation via `mlflow.evaluate`

# COMMAND ----------

import mlflow
from project.utils.llm_client import LLMClient, LLMConfig
from project.agents.graph import build_databricks_agent_graph, initial_state
from langchain_core.messages import HumanMessage

# Build the LangGraph
llm_client = LLMClient(LLMConfig())
graph = build_databricks_agent_graph(llm_client)


def predict_fn(model_input: pd.DataFrame) -> pd.DataFrame:
    """MLflow-compatible predict function: DataFrame[question] → DataFrame[answer]."""
    answers = []
    selected_agents = []
    for question in model_input["question"].tolist():
        state = graph.invoke(initial_state(str(question)))
        answers.append(state.get("final_answer", ""))
        selected_agents.append(state.get("selected_agent", ""))
    return pd.DataFrame({"answer": answers, "selected_agent": selected_agents})


# Build the MLflow dataset (targets column is ground_truth when present)
targets_col = "ground_truth" if "ground_truth" in eval_df.columns else None
mlflow_dataset = mlflow.data.from_pandas(
    eval_df.rename(columns={"question": "question"}),
    name="unity_catalog_eval_dataset",
    targets=targets_col,
)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Shared/databricks_multi_agent_evaluation")

with mlflow.start_run(run_name="multi_agent_langgraph_eval") as run:
    mlflow.log_param("eval_table", f"{EVAL_CATALOG}.{EVAL_SCHEMA}.{EVAL_TABLE}")
    mlflow.log_param("framework", "langgraph")
    mlflow.log_param("dataset_size", len(eval_df))

    eval_result = mlflow.evaluate(
        model=predict_fn,
        data=mlflow_dataset,
        model_type="text",
        targets=targets_col,
        evaluators="default",
    )

run_id = run.info.run_id
print(f"\nRun ID: {run_id}")
print("\nAggregate metrics:")
for k, v in eval_result.metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# COMMAND ----------
# MAGIC %md ## 5 · Inspect Per-Row Results

# COMMAND ----------

results_df: pd.DataFrame = eval_result.tables.get("eval_results_table")
if results_df is not None:
    display(results_df)

# COMMAND ----------
# MAGIC %md ## 6 · Write Results Back to Unity Catalog

# COMMAND ----------

if results_df is not None:
    results_sdf = spark.createDataFrame(results_df)
    (
        results_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"`{EVAL_CATALOG}`.`{EVAL_SCHEMA}`.`{RESULTS_TABLE}`")
    )
    print(f"✓ Results written to {EVAL_CATALOG}.{EVAL_SCHEMA}.{RESULTS_TABLE}")
    display(spark.table(f"`{EVAL_CATALOG}`.`{EVAL_SCHEMA}`.`{RESULTS_TABLE}`").limit(5))
