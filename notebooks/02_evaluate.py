# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 2 – Evaluate the Multi-Agent Framework with Unity Catalog Datasets
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads an evaluation dataset from a Unity Catalog Delta table
# MAGIC 2. Runs the multi-agent graph on every row
# MAGIC 3. Scores the results with `mlflow.evaluate` (toxicity, answer relevance, etc.)
# MAGIC 4. Persists per-question results back to a Unity Catalog table
# MAGIC 5. Displays a summary dashboard

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
import mlflow

# ---------------------------------------------------------------------------
# Secrets / credentials
# ---------------------------------------------------------------------------
SECRET_SCOPE = "llm-secrets"
SECRET_KEY   = "OPENAI_API_KEY"

OPENAI_API_KEY = dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ---------------------------------------------------------------------------
# Unity Catalog settings
# ---------------------------------------------------------------------------
CATALOG          = "main"
EVAL_SCHEMA      = "eval_datasets"          # schema that contains the gold set
EVAL_TABLE       = "qa_gold_set"            # evaluation table name
RESULTS_SCHEMA   = "eval_results"           # schema to write results to
RESULTS_TABLE    = "multi_agent_eval_results"
EVAL_LIMIT       = None                     # set to an int for a quick test run

# ---------------------------------------------------------------------------
# MLflow settings
# ---------------------------------------------------------------------------
MODEL_NAME       = "multi_agent_langgraph"
LLM_MODEL        = "gpt-4o-mini"
EXPERIMENT_PATH  = f"/Shared/{MODEL_NAME}_evaluation"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

print(f"Evaluation table : {CATALOG}.{EVAL_SCHEMA}.{EVAL_TABLE}")
print(f"Results table    : {CATALOG}.{RESULTS_SCHEMA}.{RESULTS_TABLE}")

# COMMAND ----------
# MAGIC %md ## 3 · Load Evaluation Dataset from Unity Catalog
# MAGIC
# MAGIC Expected columns in the table:
# MAGIC
# MAGIC | Column         | Type   | Required | Description                            |
# MAGIC |----------------|--------|----------|----------------------------------------|
# MAGIC | `question`     | string | ✅        | Input question / task for the agent    |
# MAGIC | `ground_truth` | string | optional | Reference answer for scoring           |
# MAGIC | `context`      | string | optional | Retrieval context (RAG use-cases)      |

# COMMAND ----------

from evaluation.evaluate import load_evaluation_dataset

eval_df = load_evaluation_dataset(
    catalog=CATALOG,
    schema=EVAL_SCHEMA,
    table=EVAL_TABLE,
    spark=spark,
    limit=EVAL_LIMIT,
)

print(f"Loaded {len(eval_df)} evaluation rows.")
display(eval_df.head())

# COMMAND ----------
# MAGIC %md ## 4 · Build the Multi-Agent Graph

# COMMAND ----------

from agents.multi_agent import build_multi_agent_graph

graph = build_multi_agent_graph(model=LLM_MODEL, temperature=0)
print("Multi-agent graph built successfully.")

# COMMAND ----------
# MAGIC %md ## 5 · Run Evaluation with `mlflow.evaluate`

# COMMAND ----------

from evaluation.evaluate import run_mlflow_evaluation

eval_result = run_mlflow_evaluation(
    graph=graph,
    eval_df=eval_df,
    run_name=f"{MODEL_NAME}_eval_run",
)

print("\n=== Aggregate Metrics ===")
for metric, value in eval_result.metrics.items():
    print(f"  {metric:40s}: {value}")

# COMMAND ----------
# MAGIC %md ## 6 · Inspect Per-Row Results

# COMMAND ----------

results_df = eval_result.tables.get("eval_results_table")
if results_df is not None:
    display(results_df)
else:
    print("No per-row results table available.")

# COMMAND ----------
# MAGIC %md ## 7 · Persist Results to Unity Catalog

# COMMAND ----------

if results_df is not None:
    # Ensure the target schema exists
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{RESULTS_SCHEMA}`")

    # Convert to Spark DataFrame and write as a Delta table
    results_sdf = spark.createDataFrame(results_df)
    (
        results_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"`{CATALOG}`.`{RESULTS_SCHEMA}`.`{RESULTS_TABLE}`")
    )
    print(f"Results saved to: {CATALOG}.{RESULTS_SCHEMA}.{RESULTS_TABLE}")
else:
    print("Nothing to save – no per-row results were produced.")

# COMMAND ----------
# MAGIC %md ## 8 · Summary Dashboard

# COMMAND ----------

import pandas as pd

summary_rows = []
for metric, value in eval_result.metrics.items():
    summary_rows.append({"metric": metric, "value": value})

summary_df = pd.DataFrame(summary_rows)
display(summary_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Next Steps
# MAGIC
# MAGIC * Adjust the `EVAL_TABLE` path to point to your own Unity Catalog gold set.
# MAGIC * Add custom `mlflow.metrics` (e.g. answer correctness, faithfulness) by
# MAGIC   passing them via the `extra_metrics` parameter of `run_mlflow_evaluation`.
# MAGIC * Schedule this notebook as a Databricks Job to automate regular evaluation.
