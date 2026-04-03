# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1 — Set Up and Deploy the Databricks Multi-Agent LangGraph
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Installs Python dependencies (LangGraph, LangChain-OpenAI, MLflow, etc.)
# MAGIC 2. Validates that `DATABRICKS_HOST` and `DATABRICKS_OBO_TOKEN` are available
# MAGIC 3. Smoke-tests the LangGraph multi-agent framework locally
# MAGIC 4. Logs the model to MLflow and registers it in Unity Catalog
# MAGIC 5. Optionally deploys a Databricks Model Serving endpoint

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

# ---------------------------------------------------------------------------
# Auth — the Databricks App runtime injects these automatically.
# When running the notebook interactively, set them via the cluster environment
# or Databricks Secrets:
#   databricks secrets create-scope --scope agent-obo-scope
#   databricks secrets put-secret   --scope agent-obo-scope --key obo-token
# ---------------------------------------------------------------------------

# The host is already set in most Databricks runtimes.
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", spark.conf.get("spark.databricks.workspaceUrl", ""))
DATABRICKS_HOST = DATABRICKS_HOST if DATABRICKS_HOST.startswith("http") else f"https://{DATABRICKS_HOST}"

# Retrieve OBO token from secret scope when running in a notebook.
try:
    OBO_TOKEN = dbutils.secrets.get(scope="agent-obo-scope", key="obo-token")
    os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
    os.environ["DATABRICKS_OBO_TOKEN"] = OBO_TOKEN
    print("✓ Credentials loaded from Databricks Secrets")
except Exception:
    print("⚠ Could not load from Databricks Secrets — falling back to env vars")

# MLflow / Unity Catalog settings
CATALOG    = "main"                        # <-- your Unity Catalog catalog
SCHEMA     = "agents"                      # <-- schema to register the model under
MODEL_NAME = "databricks_multi_agent"
FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# LLM endpoint — must be an enabled Foundation Model API or custom serving endpoint
MODEL_ENDPOINT = os.getenv("DATABRICKS_MODEL_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")

print(f"Host:            {DATABRICKS_HOST}")
print(f"Model endpoint:  {MODEL_ENDPOINT}")
print(f"Registry target: {FULL_MODEL_NAME}")

# COMMAND ----------
# MAGIC %md ## 3 · Smoke-Test the LangGraph Locally

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/<your-repo-path>/DatabricksEvaluation")  # adjust to your repo path

from project.utils.llm_client import LLMClient, LLMConfig
from project.agents.graph import build_databricks_agent_graph, initial_state

llm_client = LLMClient(LLMConfig())
print(f"LLM client available: {llm_client.available()}  (auth_source={llm_client.auth_source()})")

graph = build_databricks_agent_graph(llm_client)

test_queries = [
    ("Job run failed with java.lang.OutOfMemoryError in task 3. What should I check?",   "job_log_agent"),
    ("Create a new Databricks job that runs every day at 6 AM using a shared cluster.",  "databricks_add_agent"),
    ("How do I grant SELECT privilege on catalog.sales.orders to analyst_group?",        "unity_catalog_agent"),
]

for query, expected in test_queries:
    result = graph.invoke(initial_state(query))
    predicted = result.get("selected_agent", "?")
    match = "✓" if predicted == expected else "✗"
    print(f"\n{match}  Expected: {expected}  →  Got: {predicted}")
    print(f"   Q: {query[:80]}...")
    print(f"   A: {result.get('final_answer', '')[:120]}...")

# COMMAND ----------
# MAGIC %md ## 4 · Log to MLflow and Register in Unity Catalog

# COMMAND ----------

import mlflow
sys.path.insert(0, "/Workspace/Repos/<your-repo-path>/DatabricksEvaluation")
from mlflow_wrapper import MultiAgentWrapper

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Shared/{MODEL_NAME}_experiment")

with mlflow.start_run(run_name="log_multi_agent_langgraph") as run:
    mlflow.log_param("model_endpoint", MODEL_ENDPOINT)
    mlflow.log_param("framework", "langgraph")

    model_info = mlflow.pyfunc.log_model(
        artifact_path="multi_agent",
        python_model=MultiAgentWrapper(),
        pip_requirements=[
            "langchain-openai>=0.2.0",
            "langchain-core>=0.3.0",
            "langgraph>=0.2.0",
            "openai>=1.51.0",
            "mlflow>=2.14.0",
            "pydantic>=2.0.0",
        ],
        registered_model_name=FULL_MODEL_NAME,
    )

run_id = run.info.run_id
print(f"Model URI: {model_info.model_uri}")
print(f"Run ID:    {run_id}")
print(f"Registry:  {FULL_MODEL_NAME}")

# COMMAND ----------
# MAGIC %md ## 5 · (Optional) Deploy to Databricks Model Serving

# COMMAND ----------

# Uncomment to deploy the latest registered version to a serving endpoint.
#
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import (
#     EndpointCoreConfigInput,
#     ServedModelInput,
#     ServedModelInputWorkloadSize,
# )
#
# w = WorkspaceClient()
#
# latest_version = max(
#     int(v.version)
#     for v in w.registered_models.get_latest_versions(name=FULL_MODEL_NAME, stages=["None"])
# )
#
# w.serving_endpoints.create_and_wait(
#     name=MODEL_NAME,
#     config=EndpointCoreConfigInput(
#         served_models=[
#             ServedModelInput(
#                 model_name=FULL_MODEL_NAME,
#                 model_version=str(latest_version),
#                 workload_size=ServedModelInputWorkloadSize.SMALL,
#                 scale_to_zero_enabled=True,
#             )
#         ]
#     ),
# )
# print(f"Endpoint '{MODEL_NAME}' is ready.")
