# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1 – Set Up and Deploy the Multi-Agent LangChain Framework
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Installs dependencies
# MAGIC 2. Configures secrets (OpenAI key via Databricks secret scope)
# MAGIC 3. Smoke-tests the multi-agent graph locally
# MAGIC 4. Logs the model to MLflow and registers it in Unity Catalog
# MAGIC 5. Optionally deploys the model to a Databricks Model-Serving endpoint

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
# Databricks secret scope that holds your OpenAI API key.
# Create the scope + secret with:
#   databricks secrets create-scope --scope llm-secrets
#   databricks secrets put --scope llm-secrets --key OPENAI_API_KEY
# ---------------------------------------------------------------------------
SECRET_SCOPE = "llm-secrets"      # <-- change if your scope name differs
SECRET_KEY   = "OPENAI_API_KEY"   # <-- change if your key name differs

OPENAI_API_KEY = dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# MLflow / Unity Catalog settings
CATALOG       = "main"              # <-- Unity Catalog catalog name
SCHEMA        = "agents"            # <-- schema to register the model under
MODEL_NAME    = "multi_agent_langgraph"
FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# LLM settings
LLM_MODEL   = "gpt-4o-mini"        # cost-effective default
TEMPERATURE = 0

print(f"Model will be registered as: {FULL_MODEL_NAME}")

# COMMAND ----------
# MAGIC %md ## 3 · Smoke-Test the Graph Locally

# COMMAND ----------

from agents.multi_agent import build_multi_agent_graph
from langchain_core.messages import HumanMessage

graph = build_multi_agent_graph(model=LLM_MODEL, temperature=TEMPERATURE)

test_questions = [
    "What is 1024 * 1024?",
    "Convert 100 kilometres to miles.",
    "Summarise this text: The quick brown fox jumps over the lazy dog. "
    "It did so effortlessly, showing off its agility and speed.",
]

for q in test_questions:
    result = graph.invoke({"messages": [HumanMessage(content=q)]})
    print(f"\nQ: {q}")
    print(f"A: {result['messages'][-1].content}")

# COMMAND ----------
# MAGIC %md ## 4 · Log Model to MLflow and Register in Unity Catalog

# COMMAND ----------

import mlflow
from mlflow_wrapper import MultiAgentWrapper

# Use Unity Catalog as the MLflow model registry
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Shared/{MODEL_NAME}_experiment")

with mlflow.start_run(run_name="log_multi_agent_model") as run:
    mlflow.log_params({"model": LLM_MODEL, "temperature": TEMPERATURE})

    model_info = mlflow.pyfunc.log_model(
        artifact_path="multi_agent",
        python_model=MultiAgentWrapper(),
        pip_requirements=[
            "langchain>=0.2.0",
            "langchain-community>=0.2.0",
            "langchain-openai>=0.1.0",
            "langgraph>=0.1.0",
            "mlflow>=2.13.0",
            "openai>=1.30.0",
            "pydantic>=2.0.0",
        ],
        registered_model_name=FULL_MODEL_NAME,
    )
    print(f"Model URI: {model_info.model_uri}")

run_id = run.info.run_id
print(f"Run ID:    {run_id}")

# COMMAND ----------
# MAGIC %md ## 5 · (Optional) Deploy to Databricks Model Serving

# COMMAND ----------

# Uncomment and run this cell to deploy the latest version to a serving endpoint.
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
# # Get the latest model version
# latest_version = max(
#     int(v.version)
#     for v in w.registered_models.get_latest_versions(
#         name=FULL_MODEL_NAME, stages=["None"]
#     )
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
