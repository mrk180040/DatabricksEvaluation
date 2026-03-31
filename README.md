# DatabricksEvaluation

A **multi-agent LangChain framework** deployed on Databricks, evaluated with
datasets stored in **Unity Catalog**.

---

## Architecture

```
user input
    │
    ▼
┌───────────────┐
│  Supervisor   │  ──► routes to the right specialist agent
└──────┬────────┘
       │
 ┌─────┴──────────────┐
 ▼        ▼           ▼
ResearchAgent  MathAgent  SummaryAgent
       │          │           │
       └──────────┴───────────┘
              ▼
      tools (calculator, unit_converter,
             summarize_text, text_statistics)
```

The graph is built with **LangGraph**:

- **Supervisor** – reads the conversation and decides which specialist agent
  should act next (or `FINISH` if the request has been answered).
- **ResearchAgent** – factual and knowledge questions.
- **MathAgent** – arithmetic and unit conversions.
- **SummaryAgent** – text summarisation and statistics.

---

## Repository structure

```
.
├── agents/
│   ├── __init__.py
│   ├── multi_agent.py     # LangGraph multi-agent graph
│   └── tools.py           # Shared tools (calculator, unit_converter, …)
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py        # Unity Catalog dataset loader + mlflow.evaluate helpers
├── notebooks/
│   ├── 01_setup_and_deploy.py   # Databricks notebook – build, log & register model
│   └── 02_evaluate.py           # Databricks notebook – load eval dataset & score
├── tests/
│   └── test_agents.py     # Unit tests (pytest, no API key required)
├── mlflow_wrapper.py      # MLflow PythonModel wrapper for Model Serving
└── requirements.txt
```

---

## Quick start (local)

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=<your_key>

python - <<'EOF'
from agents.multi_agent import run_multi_agent
print(run_multi_agent("What is 2 ** 32?"))
EOF
```

---

## Databricks deployment

### Prerequisites

| Requirement | Details |
|---|---|
| Databricks workspace | Unity Catalog enabled |
| Databricks secret scope | `llm-secrets` with key `OPENAI_API_KEY` |
| Unity Catalog catalog | `main` (or change the `CATALOG` variable) |

### Step 1 – Set up and register the model

Import `notebooks/01_setup_and_deploy.py` into your Databricks workspace and
run all cells.  The notebook will:

1. Install Python dependencies.
2. Read your OpenAI API key from Databricks Secrets.
3. Smoke-test the agent graph locally.
4. Log the model with **MLflow** and register it in **Unity Catalog** as
   `main.agents.multi_agent_langgraph`.
5. (Optional) Deploy to a **Model Serving** endpoint.

### Step 2 – Prepare an evaluation dataset

Create a Delta table in Unity Catalog:

```sql
CREATE TABLE main.eval_datasets.qa_gold_set (
  question     STRING NOT NULL,
  ground_truth STRING,
  context      STRING
);

INSERT INTO main.eval_datasets.qa_gold_set VALUES
  ('What is 100 km in miles?', '62.137 miles',   NULL),
  ('Summarise: The sky is blue.', 'The sky is blue.', NULL),
  ('What is 12 factorial?',    '479001600',       NULL);
```

### Step 3 – Run evaluation

Import `notebooks/02_evaluate.py` into Databricks and run all cells.  The
notebook will:

1. Load the evaluation dataset from Unity Catalog.
2. Run every question through the multi-agent graph.
3. Score the answers with `mlflow.evaluate` (toxicity, answer similarity, etc.).
4. Write per-row results back to `main.eval_results.multi_agent_eval_results`.
5. Display an aggregate metrics summary.

---

## Running tests locally

```bash
pip install -r requirements.txt pytest
python -m pytest tests/ -v
```

No API key is required – all tests mock the LLM.

---

## Key dependencies

| Package | Purpose |
|---|---|
| `langchain` / `langchain-openai` | LLM abstractions & OpenAI integration |
| `langgraph` | Multi-agent graph orchestration |
| `mlflow` | Experiment tracking, model registry, evaluation |
| `databricks-sdk` | Databricks REST API (Model Serving deployment) |
| `pyspark` | Unity Catalog Delta table I/O (available in Databricks) |
