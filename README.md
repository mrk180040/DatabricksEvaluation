# Databricks Multi-Agent Framework with MLflow Evaluation

## What is included

- Supervisor router agent
- Sub-agents:
  - `job_log_agent`
  - `databricks_add_agent`
  - `unity_catalog_agent`
- Orchestrator with full trace object
- MLflow logging for params, metrics, artifacts
- Evaluation pipeline with dataset + metrics

## Evaluation vs Governance (simple)

- **Evaluation** answers: "How good is the system?"
  - Example metrics: routing accuracy, keyword match, latency
  - Used for quality tracking and model/agent improvements

- **Governance** answers: "Is the system safe and compliant?"
  - Checks for PII, prompt-injection patterns, secrets, restricted topics
  - Can **allow**, **redact**, or **block** content based on policy

## Guardrails currently implemented

- **Input guardrails**
  - Detect and redact PII (email/phone/SSN/card)
  - Detect prompt-injection attempts
  - Detect secret/token-like strings
  - Detect restricted topics
  - Block requests when policy requires

- **Output guardrails**
  - Detect and redact PII
  - Block secret leakage in output

- **Audit + visibility**
  - Governance events written to `governance/logs/audit.jsonl`
  - Query results include a `governance` object with counts/reason
  - Streamlit UI shows guardrail status and governance KPIs in Evaluation tab

## Project layout

```text
project/
  agents/
  orchestration/
  evaluation/
  utils/
  configs/
  data/
  main.py
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```bash
cat > .env << 'EOF'
LLM_PROVIDER=databricks
DATABRICKS_HOST=https://<your-databricks-workspace>
DATABRICKS_TOKEN=<your-token>
DATABRICKS_MODEL_ENDPOINT=databricks-meta-llama-3-3-70b-instruct
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=700
LLM_MAX_RETRIES=3
EOF
```

## Run one query

```bash
PYTHONPATH=. python -m project.main --mode run --query "Job failed with out of memory"
```

## Run evaluation

```bash
PYTHONPATH=. python -m project.main --mode evaluate \
  --dataset project/data/sample_dataset.json \
  --output project/data/evaluation_results.json
```

## MLflow tracking outputs

- Params: `user_query`, `selected_agent`
- Metrics: `latency`, `routing_accuracy`, `keyword_match_score`
- Artifacts: `trace.json`, `evaluation_results.json`

## Run in a Databricks notebook

1. Create a Python notebook and attach a cluster.
2. Install dependencies in a notebook cell:

```python
%pip install mlflow openai python-dotenv
```

3. Set secrets/env vars in notebook:

```python
import os
os.environ["LLM_PROVIDER"] = "databricks"
os.environ["DATABRICKS_HOST"] = "https://<workspace>"
os.environ["DATABRICKS_TOKEN"] = "<token>"
os.environ["DATABRICKS_MODEL_ENDPOINT"] = "databricks-meta-llama-3-3-70b-instruct"
```

4. Run framework:

```python
from project.main import run_query, run_evaluation

run_query("How do I grant SELECT privilege on catalog.sales.orders to analyst_group?")
run_evaluation("project/data/sample_dataset.json", "project/data/evaluation_results.json")
```
# Databricks Multi-Agent + Evaluation Framework

This repository provides:
- A LangChain/LangGraph multi-agent framework (supervisor + specialist agents)
- A lightweight evaluation framework (JSONL dataset + metrics + report generation)
- An end-to-end governance framework (PII, injection, secrets, policy actions, audit logs)
- Databricks Asset Bundle files for deployment as a Databricks Job

## Project Structure

- `src/databricks_multi_agent/` – multi-agent orchestration framework
- `src/evaluation/` – evaluation runner and metrics
- `data/kb.json` – sample retrieval knowledge base
- `evaluation/datasets/sample_eval.jsonl` – sample eval set
- `resources/evaluation_job.yml` – Databricks job resource
- `databricks.yml` – Databricks Asset Bundle config
- `governance/policy.yaml` – governance policy controls

## 1) Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create environment variables:

```bash
cp .env.example .env
```

Then set values in `.env` (or export as shell env vars).

## 2) Run Multi-Agent Framework

```bash
PYTHONPATH=src python -m databricks_multi_agent.main "What does Databricks Model Serving do?"
```

Example math query:

```bash
PYTHONPATH=src python -m databricks_multi_agent.main "Calculate 12 * (8 + 1)"
```

## 3) Run Evaluation Framework

```bash
PYTHONPATH=src python -m evaluation.runner \
	--dataset evaluation/datasets/sample_eval.jsonl \
	--output evaluation/results/latest_results.json
```

Metrics included:
- Exact Match (normalized)
- Token-overlap F1
- PII Leak Rate
- Blocked Rate
- Prompt Injection Detection Rate
- Secret Output Rate

Output:
- Console summary table
- Full JSON report in `evaluation/results/`

## 4) Deploy to Databricks (Asset Bundles)

Prerequisites:
- Databricks CLI v0.218+
- Workspace host/token configured
- Existing cluster ID set (`DATABRICKS_CLUSTER_ID`)

Deploy:

```bash
databricks bundle validate
databricks bundle deploy -t dev
databricks bundle run multi_agent_evaluation_job -t dev
```

## Agent Design

- **Supervisor agent**: decides routing (`research`, `math`, `finish`)
- **Research agent**: calls `lookup_kb` tool
- **Math agent**: calls `calculator` tool
- **Synthesizer**: produces final answer from specialist outputs

This gives a production-friendly baseline you can extend with Databricks Vector Search, Unity Catalog functions, and MLflow tracing.

## End-to-End Governance Framework

Governance is applied across the full LLM lifecycle:

- Pre-input checks: detect PII, prompt-injection patterns, secrets, and restricted topics
- Input actions: allow, redact, or block request based on policy
- Inference-stage: only sanitized input reaches the multi-agent graph
- Output checks: detect/redact PII and block secret leakage in responses
- Audit logging: input/output governance events are written to `governance/logs/audit.jsonl`
- Evaluation governance KPIs: blocked rate, PII leak rate, injection detection, and secret output rate

Default policy is in `governance/policy.yaml`:

```yaml
redact_input_pii: true
redact_output_pii: true
block_if_input_has_pii: false
block_prompt_injection: true
block_if_input_has_secrets: true
block_restricted_topics: false
restricted_topics:
	- insider trading
	- malware creation
	- carding
enable_audit_log: true
audit_log_path: governance/logs/audit.jsonl
```

Detectors include:
- Email
- Phone number
- SSN format (`123-45-6789`)
- Credit-card-like digit sequences
- Prompt-injection patterns (instruction override/system prompt extraction attempts)
- Secret patterns (API-key/token-like strings)

Governance-aware response contract is available via `run_query_with_governance`, which returns:
- `answer`
- `governance.blocked`
- `governance.reason`
- Governance counts (`input_pii_count`, `prompt_injection_count`, `secret_input_count`, `output_pii_count`, `secret_output_count`)[DEFAULT]
host = https://your-workspace.cloud.databricks.com
token = your_personal_access_token[DEFAULT]
host = https://your-workspace.cloud.databricks.com
token = your_personal_access_token[DEFAULT]
host = https://your-workspace.cloud.databricks.com
token = your_personal_access_token[DEFAULT]
host = https://your-workspace.cloud.databricks.com
token = your_personal_access_token