from __future__ import annotations

import json
from pathlib import Path


def load_dataset(path: str) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if file_path.suffix == ".jsonl":
        return [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    return json.loads(file_path.read_text(encoding="utf-8"))


def write_sample_dataset(path: str) -> None:
    samples = [
        {
            "query": "Job run failed with java.lang.OutOfMemoryError in task 3. What should I check?",
            "expected_agent": "job_log_agent",
            "expected_keywords": ["memory", "logs", "runtime"],
        },
        {
            "query": "My nightly ETL job keeps retrying and then fails after 2 hours.",
            "expected_agent": "job_log_agent",
            "expected_keywords": ["retry", "logs", "dependencies"],
        },
        {
            "query": "The workflow task throws ModuleNotFoundError: pandas on one cluster only.",
            "expected_agent": "job_log_agent",
            "expected_keywords": ["dependency", "cluster", "runtime"],
        },
        {
            "query": "Add a new user to Databricks workspace and grant access to a shared cluster.",
            "expected_agent": "databricks_add_agent",
            "expected_keywords": ["action", "parameters", "planned"],
        },
        {
            "query": "Create a new job for daily notebook execution at 6 AM.",
            "expected_agent": "databricks_add_agent",
            "expected_keywords": ["action", "schedule", "planned"],
        },
        {
            "query": "Provision a cluster with Photon enabled for data engineering team.",
            "expected_agent": "databricks_add_agent",
            "expected_keywords": ["cluster", "parameters", "planned"],
        },
        {
            "query": "How do I grant SELECT privilege on catalog.sales.orders to analyst_group?",
            "expected_agent": "unity_catalog_agent",
            "expected_keywords": ["privilege", "catalog", "table"],
        },
        {
            "query": "List required privileges to create external tables in Unity Catalog.",
            "expected_agent": "unity_catalog_agent",
            "expected_keywords": ["unity", "privilege", "entities"],
        },
        {
            "query": "What is the difference between managed and external tables in Unity Catalog?",
            "expected_agent": "unity_catalog_agent",
            "expected_keywords": ["unity", "catalog", "table"],
        },
        {
            "query": "A job fails only in production when reading from s3://raw bucket.",
            "expected_agent": "job_log_agent",
            "expected_keywords": ["analysis", "root", "next"],
        },
    ]

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(samples, indent=2), encoding="utf-8")
