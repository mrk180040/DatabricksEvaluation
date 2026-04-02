from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

# Add parent directory to path so we can import project module
sys.path.insert(0, str(Path(__file__).parent.parent))

from project.evaluation import write_sample_dataset
from project.agentbricks.evaluator import AgentBricksEvalConfig, AgentBricksEvaluator
from project.orchestration import AgentBricksConfig, AgentBricksOrchestrator
from project.utils import LLMClient, LLMConfig


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    llm_config = LLMConfig()
    llm_client = LLMClient(llm_config)
    orchestrator = AgentBricksOrchestrator(
        llm_client=llm_client,
        config=AgentBricksConfig(),
    )
    evaluator = AgentBricksEvaluator(
        orchestrator=orchestrator,
        config=AgentBricksEvalConfig(use_databricks_judges=True),
    )

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/api/query", methods=["POST"])
    def api_query() -> dict[str, Any]:
        payload = request.get_json() or {}
        query = payload.get("query", "")
        if not query:
            return jsonify({"error": "query is required"}), 400

        try:
            result = orchestrator.run(query=query, track_with_mlflow=True)
            return jsonify(result), 200
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/evaluate", methods=["POST"])
    def api_evaluate() -> dict[str, Any]:
        payload = request.get_json() or {}
        dataset_path = payload.get("dataset_path", "project/data/sample_dataset.json")
        output_path = payload.get("output_path", "project/data/evaluation_results.json")

        try:
            summary = evaluator.evaluate(dataset_path=dataset_path)
            return jsonify(summary), 200
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/status", methods=["GET"])
    def api_status() -> dict[str, Any]:
        return jsonify(
            {
                "status": "healthy",
                "llm_available": llm_client.available(),
                "agents": ["job_log_agent", "databricks_add_agent", "unity_catalog_agent"],
            }
        ), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8080, debug=False)
