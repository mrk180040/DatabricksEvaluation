from __future__ import annotations

import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

# Add parent directory to path so we can import project module
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from project.evaluation import Evaluator, write_sample_dataset
from project.orchestration import MultiAgentOrchestrator, OrchestratorConfig
from project.utils import LLMClient, LLMConfig


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    llm_config = LLMConfig()

    def build_orchestrator(obo_token: str | None = None) -> MultiAgentOrchestrator:
        llm_client = LLMClient(llm_config, access_token=obo_token)
        return MultiAgentOrchestrator(
            llm_client=llm_client,
            config=OrchestratorConfig(),
        )

    def extract_obo_token() -> str | None:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token:
                return token
        fallback = request.headers.get("X-Databricks-OBO-Token", "").strip()
        return fallback or None

    def selected_token_for_request(obo_token: str | None) -> str | None:
        if obo_token:
            return obo_token
        return os.getenv("DATABRICKS_OBO_TOKEN") or os.getenv("DATABRICKS_TOKEN")

    def token_fingerprint(token: str | None) -> str | None:
        if not token:
            return None
        return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]

    orchestrator = build_orchestrator()
    evaluator = Evaluator(orchestrator=orchestrator)

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
            obo_token = extract_obo_token()
            llm_client = LLMClient(llm_config, access_token=obo_token)
            query_orchestrator = build_orchestrator(obo_token=obo_token)
            result = query_orchestrator.run(query=query, track_with_mlflow=True)
            result.setdefault("trace", {})["auth_source"] = llm_client.auth_source()
            status_code = 200 if result.get("trace", {}).get("status") == "success" else 502
            return jsonify(result), status_code
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
        llm_client = LLMClient(llm_config)
        llm_available = llm_client.available()
        return jsonify(
            {
                "status": "healthy",
                "llm_available": llm_available,
                "auth_source": llm_client.auth_source(),
                "agents": ["job_log_agent", "databricks_add_agent", "unity_catalog_agent"],
            }
        ), 200

    @app.route("/api/whoami", methods=["GET"])
    def api_whoami() -> dict[str, Any]:
        obo_token = extract_obo_token()
        llm_client = LLMClient(llm_config, access_token=obo_token)
        selected_token = selected_token_for_request(obo_token)
        return jsonify(
            {
                "status": "ok",
                "provider": llm_config.provider,
                "model_endpoint": llm_config.model,
                "auth_source": llm_client.auth_source(),
                "token_fingerprint": token_fingerprint(selected_token),
                "header_token_present": bool(obo_token),
            }
        ), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
