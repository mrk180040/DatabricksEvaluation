from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path so we can import project module
sys.path.insert(0, str(Path(__file__).parent.parent))

from project.agentbricks.deployment import AgentBricksDeployConfig, AgentBricksDeployer
from project.agentbricks.evaluator import AgentBricksEvalConfig, AgentBricksEvaluator
from project.agentbricks.runtime import AgentBricksModel, log_agentbricks_model
from project.evaluation import Evaluator, write_sample_dataset
from project.orchestration import AgentBricksConfig, AgentBricksOrchestrator
from project.utils import LLMClient, LLMConfig


def build_runtime() -> tuple[AgentBricksOrchestrator, Evaluator]:
    llm_client = LLMClient(LLMConfig())
    orchestrator = AgentBricksOrchestrator(
        llm_client=llm_client,
        config=AgentBricksConfig(
            confidence_threshold="medium",
            fallback_agent="unity_catalog_agent",
            mlflow_experiment="databricks-agentbricks",
        ),
    )
    evaluator = Evaluator(orchestrator=orchestrator)
    return orchestrator, evaluator


def build_agentbricks_runtime() -> tuple[AgentBricksOrchestrator, AgentBricksEvaluator]:
    """Build the runtime wired to the real AgentBricks evaluator (mlflow.evaluate + judges)."""
    llm_client = LLMClient(LLMConfig())
    orchestrator = AgentBricksOrchestrator(
        llm_client=llm_client,
        config=AgentBricksConfig(
            confidence_threshold="medium",
            fallback_agent="unity_catalog_agent",
            mlflow_experiment="databricks-agentbricks",
        ),
    )
    evaluator = AgentBricksEvaluator(
        orchestrator=orchestrator,
        config=AgentBricksEvalConfig(use_databricks_judges=True),
    )
    return orchestrator, evaluator


def run_query(query: str) -> None:
    orchestrator, _ = build_runtime()
    result = orchestrator.run(query=query, track_with_mlflow=True)
    print(json.dumps(result, indent=2))


def run_evaluation(dataset_path: str, output_path: str) -> None:
    orchestrator, evaluator = build_runtime()

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        write_sample_dataset(dataset_path)

    summary = evaluator.evaluate(dataset_path=dataset_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_agentbricks_evaluation(dataset_path: str, output_path: str) -> None:
    """Evaluation via mlflow.evaluate() with Databricks Judge LLMs."""
    orchestrator, evaluator = build_agentbricks_runtime()

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        write_sample_dataset(dataset_path)

    summary = evaluator.evaluate(dataset_path=dataset_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_register(uc_model_name: str) -> None:
    """
    Log the AgentBricksModel to MLflow and register it in Unity Catalog.

    Requires: DATABRICKS_HOST, DATABRICKS_TOKEN
    """
    import mlflow
    from mlflow.entities.model_registry import RegisteredModel

    print(f"Logging AgentBricksModel → Unity Catalog: {uc_model_name}")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment("databricks-agentbricks")

    # Delete old versions if they exist (to clear any invalid schemas)
    try:
        client = mlflow.tracking.MlflowClient()
        old_model = client.get_registered_model(uc_model_name)
        if old_model and hasattr(old_model, 'latest_versions') and old_model.latest_versions:
            print(f"Found existing model with {len(old_model.latest_versions)} versions. Deleting old versions...")
            for version in old_model.latest_versions:
                try:
                    client.delete_model_version(uc_model_name, version.version)
                except Exception:
                    pass  # Version may already be deleted
            client.delete_registered_model(uc_model_name)
            print(f"Deleted old model registration for {uc_model_name}")
    except Exception as e:
        print(f"No old model to delete (expected on first run): {type(e).__name__}: {e}")

    model = AgentBricksModel()
    with mlflow.start_run(run_name="agentbricks_register"):
        model_uri = log_agentbricks_model(model=model)
        deployer = AgentBricksDeployer(
            model_uri=model_uri,
            config=AgentBricksDeployConfig(uc_model_name=uc_model_name),
        )
        version = deployer.register()
    print(f"Registered as version {version}")


def run_deploy(uc_model_name: str, version: int) -> None:
    """
    Deploy a registered UC model version to a Databricks serving endpoint.

    Requires: pip install databricks-agents>=0.5.0
    """
    deployer = AgentBricksDeployer(
        model_uri="",  # not needed for deploy-only
        config=AgentBricksDeployConfig(uc_model_name=uc_model_name),
    )
    info = deployer.deploy(version)
    review_url = deployer.get_review_app_url(version)
    print(f"Deployed: {getattr(info, 'endpoint_url', info)}")
    print(f"Review app: {review_url}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Databricks AgentBricks orchestrator")
    parser.add_argument(
        "--mode",
        choices=["run", "evaluate", "agentbricks-evaluate", "register", "deploy"],
        default="run",
        help=(
            "run                  → execute a single query\n"
            "evaluate             → custom metric evaluation (no LLM judges)\n"
            "agentbricks-evaluate → mlflow.evaluate() with Databricks Judge LLMs\n"
            "register             → log + register model in Unity Catalog\n"
            "deploy               → deploy registered UC model version to serving\n"
        ),
    )
    parser.add_argument("--query", default="How do I grant SELECT on catalog.sales.orders to analyst_group?")
    parser.add_argument("--dataset", default="project/data/sample_dataset.json")
    parser.add_argument("--output", default="project/data/evaluation_results.json")
    parser.add_argument("--uc-model", default="main.agents.databricks_agentbricks",
                        help="Unity Catalog model name (catalog.schema.model)")
    parser.add_argument("--version", type=int, default=1,
                        help="Registered model version (for --mode deploy)")
    args = parser.parse_args()

    if args.mode == "run":
        run_query(args.query)
    elif args.mode == "evaluate":
        run_evaluation(dataset_path=args.dataset, output_path=args.output)
    elif args.mode == "agentbricks-evaluate":
        run_agentbricks_evaluation(dataset_path=args.dataset, output_path=args.output)
    elif args.mode == "register":
        run_register(uc_model_name=args.uc_model)
    elif args.mode == "deploy":
        run_deploy(uc_model_name=args.uc_model, version=args.version)


if __name__ == "__main__":
    main()
