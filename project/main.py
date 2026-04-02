from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path so we can import project module
sys.path.insert(0, str(Path(__file__).parent.parent))

from project.evaluation import Evaluator, write_sample_dataset
from project.orchestration import MultiAgentOrchestrator, OrchestratorConfig
from project.utils import LLMClient, LLMConfig


def build_runtime(obo_token: str | None = None) -> tuple[MultiAgentOrchestrator, Evaluator]:
    llm_client = LLMClient(LLMConfig(), access_token=obo_token)
    orchestrator = MultiAgentOrchestrator(
        llm_client=llm_client,
        config=OrchestratorConfig(
            confidence_threshold="medium",
            fallback_agent="unity_catalog_agent",
            mlflow_experiment="databricks-multi-agent",
        ),
    )
    evaluator = Evaluator(orchestrator=orchestrator)
    return orchestrator, evaluator


def run_query(query: str, obo_token: str | None = None) -> None:
    orchestrator, _ = build_runtime(obo_token=obo_token)
    result = orchestrator.run(query=query, track_with_mlflow=True)
    print(json.dumps(result, indent=2))


def run_evaluation(dataset_path: str, output_path: str, obo_token: str | None = None) -> None:
    orchestrator, evaluator = build_runtime(obo_token=obo_token)

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        write_sample_dataset(dataset_path)

    summary = evaluator.evaluate(dataset_path=dataset_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Databricks multi-agent orchestrator with MLflow evaluation")
    parser.add_argument("--mode", choices=["run", "evaluate"], default="run")
    parser.add_argument("--query", default="How do I grant SELECT on catalog.sales.orders to analyst_group?")
    parser.add_argument("--dataset", default="project/data/sample_dataset.json")
    parser.add_argument("--output", default="project/data/evaluation_results.json")
    parser.add_argument("--obo-token", default=None, help="Delegated Databricks access token for On-Behalf-Of flow")
    args = parser.parse_args()

    if args.mode == "run":
        run_query(args.query, obo_token=args.obo_token)
        return
    run_evaluation(dataset_path=args.dataset, output_path=args.output, obo_token=args.obo_token)


if __name__ == "__main__":
    main()
