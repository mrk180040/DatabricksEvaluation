"""
AgentBricks Deployment — Unity Catalog registration + databricks.agents.deploy().

Workflow
--------
1. Log the AgentBricksModel as an MLflow artifact via log_agentbricks_model().
2. Register the logged artifact in Unity Catalog with deployer.register().
3. Deploy to a Databricks serving endpoint with deployer.deploy().
4. Optionally open the Databricks Review App.

Requirements
------------
  pip install databricks-agents>=0.5.0

Environment variables
---------------------
  DATABRICKS_HOST   — e.g. https://adb-xxx.azuredatabricks.net
  DATABRICKS_TOKEN  — personal access token
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from project.utils.logger import get_logger

logger = get_logger("agentbricks.deployer")


@dataclass
class AgentBricksDeployConfig:
    # Unity Catalog 3-part name: catalog.schema.model_name
    uc_model_name: str
    # Scaling / endpoint settings
    environment_vars: dict[str, str] | None = None
    scale_to_zero: bool = True
    # Optional tags forwarded to the registered model version
    version_tags: dict[str, str] | None = None


class AgentBricksDeployer:
    """
    Handles the full Databricks Agent Framework deploy lifecycle:

      register() → registers the MLflow run as a UC model version
      deploy()   → calls databricks.agents.deploy() to create/update
                   the serving endpoint + review app
      get_review_app_url() → returns the review app URL for human eval
    """

    def __init__(self, model_uri: str, config: AgentBricksDeployConfig) -> None:
        self.model_uri = model_uri
        self.config = config

    # ------------------------------------------------------------------
    # Step 1 — Register in Unity Catalog
    # ------------------------------------------------------------------

    def register(self, description: str = "AgentBricks multi-agent orchestrator") -> int:
        """
        Register the model artifact in Unity Catalog.

        Returns the registered model version number.

        Example
        -------
        deployer = AgentBricksDeployer(
            model_uri="runs:/abc123/agentbricks_agent",
            config=AgentBricksDeployConfig(uc_model_name="main.agents.databricks_agent"),
        )
        version = deployer.register()
        """
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("mlflow is required for registration.") from exc

        mlflow.set_registry_uri("databricks-uc")

        logger.info(
            "Registering model %s → %s",
            self.model_uri,
            self.config.uc_model_name,
        )

        mv = mlflow.register_model(
            model_uri=self.model_uri,
            name=self.config.uc_model_name,
        )
        version = int(mv.version)

        client = mlflow.tracking.MlflowClient()
        client.update_model_version(
            name=self.config.uc_model_name,
            version=str(version),
            description=description,
        )

        if self.config.version_tags:
            for tag_key, tag_val in self.config.version_tags.items():
                client.set_model_version_tag(
                    name=self.config.uc_model_name,
                    version=str(version),
                    key=tag_key,
                    value=tag_val,
                )

        logger.info("Registered as version %d", version)
        return version

    # ------------------------------------------------------------------
    # Step 2 — Deploy to Databricks serving endpoint
    # ------------------------------------------------------------------

    def deploy(self, version: int) -> Any:
        """
        Deploy the registered UC model version using the
        ``databricks-agents`` SDK.

        Returns the DeploymentInfo object with ``endpoint_url``,
        ``review_app_url``, and ``query_endpoint``.

        Requires: pip install databricks-agents>=0.5.0
        """
        try:
            from databricks.agents import deploy as agents_deploy  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "databricks-agents>=0.5.0 is required for deployment.\n"
                "Install via: pip install databricks-agents"
            ) from exc

        env_vars = self.config.environment_vars or self._default_env_vars()

        logger.info(
            "Deploying %s version %d to Databricks serving…",
            self.config.uc_model_name,
            version,
        )

        deployment_info = agents_deploy(
            model_name=self.config.uc_model_name,
            model_version=version,
            scale_to_zero_enabled=self.config.scale_to_zero,
            environment_vars=env_vars,
        )

        logger.info("Deployed. Endpoint: %s", getattr(deployment_info, "endpoint_url", "unknown"))
        return deployment_info

    # ------------------------------------------------------------------
    # Step 3 — Review App URL (human evaluation)
    # ------------------------------------------------------------------

    def get_review_app_url(self, version: int) -> str:
        """
        Returns the URL for the Databricks Review App where human
        reviewers can rate agent responses.

        Requires: pip install databricks-agents>=0.5.0
        """
        try:
            from databricks.agents import get_review_app_url  # type: ignore[import-untyped]
            return str(get_review_app_url(
                model_name=self.config.uc_model_name,
                model_version=version,
            ))
        except ImportError:
            logger.warning("databricks-agents not installed; cannot get review app URL.")
            host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
            return f"{host}/#mlflow/models/{self.config.uc_model_name}/versions/{version}"
        except Exception as exc:
            logger.warning("get_review_app_url failed: %s", exc)
            return "unknown"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_env_vars() -> dict[str, str]:
        env: dict[str, str] = {}
        for key in (
            "DATABRICKS_HOST",
            "DATABRICKS_TOKEN",
            "DATABRICKS_MODEL_ENDPOINT",
            "LLM_PROVIDER",
            "LLM_TEMPERATURE",
            "LLM_MAX_TOKENS",
        ):
            val = os.getenv(key)
            if val:
                env[key] = val
        return env
