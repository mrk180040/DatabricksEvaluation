"""
MLflow ``pyfunc`` wrapper for the Databricks multi-agent LangGraph.

Register the graph as a Unity Catalog model::

    import mlflow
    from mlflow_wrapper import MultiAgentWrapper

    mlflow.set_registry_uri("databricks-uc")
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="multi_agent",
            python_model=MultiAgentWrapper(),
            pip_requirements=open("requirements.txt").read().splitlines(),
            registered_model_name="main.agents.databricks_multi_agent",
        )

Load and call it::

    model = mlflow.pyfunc.load_model("models:/main.agents.databricks_multi_agent/1")
    result = model.predict(pd.DataFrame({"question": ["Job failed with OOMError"]}))
    # → DataFrame with column 'answer'
"""

from __future__ import annotations

import os

import pandas as pd
import mlflow
from langchain_core.messages import HumanMessage

from project.agents.graph import build_databricks_agent_graph, initial_state
from project.utils.llm_client import LLMClient, LLMConfig


class MultiAgentWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel wrapper around the Databricks LangGraph multi-agent graph.

    Input schema
    ------------
    ``pandas.DataFrame`` with a single column **``question``** (str).

    Output
    ------
    ``pandas.DataFrame`` with a single column **``answer``** (str).
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:  # noqa: ARG002
        """Build the compiled LangGraph once at model-load time."""
        llm_client = LLMClient(LLMConfig())
        self._graph = build_databricks_agent_graph(llm_client)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,  # noqa: ARG002
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run the multi-agent graph for each row in *model_input*.

        Parameters
        ----------
        model_input : DataFrame with column ``question``.

        Returns
        -------
        DataFrame with column ``answer``.
        """
        if isinstance(model_input, pd.DataFrame):
            questions = model_input["question"].tolist()
        else:
            questions = list(model_input)

        answers = []
        for question in questions:
            state = self._graph.invoke(initial_state(str(question)))
            answers.append(state.get("final_answer", ""))

        return pd.DataFrame({"answer": answers})
