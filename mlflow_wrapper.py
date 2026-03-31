"""
MLflow Python model wrapper for the multi-agent framework.

This wrapper lets you log the multi-agent graph with ``mlflow.pyfunc.log_model``
and later load it for batch inference or as a Databricks Model-Serving endpoint.

Usage
-----
Register the model::

    import mlflow
    from mlflow_wrapper import MultiAgentWrapper

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="multi_agent",
            python_model=MultiAgentWrapper(),
            pip_requirements=open("requirements.txt").read().splitlines(),
        )

Load and call it::

    model = mlflow.pyfunc.load_model("runs:/<run_id>/multi_agent")
    result = model.predict(pd.DataFrame({"question": ["What is 2 + 2?"]}))
"""

from __future__ import annotations

import pandas as pd
import mlflow

from agents.multi_agent import build_multi_agent_graph, run_multi_agent
from langchain_core.messages import HumanMessage


class MultiAgentWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel wrapper around the LangGraph multi-agent graph.

    Input schema
    ------------
    ``pandas.DataFrame`` with a single column **``question``** (str).

    Output
    ------
    ``pandas.DataFrame`` with a single column **``answer``** (str).
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:  # noqa: ARG002
        """Build the compiled graph once at load time."""
        self._graph = build_multi_agent_graph()

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
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=str(question))]}
            )
            answers.append(result["messages"][-1].content)

        return pd.DataFrame({"answer": answers})
