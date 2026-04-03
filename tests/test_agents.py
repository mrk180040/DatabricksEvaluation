"""
Unit tests for the Databricks LangChain/LangGraph multi-agent framework.

All LLM calls are mocked so tests run without a Databricks token and without
network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helper: mock ChatDatabricks (BaseChatModel)
# ---------------------------------------------------------------------------

def _mock_chat_model(json_payload: dict):
    """
    Return a mock that behaves like a LangChain ``BaseChatModel`` inside an LCEL chain.

    We use ``RunnableLambda`` so the mock is a proper LangChain ``Runnable`` and
    participates correctly in the ``prompt | model | parser`` chain composition.
    """
    import json
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    def _fake_invoke(messages):
        return AIMessage(content=json.dumps(json_payload))

    return RunnableLambda(_fake_invoke)


# ---------------------------------------------------------------------------
# Tests for project/agents/graph.py (LangGraph structure — no LLM calls)
# ---------------------------------------------------------------------------

class TestDatabricksAgentGraphStructure:
    def test_initial_state_keys(self):
        from project.agents.graph import initial_state
        state = initial_state("test query")
        assert state["query"] == "test query"
        assert len(state["messages"]) == 1
        assert state["selected_agent"] == ""
        assert state["final_answer"] == ""

    def test_build_returns_compiled_graph(self):
        from project.agents.graph import build_databricks_agent_graph
        graph = build_databricks_agent_graph(_mock_chat_model({}))
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        from project.agents.graph import build_databricks_agent_graph
        graph = build_databricks_agent_graph(_mock_chat_model({}))
        node_names = set(graph.get_graph().nodes.keys())
        assert "supervisor" in node_names
        assert "job_log_agent" in node_names
        assert "databricks_add_agent" in node_names
        assert "unity_catalog_agent" in node_names


# ---------------------------------------------------------------------------
# Tests for project/agents/* (individual agent LCEL chains — mocked model)
# ---------------------------------------------------------------------------

class TestSupervisorAgent:
    def test_valid_response_parsed(self):
        from project.agents.supervisor import SupervisorAgent

        agent = SupervisorAgent(
            chat_model=_mock_chat_model({
                "selected_agent": "job_log_agent",
                "reason": "job failure question",
                "confidence": "high",
            })
        )
        result = agent.run("Job failed with OOMError")
        assert result["selected_agent"] == "job_log_agent"
        assert result["confidence"] == "high"

    def test_invalid_agent_raises(self):
        from project.agents.supervisor import SupervisorAgent
        from project.utils.llm_client import LLMResponseFormatError

        agent = SupervisorAgent(
            chat_model=_mock_chat_model({
                "selected_agent": "unknown_agent",
                "reason": "x",
                "confidence": "high",
            })
        )
        with pytest.raises(LLMResponseFormatError):
            agent.run("test")


class TestJobLogAgent:
    def test_valid_response(self):
        from project.agents.job_log_agent import JobLogAgent

        agent = JobLogAgent(
            chat_model=_mock_chat_model({
                "analysis": "OOM in executor",
                "possible_root_cause": "large shuffle",
                "next_steps": ["increase memory", "check logs"],
            })
        )
        result = agent.run("OOMError in task 3")
        assert result["analysis"] == "OOM in executor"
        assert isinstance(result["next_steps"], list)


class TestDatabricksAddAgent:
    def test_valid_response(self):
        from project.agents.databricks_add_agent import DatabricksAddAgent

        agent = DatabricksAddAgent(
            chat_model=_mock_chat_model({
                "action": "create_cluster",
                "parameters": {"num_workers": 4},
                "status": "planned",
            })
        )
        result = agent.run("Create a cluster with 4 workers")
        assert result["action"] == "create_cluster"
        assert result["status"] == "planned"


class TestUnityCatalogAgent:
    def test_valid_response(self):
        from project.agents.unity_catalog_agent import UnityCatalogAgent

        agent = UnityCatalogAgent(
            chat_model=_mock_chat_model({
                "answer": "Use GRANT SELECT ON TABLE ...",
                "entities": ["catalog.schema.table", "analyst_group"],
                "confidence": "high",
            })
        )
        result = agent.run("How do I grant SELECT?")
        assert "GRANT" in result["answer"]
        assert result["confidence"] == "high"


# ---------------------------------------------------------------------------
# Tests for mlflow_wrapper.py
# ---------------------------------------------------------------------------

class TestMultiAgentWrapper:
    def test_predict_returns_dataframe(self):
        from mlflow_wrapper import MultiAgentWrapper

        wrapper = MultiAgentWrapper()
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"final_answer": "The answer is 42.", "selected_agent": "job_log_agent"}
        wrapper._graph = mock_graph

        input_df = pd.DataFrame({"question": ["What is 6 * 7?"]})
        output = wrapper.predict(context=None, model_input=input_df)

        assert isinstance(output, pd.DataFrame)
        assert "answer" in output.columns
        assert output["answer"].iloc[0] == "The answer is 42."

    def test_predict_multiple_rows(self):
        from mlflow_wrapper import MultiAgentWrapper

        wrapper = MultiAgentWrapper()

        def fake_invoke(state):
            q = state["query"]
            return {"final_answer": f"Answer to: {q}", "selected_agent": "unity_catalog_agent"}

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = fake_invoke
        wrapper._graph = mock_graph

        input_df = pd.DataFrame({"question": ["Q1", "Q2", "Q3"]})
        output = wrapper.predict(context=None, model_input=input_df)

        assert len(output) == 3
        assert output["answer"].iloc[0] == "Answer to: Q1"
        assert output["answer"].iloc[2] == "Answer to: Q3"


# ---------------------------------------------------------------------------
# Tests for project/evaluation (dataset loading, metrics)
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_file_not_found(self, tmp_path):
        from project.evaluation.dataset import load_dataset
        with pytest.raises(FileNotFoundError):
            load_dataset(str(tmp_path / "missing.json"))

    def test_loads_json_list(self, tmp_path):
        import json
        from project.evaluation.dataset import load_dataset

        data = [{"query": "q1", "expected_agent": "job_log_agent"}]
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data))

        result = load_dataset(str(p))
        assert len(result) == 1
        assert result[0]["query"] == "q1"

    def test_loads_jsonl(self, tmp_path):
        import json
        from project.evaluation.dataset import load_dataset

        rows = [
            {"query": "q1", "expected_agent": "job_log_agent"},
            {"query": "q2", "expected_agent": "unity_catalog_agent"},
        ]
        p = tmp_path / "data.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows))

        result = load_dataset(str(p))
        assert len(result) == 2


class TestMetrics:
    def test_routing_accuracy_correct(self):
        from project.evaluation.metrics import routing_accuracy
        assert routing_accuracy("job_log_agent", "job_log_agent") == 1.0

    def test_routing_accuracy_wrong(self):
        from project.evaluation.metrics import routing_accuracy
        assert routing_accuracy("job_log_agent", "unity_catalog_agent") == 0.0

    def test_keyword_match_all_present(self):
        from project.evaluation.metrics import keyword_match_score
        assert keyword_match_score("analysis root cause memory", ["analysis", "memory"]) == 1.0

    def test_keyword_match_partial(self):
        from project.evaluation.metrics import keyword_match_score
        score = keyword_match_score("only memory here", ["analysis", "memory"])
        assert score == 0.5

    def test_keyword_match_empty_keywords(self):
        from project.evaluation.metrics import keyword_match_score
        assert keyword_match_score("anything", []) == 1.0

