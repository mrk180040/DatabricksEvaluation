"""
Unit tests for the multi-agent LangChain framework.

These tests mock the LLM so they run without an OpenAI API key and without
network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agents.tools import calculator, summarize_text, text_statistics, unit_converter, get_agent_tools


# ---------------------------------------------------------------------------
# Tests for tools.py
# ---------------------------------------------------------------------------

class TestCalculatorTool:
    def test_basic_addition(self):
        assert calculator.invoke("2 + 2") == "4"

    def test_power(self):
        assert calculator.invoke("2 ** 10") == "1024"

    def test_math_function(self):
        # Math functions are exposed without the 'math.' prefix in the eval namespace
        result = calculator.invoke("sqrt(144)")
        assert result == "12.0"

    def test_rejects_code_injection(self):
        result = calculator.invoke("__import__('os').getcwd()")
        assert "Error" in result

    def test_division(self):
        assert calculator.invoke("10 / 4") == "2.5"


class TestSummarizeTextTool:
    def test_truncates_to_default_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = summarize_text.invoke({"text": text, "max_sentences": 2})
        assert "First sentence" in result
        assert "Fourth sentence" not in result

    def test_short_text_unchanged(self):
        text = "Only one sentence."
        result = summarize_text.invoke({"text": text, "max_sentences": 3})
        assert result == text


class TestTextStatisticsTool:
    def test_word_count(self):
        result = text_statistics.invoke("Hello world this is a test")
        assert "Words: 6" in result

    def test_sentence_count(self):
        result = text_statistics.invoke("Hello world. This is a test.")
        assert "Sentences: 2" in result


class TestUnitConverterTool:
    def test_km_to_miles(self):
        result = unit_converter.invoke({"value": 100.0, "from_unit": "kilometers", "to_unit": "miles"})
        assert "62.1" in result

    def test_celsius_to_fahrenheit(self):
        result = unit_converter.invoke({"value": 100.0, "from_unit": "celsius", "to_unit": "fahrenheit"})
        assert "212" in result

    def test_kg_to_pounds(self):
        result = unit_converter.invoke({"value": 1.0, "from_unit": "kilograms", "to_unit": "pounds"})
        assert "2.20" in result

    def test_unsupported_conversion(self):
        result = unit_converter.invoke({"value": 1.0, "from_unit": "liters", "to_unit": "cups"})
        assert "Unsupported" in result


class TestGetAgentTools:
    def test_returns_list(self):
        tools = get_agent_tools()
        assert isinstance(tools, list)
        assert len(tools) == 4

    def test_tool_names(self):
        tool_names = {t.name for t in get_agent_tools()}
        assert "calculator" in tool_names
        assert "summarize_text" in tool_names
        assert "text_statistics" in tool_names
        assert "unit_converter" in tool_names


# ---------------------------------------------------------------------------
# Tests for multi_agent.py (graph structure – no LLM calls)
# ---------------------------------------------------------------------------

class TestMultiAgentGraphStructure:
    @patch("agents.multi_agent.ChatOpenAI")
    def test_build_returns_compiled_graph(self, mock_llm_cls):
        """build_multi_agent_graph should return a compiled StateGraph."""
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        # create_react_agent will be called; we need a minimal mock
        with patch("agents.multi_agent.create_react_agent") as mock_react:
            mock_agent = MagicMock()
            # Fake agent.invoke returning a messages list
            mock_agent.invoke.return_value = {
                "messages": [MagicMock(content="ok", name="worker")]
            }
            mock_react.return_value = mock_agent

            from agents.multi_agent import build_multi_agent_graph
            graph = build_multi_agent_graph()
            assert graph is not None

    def test_multi_agent_state_keys(self):
        from agents.multi_agent import MultiAgentState
        import typing
        hints = typing.get_type_hints(MultiAgentState)
        assert "messages" in hints
        assert "next_agent" in hints


# ---------------------------------------------------------------------------
# Tests for evaluation/evaluate.py (mocked Spark + MLflow)
# ---------------------------------------------------------------------------

class TestLoadEvaluationDataset:
    def test_raises_without_spark(self):
        from evaluation.evaluate import load_evaluation_dataset
        with pytest.raises(RuntimeError, match="SparkSession"):
            load_evaluation_dataset("main", "schema", "table", spark=None)

    def test_loads_from_spark(self):
        from evaluation.evaluate import load_evaluation_dataset

        expected_df = pd.DataFrame(
            {"question": ["q1", "q2"], "ground_truth": ["a1", "a2"]}
        )

        mock_spark = MagicMock()
        mock_sdf = MagicMock()
        mock_sdf.toPandas.return_value = expected_df
        mock_spark.sql.return_value = mock_sdf

        result = load_evaluation_dataset("main", "schema", "table", spark=mock_spark)
        pd.testing.assert_frame_equal(result, expected_df)

    def test_limit_appended_to_query(self):
        from evaluation.evaluate import load_evaluation_dataset

        mock_spark = MagicMock()
        mock_sdf = MagicMock()
        mock_sdf.toPandas.return_value = pd.DataFrame({"question": []})
        mock_spark.sql.return_value = mock_sdf

        load_evaluation_dataset("main", "schema", "table", spark=mock_spark, limit=5)

        called_query = mock_spark.sql.call_args[0][0]
        assert "LIMIT 5" in called_query


class TestMLflowWrapper:
    def test_predict_returns_dataframe(self):
        from mlflow_wrapper import MultiAgentWrapper

        wrapper = MultiAgentWrapper()

        mock_graph = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "The answer is 42."
        mock_graph.invoke.return_value = {"messages": [mock_message]}
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
            q = state["messages"][0].content
            return {"messages": [MagicMock(content=f"Answer to: {q}")]}

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = fake_invoke
        wrapper._graph = mock_graph

        input_df = pd.DataFrame({"question": ["Q1", "Q2", "Q3"]})
        output = wrapper.predict(context=None, model_input=input_df)

        assert len(output) == 3
        assert output["answer"].iloc[0] == "Answer to: Q1"
        assert output["answer"].iloc[2] == "Answer to: Q3"
