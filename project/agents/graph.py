"""
LangGraph multi-agent graph for Databricks operations.

Architecture
------------
A supervisor node receives the user query and routes it to one of three
specialist worker nodes:

  • job_log_agent       – Databricks job failure analysis
  • databricks_add_agent – Platform provisioning / configuration planning
  • unity_catalog_agent  – Unity Catalog governance and metadata Q&A

                  ┌──────────────┐
   user query ──► │  Supervisor  │
                  └──────┬───────┘
                         │  conditional routing
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   job_log_agent  databricks_add  unity_catalog
         │               │               │
         └───────────────┴───────────────┘
                         │
                        END

Each worker executes exactly once per query (single-turn).  All nodes share
a typed ``DatabricksAgentState`` that accumulates LangChain messages so the
full conversation is available to every downstream node and to MLflow tracing.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langgraph.graph import END, StateGraph

from project.agents.databricks_add_agent import DatabricksAddAgent
from project.agents.job_log_agent import JobLogAgent
from project.agents.supervisor import SupervisorAgent
from project.agents.unity_catalog_agent import UnityCatalogAgent
from project.utils.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class DatabricksAgentState(TypedDict):
    """State that flows through every node in the LangGraph."""

    # Original user query (immutable across nodes)
    query: str
    # Accumulated LangChain messages (append-only via operator.add)
    messages: Annotated[list[BaseMessage], operator.add]
    # Supervisor routing decision
    selected_agent: str
    confidence: str
    reason: str
    # Worker output
    agent_response: dict
    final_answer: str


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

_FALLBACK_AGENT = "unity_catalog_agent"
_VALID_AGENTS = {"job_log_agent", "databricks_add_agent", "unity_catalog_agent"}


def build_databricks_agent_graph(llm_client: LLMClient):
    """
    Build and compile the Databricks multi-agent LangGraph.

    Parameters
    ----------
    llm_client:
        A configured ``LLMClient`` instance.  The underlying chat model is
        derived via ``llm_client.as_langchain_chat_model()``.

    Returns
    -------
    A compiled LangGraph ``CompiledGraph`` ready to be invoked with an
    initial ``DatabricksAgentState``::

        graph.invoke({
            "query": "Job failed with OOMError",
            "messages": [HumanMessage(content="Job failed with OOMError")],
            "selected_agent": "",
            "confidence": "",
            "reason": "",
            "agent_response": {},
            "final_answer": "",
        })
    """
    supervisor = SupervisorAgent(llm_client=llm_client)
    job_log = JobLogAgent(llm_client=llm_client)
    databricks_add = DatabricksAddAgent(llm_client=llm_client)
    unity_catalog = UnityCatalogAgent(llm_client=llm_client)

    # --- nodes --------------------------------------------------------------

    def supervisor_node(state: DatabricksAgentState) -> dict:
        result = supervisor.run(state["query"])
        routing_msg = (
            f"Routing to {result['selected_agent']} "
            f"(confidence={result['confidence']}): {result['reason']}"
        )
        return {
            "selected_agent": result["selected_agent"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "messages": [AIMessage(content=routing_msg, name="supervisor")],
        }

    def job_log_node(state: DatabricksAgentState) -> dict:
        result = job_log.run(state["query"])
        answer = result.get("analysis", str(result))
        return {
            "agent_response": result,
            "final_answer": answer,
            "messages": [AIMessage(content=answer, name="job_log_agent")],
        }

    def databricks_add_node(state: DatabricksAgentState) -> dict:
        result = databricks_add.run(state["query"])
        answer = result.get("action", str(result))
        return {
            "agent_response": result,
            "final_answer": answer,
            "messages": [AIMessage(content=answer, name="databricks_add_agent")],
        }

    def unity_catalog_node(state: DatabricksAgentState) -> dict:
        result = unity_catalog.run(state["query"])
        answer = result.get("answer", str(result))
        return {
            "agent_response": result,
            "final_answer": answer,
            "messages": [AIMessage(content=answer, name="unity_catalog_agent")],
        }

    # --- routing ------------------------------------------------------------

    def route_to_agent(state: DatabricksAgentState) -> str:
        agent = state.get("selected_agent", _FALLBACK_AGENT)
        return agent if agent in _VALID_AGENTS else _FALLBACK_AGENT

    # --- graph assembly -----------------------------------------------------

    graph = StateGraph(DatabricksAgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("job_log_agent", job_log_node)
    graph.add_node("databricks_add_agent", databricks_add_node)
    graph.add_node("unity_catalog_agent", unity_catalog_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "job_log_agent": "job_log_agent",
            "databricks_add_agent": "databricks_add_agent",
            "unity_catalog_agent": "unity_catalog_agent",
        },
    )

    graph.add_edge("job_log_agent", END)
    graph.add_edge("databricks_add_agent", END)
    graph.add_edge("unity_catalog_agent", END)

    return graph.compile()


def initial_state(query: str) -> DatabricksAgentState:
    """Return a zeroed ``DatabricksAgentState`` ready to be passed to ``graph.invoke``."""
    return DatabricksAgentState(
        query=query,
        messages=[HumanMessage(content=query)],
        selected_agent="",
        confidence="",
        reason="",
        agent_response={},
        final_answer="",
    )
