"""
Multi-agent LangChain framework using LangGraph.

Architecture
------------
A *supervisor* agent receives user requests and routes them to one of the
specialist *worker* agents:

  • ResearchAgent  – answers factual / knowledge questions
  • MathAgent      – handles numerical and unit-conversion problems
  • SummaryAgent   – summarises or analyses text

The supervisor decides which agent to call next (or FINISH when done) based on
the accumulated conversation messages.  Each worker can use the shared tool set
defined in ``agents/tools.py``.

                 ┌──────────────┐
  user input ──► │  Supervisor  │ ──► FINISH ──► final answer
                 └──────┬───────┘
                        │ routes to
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   ResearchAgent   MathAgent   SummaryAgent
"""

from __future__ import annotations

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from agents.tools import get_agent_tools, calculator, unit_converter, summarize_text, text_statistics

# ---------------------------------------------------------------------------
# Shared state definition
# ---------------------------------------------------------------------------

class MultiAgentState(TypedDict):
    """State that flows through every node in the graph."""

    # Accumulated conversation messages (append-only via the operator.add reducer)
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Which agent the supervisor chose to call next
    next_agent: str


# ---------------------------------------------------------------------------
# Agent / node definitions
# ---------------------------------------------------------------------------

WORKER_NAMES = ["ResearchAgent", "MathAgent", "SummaryAgent"]
FINISH = "FINISH"
ROUTING_OPTIONS = WORKER_NAMES + [FINISH]


def _make_llm(model: str = "gpt-4o-mini", temperature: float = 0) -> ChatOpenAI:
    """Instantiate a ChatOpenAI model (uses OPENAI_API_KEY from the environment)."""
    return ChatOpenAI(model=model, temperature=temperature)


# --- Supervisor node --------------------------------------------------------

_SUPERVISOR_SYSTEM = (
    "You are a supervisor managing a team of specialist agents.\n"
    "Given a conversation, decide which agent should act next, or reply FINISH "
    "if the request has been fully answered.\n\n"
    "Available agents:\n"
    "  • ResearchAgent  – factual questions, definitions, explanations\n"
    "  • MathAgent      – arithmetic, algebra, unit conversions\n"
    "  • SummaryAgent   – text summarisation and text statistics\n\n"
    "Reply with exactly one of: ResearchAgent, MathAgent, SummaryAgent, FINISH"
)


def make_supervisor_node(llm: ChatOpenAI):
    """Return the supervisor node function."""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=_SUPERVISOR_SYSTEM),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(
                content=(
                    "Based on the conversation above, who should act next? "
                    f"Choose one of: {', '.join(ROUTING_OPTIONS)}"
                )
            ),
        ]
    )
    chain = prompt | llm

    def supervisor_node(state: MultiAgentState) -> dict:
        response = chain.invoke({"messages": state["messages"]})
        chosen = response.content.strip()
        # Normalise / guard against unexpected output
        if chosen not in ROUTING_OPTIONS:
            chosen = FINISH
        return {"next_agent": chosen}

    return supervisor_node


# --- Worker nodes -----------------------------------------------------------

_RESEARCH_SYSTEM = (
    "You are a knowledgeable research assistant.  Answer factual questions "
    "clearly and concisely.  Use tools when they help you produce a better answer."
)

_MATH_SYSTEM = (
    "You are a precise mathematics assistant.  Solve numerical problems and "
    "unit conversions step-by-step using the available tools."
)

_SUMMARY_SYSTEM = (
    "You are an expert text analyst.  Summarise text or produce statistics "
    "about it using the available tools."
)


def _make_worker_node(name: str, system_prompt: str, tools: list, llm: ChatOpenAI):
    """Wrap a ReAct agent in a LangGraph node function."""
    agent = create_react_agent(llm, tools=tools, state_modifier=system_prompt)

    def worker_node(state: MultiAgentState) -> dict:
        result = agent.invoke(state)
        # The ReAct agent returns a dict with a 'messages' key; we surface the
        # last AI message back as a new message so the supervisor sees it.
        last_message = result["messages"][-1]
        last_message.name = name
        return {"messages": [last_message]}

    return worker_node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_multi_agent_graph(
    model: str = "gpt-4o-mini",
    temperature: float = 0,
) -> StateGraph:
    """
    Build and compile the multi-agent LangGraph graph.

    Parameters
    ----------
    model       : OpenAI model name (default ``"gpt-4o-mini"``).
    temperature : Sampling temperature (default ``0``).

    Returns
    -------
    A compiled LangGraph ``CompiledGraph`` that can be invoked with::

        graph.invoke({"messages": [HumanMessage(content="What is 2 ** 32?")]})
    """
    llm = _make_llm(model=model, temperature=temperature)
    all_tools = get_agent_tools()

    # Build the graph
    builder = StateGraph(MultiAgentState)

    # Supervisor
    supervisor_node = make_supervisor_node(llm)
    builder.add_node("supervisor", supervisor_node)

    # Workers – each gets the full tool set; you can give each a subset too
    builder.add_node(
        "ResearchAgent",
        _make_worker_node("ResearchAgent", _RESEARCH_SYSTEM, all_tools, llm),
    )
    builder.add_node(
        "MathAgent",
        _make_worker_node("MathAgent", _MATH_SYSTEM, [calculator, unit_converter], llm),
    )
    builder.add_node(
        "SummaryAgent",
        _make_worker_node("SummaryAgent", _SUMMARY_SYSTEM, [summarize_text, text_statistics], llm),
    )

    # Edges – supervisor always runs first, then routes to a worker or END
    builder.set_entry_point("supervisor")

    # Use Union[str] to avoid hard-coding agent names a second time here.
    def _route(state: MultiAgentState) -> str:
        return state["next_agent"] if state["next_agent"] != FINISH else END

    builder.add_conditional_edges("supervisor", _route)

    # After each worker, return to the supervisor
    for worker in WORKER_NAMES:
        builder.add_edge(worker, "supervisor")

    return builder.compile()


# ---------------------------------------------------------------------------
# Convenience: run the graph and return the final answer string
# ---------------------------------------------------------------------------

def run_multi_agent(question: str, model: str = "gpt-4o-mini") -> str:
    """
    End-to-end helper: build the graph, run it with *question*, and return the
    final answer as a plain string.

    Parameters
    ----------
    question : The user question or task.
    model    : OpenAI model name.

    Returns
    -------
    The last AI message content as a string.
    """
    graph = build_multi_agent_graph(model=model)
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content
