"""
Databricks LangChain model factory.

This module is the single place in the project where the LangChain
``ChatDatabricks`` model is constructed.  Every agent and graph node
imports ``make_chat_model()`` from here rather than instantiating
``ChatDatabricks`` directly, so configuration is centralised.

``ChatDatabricks`` is the official Databricks LangChain integration
(``langchain-databricks`` package).  It uses the Databricks SDK auth
chain automatically — no manual token wiring required when running inside
a Databricks App or a Databricks notebook.

When running outside Databricks (e.g. local development), set:
  DATABRICKS_HOST  — your workspace URL
  DATABRICKS_TOKEN — a personal access token or OBO token

Usage::

    from project.utils.databricks_llm import make_chat_model

    llm = make_chat_model()
    response = llm.invoke("Hello")

    # Or in an LCEL chain:
    from langchain_core.prompts import ChatPromptTemplate
    chain = ChatPromptTemplate.from_messages([("system", "..."), ("human", "{input}")]) | llm
    result = chain.invoke({"input": "What is Unity Catalog?"})
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel


def make_chat_model(
    endpoint: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """
    Return a ``ChatDatabricks`` instance configured from environment variables.

    Parameters
    ----------
    endpoint:
        Databricks Foundation Model API or Model Serving endpoint name.
        Defaults to ``DATABRICKS_MODEL_ENDPOINT`` env var, then
        ``"databricks-meta-llama-3-3-70b-instruct"``.
    temperature:
        Sampling temperature.  Defaults to ``LLM_TEMPERATURE`` env var (0.1).
    max_tokens:
        Maximum tokens in the response.  Defaults to ``LLM_MAX_TOKENS`` (700).

    Returns
    -------
    A ``ChatDatabricks`` instance ready to use in LCEL chains or LangGraph nodes.
    """
    from langchain_databricks import ChatDatabricks

    _endpoint = endpoint or os.getenv(
        "DATABRICKS_MODEL_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"
    )
    _temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.1"))
    _max_tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "700"))

    return ChatDatabricks(
        endpoint=_endpoint,
        temperature=_temperature,
        max_tokens=_max_tokens,
    )
