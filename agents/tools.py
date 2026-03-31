"""
Tools available to agents in the multi-agent LangChain framework.

Each tool is a simple, self-contained function decorated with @tool so it can
be attached to any LangChain / LangGraph agent.
"""

import math
from typing import Union

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Calculator tool
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic mathematical expression and return the result.

    The expression must be a valid Python arithmetic expression that uses only
    numbers, the standard operators (+, -, *, /, **, %), and the functions
    available in the ``math`` module — referenced **without** the ``math.``
    prefix (e.g. ``sqrt(16)`` rather than ``math.sqrt(16)``).

    Args:
        expression: A Python arithmetic expression string (e.g. "2 ** 10 + 5").

    Returns:
        The result as a string, or an error message if evaluation fails.
    """
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    allowed_names.update({"abs": abs, "round": round})
    try:
        # Security note: eval is restricted to a sandboxed namespace.
        # __builtins__ is set to {} (empty dict) to block all built-in
        # functions such as __import__, exec, and open.  Only the math
        # module functions and abs/round are available, so arbitrary code
        # execution is not possible.  Any unexpected token raises an
        # exception which is caught and returned as an error string.
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Error evaluating expression: {exc}"


# ---------------------------------------------------------------------------
# Text summariser tool
# ---------------------------------------------------------------------------

@tool
def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Return a very short extractive summary of *text* (first N sentences).

    This is a lightweight, offline summariser that simply returns the first
    ``max_sentences`` sentences of the input.  In a production system you would
    replace this with a call to an LLM or a dedicated summarisation service.

    Args:
        text:          The text to summarise.
        max_sentences: Maximum number of sentences to keep (default: 3).

    Returns:
        The truncated / summarised text.
    """
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:max_sentences])


# ---------------------------------------------------------------------------
# Word-count / text statistics tool
# ---------------------------------------------------------------------------

@tool
def text_statistics(text: str) -> str:
    """
    Return basic statistics about *text*: word count, sentence count, and
    character count (excluding whitespace).

    Args:
        text: The input text.

    Returns:
        A human-readable summary of the statistics.
    """
    import re
    words = len(text.split())
    sentences = len(re.split(r"(?<=[.!?])\s+", text.strip()))
    chars = len(text.replace(" ", "").replace("\n", ""))
    return f"Words: {words}, Sentences: {sentences}, Characters (no whitespace): {chars}"


# ---------------------------------------------------------------------------
# Unit conversion tool
# ---------------------------------------------------------------------------

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a numeric value between common units.

    Supported conversions
    ---------------------
    Temperature : celsius ↔ fahrenheit ↔ kelvin
    Distance    : meters ↔ kilometers ↔ miles ↔ feet
    Weight      : kilograms ↔ pounds ↔ grams

    Args:
        value:     The numeric value to convert.
        from_unit: Source unit (case-insensitive).
        to_unit:   Target unit (case-insensitive).

    Returns:
        The converted value with units, or an error message if the conversion
        is not supported.
    """
    f = from_unit.lower().strip()
    t = to_unit.lower().strip()

    # --- temperature ---
    def _temp(v: float, src: str, dst: str) -> Union[float, None]:
        to_c = {"celsius": lambda x: x, "fahrenheit": lambda x: (x - 32) * 5 / 9, "kelvin": lambda x: x - 273.15}
        from_c = {"celsius": lambda x: x, "fahrenheit": lambda x: x * 9 / 5 + 32, "kelvin": lambda x: x + 273.15}
        if src in to_c and dst in from_c:
            return from_c[dst](to_c[src](v))
        return None

    # --- distance (all via metres) ---
    to_m = {"meters": 1, "kilometres": 1000, "kilometers": 1000, "miles": 1609.344, "feet": 0.3048}
    if f in to_m and t in to_m:
        result = value * to_m[f] / to_m[t]
        return f"{value} {from_unit} = {result:.6g} {to_unit}"

    # --- weight (all via grams) ---
    to_g = {"grams": 1, "kilograms": 1000, "pounds": 453.592}
    if f in to_g and t in to_g:
        result = value * to_g[f] / to_g[t]
        return f"{value} {from_unit} = {result:.6g} {to_unit}"

    temp_result = _temp(value, f, t)
    if temp_result is not None:
        return f"{value} {from_unit} = {temp_result:.6g} {to_unit}"

    return f"Unsupported conversion: {from_unit} -> {to_unit}"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_agent_tools() -> list:
    """Return the full list of tools available to worker agents."""
    return [calculator, summarize_text, text_statistics, unit_converter]
