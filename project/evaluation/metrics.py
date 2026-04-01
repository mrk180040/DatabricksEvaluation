from __future__ import annotations

from typing import Iterable


def routing_accuracy(predicted_agent: str, expected_agent: str) -> float:
    return 1.0 if predicted_agent == expected_agent else 0.0


def keyword_match_score(text: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    lowered = text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lowered)
    return hits / len(expected_keywords)


def response_quality_score_placeholder() -> float:
    return 0.0


def average(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
