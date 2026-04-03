from __future__ import annotations

import re

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"disregard\s+all\s+rules",
    r"system\s+prompt",
    r"reveal\s+hidden\s+instructions",
    r"jailbreak",
    r"developer\s+mode",
]

SECRET_PATTERNS = [
    r"\bsk-[A-Za-z0-9]{20,}\b",
    r"\bdapi[a-f0-9]{32}\b",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\b(?:api[_-]?key|token|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{10,}['\"]?",
]


def detect_prompt_injection(text: str) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            hits.append(pattern)
    return hits


def detect_secrets(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in SECRET_PATTERNS:
        for match in re.findall(pattern, text):
            hits.append(match if isinstance(match, str) else "possible_secret")
    return hits


def detect_restricted_topics(text: str, topics: list[str]) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for topic in topics:
        if topic.lower() in lowered:
            found.append(topic)
    return found
