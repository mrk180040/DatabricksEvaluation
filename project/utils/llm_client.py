from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore[assignment]


@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "databricks")
    model: str = os.getenv("DATABRICKS_MODEL_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "700"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.client = self._build_client()

    def _build_client(self) -> OpenAI | None:
        if OpenAI is None:
            return None
        provider = self.config.provider.lower()
        if provider == "databricks":
            host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
            token = os.getenv("DATABRICKS_TOKEN")
            if not host or not token:
                return None
            return OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None
            return OpenAI(api_key=api_key)

        return None

    def available(self) -> bool:
        return self.client is not None

    def chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if self.client is None:
            raise RuntimeError("LLM client is not configured. Missing credentials or unsupported provider.")

        last_err: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature if temperature is None else temperature,
                    max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_err = exc
                if attempt == self.config.max_retries:
                    break
                time.sleep(0.5 * attempt)

        raise RuntimeError(f"LLM completion failed after retries: {last_err}")

    def json_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            raw = self.chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)
            return self._safe_json(raw, fallback=fallback)
        except Exception:
            return fallback

    @staticmethod
    def _safe_json(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return fallback
            return fallback
