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

from project.utils.logger import get_logger, log_step

_logger = get_logger("llm_client")


class LLMClientError(RuntimeError):
    """Base class for LLM client failures."""


class LLMNotConfiguredError(LLMClientError):
    """Raised when the provider client cannot be constructed."""


class LLMCompletionError(LLMClientError):
    """Raised when completion retries are exhausted."""


class LLMResponseFormatError(LLMClientError):
    """Raised when the model response is not valid JSON or schema-compatible."""


@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "databricks")
    model: str = os.getenv("DATABRICKS_MODEL_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "700"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))


class LLMClient:
    def __init__(self, config: LLMConfig | None = None, access_token: str | None = None):
        self.config = config or LLMConfig()
        self._access_token_override = access_token
        self.client = self._build_client()
        log_step(
            _logger,
            "llm_client_initialized",
            provider=self.config.provider,
            model=self.config.model,
            host_set=bool(os.getenv("DATABRICKS_HOST")),
            access_token_override_provided=bool(access_token),
            env_obo_token_set=bool(os.getenv("DATABRICKS_OBO_TOKEN")),
            auth_source=self.auth_source(),
            client_available=self.client is not None,
        )

    def _build_client(self) -> OpenAI | None:
        if OpenAI is None:
            log_step(_logger, "llm_client_build_failed", reason="openai_package_not_installed")
            return None
        provider = self.config.provider.lower()
        if provider == "databricks":
            host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
            token = self._resolve_databricks_token()
            if not host:
                log_step(
                    _logger,
                    "llm_client_build_failed",
                    provider=provider,
                    reason="missing_host",
                    host_set=False,
                    token_set=bool(token),
                    auth_source=self.auth_source(),
                )
                return None
            if not token:
                log_step(
                    _logger,
                    "llm_client_build_failed",
                    provider=provider,
                    reason="missing_token",
                    host_set=True,
                    token_set=False,
                    access_token_override_provided=bool(self._access_token_override),
                    env_obo_token_set=bool(os.getenv("DATABRICKS_OBO_TOKEN")),
                )
                return None
            log_step(
                _logger,
                "llm_client_build_succeeded",
                provider=provider,
                auth_source=self.auth_source(),
            )
            return OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                log_step(_logger, "llm_client_build_failed", provider=provider, reason="missing_openai_api_key")
                return None
            log_step(_logger, "llm_client_build_succeeded", provider=provider, auth_source="openai_api_key")
            return OpenAI(api_key=api_key)

        log_step(_logger, "llm_client_build_failed", provider=provider, reason="unknown_provider")
        return None

    def _resolve_databricks_token(self) -> str | None:
        """Resolve Databricks token with precedence: request override > OBO env."""
        if self._access_token_override:
            log_step(_logger, "token_resolved", source="access_token_override")
            return self._access_token_override
        # Always check environment variable at resolution time (not at init time).
        # This handles Streamlit caching and dynamic env var loading scenarios.
        obo_token = os.getenv("DATABRICKS_OBO_TOKEN")
        if obo_token:
            log_step(_logger, "token_resolved", source="env_obo_token")
        else:
            log_step(
                _logger,
                "token_resolved",
                source="none",
                env_obo_token_set=False,
            )
        return obo_token

    def auth_source(self) -> str:
        provider = self.config.provider.lower()
        if provider == "databricks":
            if self._access_token_override:
                return "request_token"
            if os.getenv("DATABRICKS_OBO_TOKEN"):
                return "env_obo_token"
            return "none"
        if provider == "openai":
            return "openai_api_key" if os.getenv("OPENAI_API_KEY") else "none"
        return "unknown"

    def has_access_token_override(self) -> bool:
        """Return True if an OBO token was passed directly at construction time."""
        return bool(self._access_token_override)

    def as_langchain_chat_model(self):
        """Return a LangChain ``BaseChatModel`` configured for this client's provider.

        Uses ``langchain_openai.ChatOpenAI`` pointed at the Databricks serving-endpoints
        OpenAI-compatible API so that the OBO token flow is respected identically to the
        rest of the application.  Falls back to a standard OpenAI target when
        ``provider == "openai"``.
        """
        from langchain_openai import ChatOpenAI

        provider = self.config.provider.lower()
        if provider == "databricks":
            host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
            token = self._resolve_databricks_token() or "placeholder"
            return ChatOpenAI(
                api_key=token,
                base_url=f"{host}/serving-endpoints",
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        if provider == "openai":
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        raise ValueError(f"as_langchain_chat_model: unsupported provider '{self.config.provider}'")

    def set_access_token(self, access_token: str | None) -> None:
        self._access_token_override = access_token
        log_step(
            _logger,
            "access_token_updated",
            access_token_provided=bool(access_token),
            auth_source=self.auth_source(),
        )
        self.client = self._build_client()

    def _ensure_client(self) -> None:
        """Lazily rebuild the client if it is not yet initialised.

        Tokens may become available after construction (e.g. Databricks App OBO
        env-var injection, `.env` file loaded asynchronously, Streamlit cache
        reuse across reruns).  Re-attempting the build here ensures those tokens
        are picked up at call-time rather than only at init-time.
        """
        if self.client is None:
            log_step(
                _logger,
                "ensure_client_rebuild_attempt",
                provider=self.config.provider,
                host_set=bool(os.getenv("DATABRICKS_HOST")),
                access_token_override_provided=bool(self._access_token_override),
                env_obo_token_set=bool(os.getenv("DATABRICKS_OBO_TOKEN")),
                auth_source=self.auth_source(),
            )
            self.client = self._build_client()
            log_step(
                _logger,
                "ensure_client_rebuild_result",
                client_available=self.client is not None,
                auth_source=self.auth_source(),
            )

    def available(self) -> bool:
        self._ensure_client()
        return self.client is not None

    def chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self._ensure_client()
        if self.client is None:
            if self.config.provider.lower() == "databricks":
                host = os.getenv("DATABRICKS_HOST", "")
                token = self._resolve_databricks_token()
                if not host:
                    raise LLMNotConfiguredError(
                        "DATABRICKS_HOST is required. "
                        "Deploy as a Databricks App (auto-injected via app.yaml) or set DATABRICKS_HOST manually. "
                        f"auth_source={self.auth_source()}"
                    )
                if not token:
                    raise LLMNotConfiguredError(
                        "DATABRICKS_OBO_TOKEN is required. "
                        "Deploy as a Databricks App with agent-obo-scope/obo-token configured in app.yaml. "
                        f"auth_source={self.auth_source()}"
                    )
            raise LLMNotConfiguredError(
                f"LLM client is not configured for provider={self.config.provider} auth_source={self.auth_source()}."
            )

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

        raise LLMCompletionError(
            f"LLM completion failed after retries for provider={self.config.provider} "
            f"model={self.config.model} auth_source={self.auth_source()}: {last_err}"
        )

    def json_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        raw = self.chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)
        return self._safe_json(raw)

    @staticmethod
    def _safe_json(text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise LLMResponseFormatError("Model returned JSON that is not an object.")
            return parsed
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                try:
                    parsed = json.loads(text[start : end + 1])
                    if not isinstance(parsed, dict):
                        raise LLMResponseFormatError("Model returned embedded JSON that is not an object.")
                    return parsed
                except json.JSONDecodeError:
                    raise LLMResponseFormatError("Model returned invalid JSON response.")
            raise LLMResponseFormatError("Model response did not contain valid JSON.")
