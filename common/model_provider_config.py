"""Shared LM Studio/OpenAI-compatible configuration helpers for workshop demos."""

from __future__ import annotations

import os

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LMSTUDIO_MODEL = "google/gemma-4-26b-a4b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _has_openai_credentials() -> bool:
    """Return True when a real OpenAI API key is present."""
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    return bool(api_key and api_key.lower() != "lm-studio")


def configure_openai_compat_env(
    base_url: str | None = None, api_key: str | None = None
) -> str | None:
    """Use OpenAI env vars as canonical config, defaulting them to LM Studio values."""
    resolved_base_url = (
        base_url
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or DEFAULT_BASE_URL
    )
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or "lm-studio"

    if not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = resolved_base_url
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = resolved_base_url
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = resolved_api_key

    return resolved_base_url


def resolve_model_id(
    default_model: str | None = None, include_agent_model: bool = False
) -> str:
    """Resolve an OpenAI-provider model id from OpenAI-style environment variables."""
    use_openai = _has_openai_credentials()
    resolved_default = default_model or (
        DEFAULT_OPENAI_MODEL if use_openai else DEFAULT_LMSTUDIO_MODEL
    )

    candidates: list[str | None] = []
    if include_agent_model:
        candidates.append(os.environ.get("AGENT_MODEL"))
    candidates.append(os.environ.get("ADK_MODEL"))
    candidates.append(os.environ.get("OPENAI_MODEL"))
    candidates.append(resolved_default)

    raw_model = next((m for m in candidates if m), resolved_default)
    return raw_model if raw_model.startswith("openai/") else f"openai/{raw_model}"


def setup_model(
    default_model: str | None = None, include_agent_model: bool = False
) -> str:
    """Return model id, preferring OpenAI config and falling back to LM Studio."""
    configure_openai_compat_env()
    return resolve_model_id(default_model=default_model, include_agent_model=include_agent_model)


def resolve_openai_client_config(
    base_url: str | None = None,
    api_key: str | None = None,
    default_api_key: str = "lm-studio",
) -> tuple[str | None, str]:
    """Return resolved OpenAI-compatible base URL and API key for client initialization."""
    resolved_base_url = configure_openai_compat_env(base_url=base_url, api_key=api_key)
    resolved_api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or default_api_key
    return resolved_base_url, resolved_api_key
