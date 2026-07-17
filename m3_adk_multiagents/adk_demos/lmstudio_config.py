"""Shared LM Studio/OpenAI-compatible configuration helpers for ADK demos."""

from __future__ import annotations

import os

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "google/gemma-4-26b-a4b"


def configure_openai_compat_env(
    base_url: str | None = None, api_key: str | None = None
) -> str:
    """Set OpenAI-compatible env vars used by ADK/LiteLLM and return base URL."""
    resolved_base_url = base_url or os.environ.get("LMSTUDIO_BASE_URL", DEFAULT_BASE_URL)
    resolved_api_key = api_key or os.environ.get("LMSTUDIO_API_KEY", "lm-studio")

    if not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = resolved_base_url
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = resolved_base_url
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = resolved_api_key

    return resolved_base_url


def resolve_model_id(
    default_model: str = DEFAULT_MODEL, include_agent_model: bool = False
) -> str:
    """Resolve a provider-prefixed ADK model id from environment variables."""
    candidates: list[str | None] = []
    if include_agent_model:
        candidates.append(os.environ.get("AGENT_MODEL"))
    candidates.extend([
        os.environ.get("ADK_MODEL"),
        os.environ.get("LMSTUDIO_MODEL"),
        default_model,
    ])

    raw_model = next((m for m in candidates if m), default_model)
    return raw_model if raw_model.startswith("openai/") else f"openai/{raw_model}"


def setup_lmstudio_model(
    default_model: str = DEFAULT_MODEL, include_agent_model: bool = False
) -> str:
    """Configure env for LM Studio and return normalized provider model id."""
    configure_openai_compat_env()
    return resolve_model_id(
        default_model=default_model, include_agent_model=include_agent_model
    )
