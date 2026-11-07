from langgraph.types import RunnableConfig
from .base import ChatProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider

_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
    "gemini": "gemini-3.5-flash",
}

def get_provider(config: RunnableConfig) -> ChatProvider:
    cfg = config.get("configurable") or {}
    provider_name = cfg.get("llm_provider", "openai")

    if provider_name == "openai":
        return OpenAIProvider(config)
    elif provider_name == "anthropic":
        return AnthropicProvider(config)
    elif provider_name == "gemini":
        return GeminiProvider(config)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

def resolve_model(config: RunnableConfig) -> str:
    cfg = config.get("configurable") or {}
    provider_name = cfg.get("llm_provider", "openai")
    return cfg.get("llm_model") or _DEFAULT_MODELS.get(
        provider_name, "gpt-4o"
    )


def background_model(creds: dict) -> str:
    """The model a non-chat agent runs on, from a credentials response.

    The backend sends the platform's cheap model for the tenant's provider as
    ``backgroundModel`` (aurora, sherlock, clara's triage and task extraction);
    chat agents and clara's drafts keep ``model``. An older backend omits the
    field, which means no split: the tenant's model everywhere.
    """
    return creds.get("backgroundModel") or creds.get("model") or ""


def resolve_background_model(config: RunnableConfig) -> str:
    """``llm_background_model`` from ``configurable``, else the tenant's model."""
    cfg = config.get("configurable") or {}
    return cfg.get("llm_background_model") or resolve_model(config)
