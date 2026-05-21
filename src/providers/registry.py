from langgraph.types import RunnableConfig
from .base import ChatProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .vertex import VertexProvider

_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
    "vertex": "gemini-1.5-flash",
}

def get_provider(config: RunnableConfig) -> ChatProvider:
    cfg = config.get("configurable") or {}
    provider_name = cfg.get("llm_provider", "openai")

    if provider_name == "openai":
        return OpenAIProvider(config)
    elif provider_name == "anthropic":
        return AnthropicProvider(config)
    elif provider_name == "vertex":
        return VertexProvider(config)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

def resolve_model(config: RunnableConfig) -> str:
    cfg = config.get("configurable") or {}
    provider_name = cfg.get("llm_provider", "openai")
    return cfg.get("llm_model") or _DEFAULT_MODELS.get(
        provider_name, "gpt-4o"
    )
