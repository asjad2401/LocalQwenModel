from .base import LLMProvider, Message
from .ollama_provider import OllamaProvider
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def get_provider(cfg: dict) -> LLMProvider:
    provider_type = cfg.get("provider", "ollama")
    if provider_type not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider_type}'. Available: {list(_PROVIDERS)}"
        )
    return _PROVIDERS[provider_type].from_config(cfg)


def register_provider(name: str, cls: type[LLMProvider]) -> None:
    _PROVIDERS[name] = cls


__all__ = [
    "LLMProvider", "Message",
    "OllamaProvider", "AnthropicProvider", "OpenAIProvider",
    "get_provider", "register_provider",
]
