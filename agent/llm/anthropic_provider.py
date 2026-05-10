from typing import Iterator

from .base import LLMProvider, Message


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key or None)
        return self._client

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    @classmethod
    def from_config(cls, cfg: dict) -> "AnthropicProvider":
        return cls(
            model=cfg.get("model", "claude-sonnet-4-6"),
            api_key=cfg.get("api_key", ""),
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 4096),
        )

    def complete(self, messages: list[Message]) -> str:
        client = self._get_client()
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]
        system = system_msgs[0].content if system_msgs else None

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[m.to_dict() for m in other_msgs],
        )
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        return response.content[0].text

    def stream(self, messages: list[Message]) -> Iterator[str]:
        client = self._get_client()
        system_msgs = [m for m in messages if m.role == "system"]
        other_msgs = [m for m in messages if m.role != "system"]
        system = system_msgs[0].content if system_msgs else None

        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[m.to_dict() for m in other_msgs],
        )
        if system:
            kwargs["system"] = system

        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Anthropic does not provide an embedding API. "
            "Configure a separate embeddings provider (e.g. ollama) in config.yaml."
        )
