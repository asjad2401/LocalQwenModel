from typing import Iterator

from .base import LLMProvider, Message


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        embed_model: str = "text-embedding-3-small",
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.embed_model = embed_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key or None,
                base_url=self.base_url or None,
            )
        return self._client

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    @classmethod
    def from_config(cls, cfg: dict) -> "OpenAIProvider":
        return cls(
            model=cfg.get("model", "gpt-4o"),
            api_key=cfg.get("api_key", ""),
            base_url=cfg.get("base_url"),
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 4096),
            embed_model=cfg.get("embed_model", "text-embedding-3-small"),
        )

    def complete(self, messages: list[Message]) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def stream(self, messages: list[Message]) -> Iterator[str]:
        client = self._get_client()
        stream = client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def embed(self, text: str) -> list[float]:
        client = self._get_client()
        response = client.embeddings.create(model=self.embed_model, input=text)
        return response.data[0].embedding
