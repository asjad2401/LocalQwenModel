import json
from typing import Iterator

import httpx

from .base import LLMProvider, Message


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "qwen2.5-coder:7b-instruct-q4_K_M",
        base_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.embed_model = embed_model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    @classmethod
    def from_config(cls, cfg: dict) -> "OllamaProvider":
        return cls(
            model=cfg.get("model", "qwen2.5-coder:7b-instruct-q4_K_M"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            embed_model=cfg.get("embed_model", "nomic-embed-text"),
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 4096),
        )

    def _supports_chat(self) -> bool:
        """Check if the model supports the /api/chat endpoint."""
        try:
            with httpx.Client(timeout=10) as client:
                r = client.post(
                    f"{self.base_url}/api/chat",
                    json={"model": self.model, "messages": [{"role": "user", "content": "ping"}], "stream": False},
                )
                return r.status_code != 400
        except Exception:
            return False

    @staticmethod
    def _messages_to_prompt(messages: list[Message]) -> str:
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(f"System: {m.content}")
            elif m.role == "user":
                parts.append(f"User: {m.content}")
            elif m.role == "assistant":
                parts.append(f"Assistant: {m.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def complete(self, messages: list[Message]) -> str:
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        with httpx.Client(timeout=600) as client:
            r = client.post(f"{self.base_url}/api/chat", json=payload)
            if r.status_code == 400 and "does not support chat" in r.text:
                # fall back to generate API for legacy models
                gen_payload = {
                    "model": self.model,
                    "prompt": self._messages_to_prompt(messages),
                    "stream": False,
                    "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
                }
                r = client.post(f"{self.base_url}/api/generate", json=gen_payload)
                r.raise_for_status()
                return r.json()["response"]
            r.raise_for_status()
            return r.json()["message"]["content"]

    def stream(self, messages: list[Message]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        with httpx.Client(timeout=600) as client:
            r_check = client.post(
                f"{self.base_url}/api/chat",
                json={**payload, "stream": False, "options": {"num_predict": 1}},
            )
            use_generate = r_check.status_code == 400 and "does not support chat" in r_check.text

        if use_generate:
            gen_payload = {
                "model": self.model,
                "prompt": self._messages_to_prompt(messages),
                "stream": True,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            }
            with httpx.Client(timeout=600) as client:
                with client.stream("POST", f"{self.base_url}/api/generate", json=gen_payload) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        if not data.get("done"):
                            yield data.get("response", "")
            return

        with httpx.Client(timeout=600) as client:
            with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if not data.get("done"):
                        yield data["message"]["content"]

    def embed(self, text: str) -> list[float]:
        with httpx.Client(timeout=120) as client:
            r = client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embed_model, "input": text},
            )
            r.raise_for_status()
            return r.json()["embeddings"][0]
