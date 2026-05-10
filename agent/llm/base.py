from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Message:
    role: str   # system | user | assistant
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[Message]) -> str:
        pass

    def stream(self, messages: list[Message]) -> Iterator[str]:
        # providers override this for true streaming; default wraps complete()
        yield self.complete(messages)

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: dict) -> "LLMProvider":
        pass
