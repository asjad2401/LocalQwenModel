import os
from pathlib import Path

import yaml

_DEFAULT_PATHS = [
    "config.yaml",
    Path.home() / ".config" / "agent" / "config.yaml",
]


def load_config(path: str | None = None) -> dict:
    candidates = ([Path(path)] if path else []) + _DEFAULT_PATHS
    for p in candidates:
        p = Path(p)
        if p.exists():
            with p.open() as f:
                return yaml.safe_load(f) or {}
    return {}


def build_agent(cfg: dict, collection: str | None = None):
    from .llm import get_provider
    from .memory.vectordb import VectorDB
    from .core.agent import Agent

    llm_cfg = cfg.get("llm", {})
    embed_cfg = cfg.get("embeddings", {})
    mem_cfg = cfg.get("memory", {})

    llm = get_provider(llm_cfg)

    # build a separate embed provider only if config differs from llm
    embed_provider = None
    if embed_cfg and embed_cfg.get("provider") != llm_cfg.get("provider"):
        embed_provider = get_provider(embed_cfg)
    elif embed_cfg and embed_cfg.get("provider") == "ollama":
        # same provider type but may have different model/base_url
        from .llm.ollama_provider import OllamaProvider
        embed_provider = OllamaProvider.from_config({
            "model": embed_cfg.get("model", "nomic-embed-text"),
            "base_url": embed_cfg.get("base_url", "http://localhost:11434"),
            "embed_model": embed_cfg.get("model", "nomic-embed-text"),
        })

    db_path = mem_cfg.get("db_path")
    db = VectorDB(db_path) if db_path else None

    col = collection or mem_cfg.get("default_collection", "default")

    return Agent(llm=llm, db=db, embed_provider=embed_provider, collection=col)
