"""FastAPI REST API — run with: agent serve  OR  uvicorn agent.api:app"""
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import load_config, build_agent

app = FastAPI(title="Local Agent API", version="0.1.0")

# config is loaded once at startup; collection can be overridden per-request
_cfg: dict = {}
_default_collection = "default"


@app.on_event("startup")
def _startup():
    global _cfg, _default_collection
    _cfg = load_config()
    _default_collection = _cfg.get("memory", {}).get("default_collection", "default")


def _agent(collection: str | None = None):
    return build_agent(_cfg, collection=collection or _default_collection)


# ------------------------------------------------------------------ #
#  Request / Response models                                           #
# ------------------------------------------------------------------ #

class ReviewRequest(BaseModel):
    content: str = Field(..., description="File content or diff to review")
    filename: str = Field("", description="Optional filename for context")
    collection: str | None = None
    stream: bool = False

class DebugRequest(BaseModel):
    error: str = Field(..., description="Error message or traceback")
    hint: str = Field("", description="Extra context hint")
    collection: str | None = None
    stream: bool = False

class ExplainRequest(BaseModel):
    content: str
    filename: str = ""
    stream: bool = False

class ChatMessage(BaseModel):
    role: str   # system | user | assistant
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    stream: bool = False

class SearchRequest(BaseModel):
    query: str
    n: int = Field(5, ge=1, le=20)
    collection: str | None = None

class IndexRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to the repository")
    collection: str | None = None

class TextResponse(BaseModel):
    result: str
    provider: str


# ------------------------------------------------------------------ #
#  Streaming helper                                                     #
# ------------------------------------------------------------------ #

def _sse(tokens) -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        for token in tokens:
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


# ------------------------------------------------------------------ #
#  Endpoints                                                            #
# ------------------------------------------------------------------ #

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/review")
def review(req: ReviewRequest):
    agent = _agent(req.collection)
    try:
        if req.stream:
            return _sse(agent.review(req.content, filename=req.filename, stream=True))
        result = agent.review(req.content, filename=req.filename, stream=False)
        return TextResponse(result=result, provider=agent.llm.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug")
def debug(req: DebugRequest):
    agent = _agent(req.collection)
    try:
        if req.stream:
            return _sse(agent.debug(req.error, hint=req.hint, stream=True))
        result = agent.debug(req.error, hint=req.hint, stream=False)
        return TextResponse(result=result, provider=agent.llm.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
def explain(req: ExplainRequest):
    agent = _agent()
    try:
        if req.stream:
            return _sse(agent.explain(req.content, filename=req.filename, stream=True))
        result = agent.explain(req.content, filename=req.filename, stream=False)
        return TextResponse(result=result, provider=agent.llm.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(req: ChatRequest):
    from .llm.base import Message
    agent = _agent()
    history = [Message(m.role, m.content) for m in req.messages]
    try:
        if req.stream:
            return _sse(agent.llm.stream(history))
        result = agent.llm.complete(history)
        return TextResponse(result=result, provider=agent.llm.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search(req: SearchRequest):
    agent = _agent(req.collection)
    results = agent.search(req.query, n=req.n)
    return {"results": results, "count": len(results)}


@app.post("/index")
def index(req: IndexRequest):
    agent = _agent(req.collection)
    if agent.db is None:
        raise HTTPException(status_code=400, detail="No vector DB configured in config.yaml")
    if not Path(req.repo_path).exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {req.repo_path}")
    try:
        total = agent.index(req.repo_path)
        return {"indexed_chunks": total, "collection": agent.collection}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
def list_collections():
    from .memory.vectordb import VectorDB
    db_path = _cfg.get("memory", {}).get("db_path")
    if not db_path:
        raise HTTPException(status_code=400, detail="memory.db_path not set in config.yaml")
    db = VectorDB(db_path)
    cols = db.list_collections()
    return {"collections": [{"name": c, "count": db.collection_count(c)} for c in cols]}
