# Local AI Agent

A local AI agent for code review, debugging, and repo maintenance. Runs fully offline using [Ollama](https://ollama.com) with modular support for swapping in cloud LLMs (Anthropic, OpenAI) when needed. Uses ChromaDB for vector search and RAG context.

## Stack

| Layer | Tool |
|---|---|
| Local LLM | qwen3:8b via Ollama |
| Embeddings | nomic-embed-text via Ollama |
| Vector DB | ChromaDB (persistent, local) |
| CLI | Click |
| API | FastAPI + SSE streaming |

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/download) installed and running

## Setup

```bash
git clone https://github.com/asjad2401/LocalQwenModel.git
cd LocalQwenModel

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Pull the models
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Make the `agent` command available from anywhere:
```bash
mkdir -p ~/.local/bin ~/.config/agent
ln -sf $(pwd)/.venv/bin/agent ~/.local/bin/agent
ln -sf $(pwd)/config.yaml ~/.config/agent/config.yaml
```

## Configuration

Edit `config.yaml` to switch providers or models:

```yaml
llm:
  provider: ollama          # ollama | anthropic | openai
  model: qwen3:8b
  base_url: http://localhost:11434

embeddings:
  provider: ollama
  model: nomic-embed-text

memory:
  db_path: ~/.local/share/agent/chroma
  default_collection: default
```

**Use a cloud LLM with local embeddings:**
```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-6
  api_key: sk-ant-...

embeddings:
  provider: ollama          # embeddings stay local
  model: nomic-embed-text
```

## CLI Usage

### Review code

```bash
agent review src/main.py          # review a file
agent review --diff               # review unstaged git changes
agent review --staged             # review staged changes (pre-commit)
```

### Debug errors

```bash
agent debug "TypeError: 'NoneType' object is not subscriptable"
agent debug --file error.log
python app.py 2>&1 | agent debug
```

### Explain code

```bash
agent explain src/auth/middleware.py
```

### Index a repo for semantic search + RAG

```bash
agent index . --collection myproject
```

Re-run whenever the codebase changes significantly. Safe to re-run (upserts, no duplicates).

### Semantic search

```bash
agent --collection myproject search "rate limiting logic"
agent --collection myproject search "JWT token validation" -n 10
```

### Interactive chat

```bash
agent chat
```

### Start the REST API

```bash
agent serve                        # http://127.0.0.1:8000
agent serve --host 0.0.0.0 --port 8080
```

Docs available at `http://127.0.0.1:8000/docs`.

### Other

```bash
agent models                       # list local Ollama models
agent collections                  # list indexed collections + chunk counts
```

## REST API

All endpoints accept `"stream": true` for Server-Sent Events streaming.

| Method | Endpoint | Body |
|---|---|---|
| `GET` | `/health` | — |
| `POST` | `/review` | `{"content": "...", "filename": "...", "collection": "..."}` |
| `POST` | `/debug` | `{"error": "...", "hint": "..."}` |
| `POST` | `/explain` | `{"content": "..."}` |
| `POST` | `/chat` | `{"messages": [{"role": "user", "content": "..."}]}` |
| `POST` | `/search` | `{"query": "...", "n": 5, "collection": "..."}` |
| `POST` | `/index` | `{"repo_path": "/path/to/repo", "collection": "..."}` |
| `GET` | `/collections` | — |

```bash
curl http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{"content": "def add(a, b): return a - b", "filename": "math.py"}'
```

## Adding a Custom LLM Provider

```python
from agent.llm import register_provider, LLMProvider, Message
from typing import Iterator

class MyProvider(LLMProvider):
    @property
    def name(self) -> str:
        return "myprovider/mymodel"

    @classmethod
    def from_config(cls, cfg: dict) -> "MyProvider":
        return cls()

    def complete(self, messages: list[Message]) -> str:
        ...

    def stream(self, messages: list[Message]) -> Iterator[str]:
        ...

    def embed(self, text: str) -> list[float]:
        ...

register_provider("myprovider", MyProvider)
```

## Recommended Workflow

```bash
# 1. Index your project once
agent index . --collection myproject

# 2. Review before committing
agent review --staged

# 3. Debug failures
pytest 2>&1 | agent debug

# 4. Explore unfamiliar code
agent --collection myproject search "how does auth work"
agent explain src/auth/jwt.py

# 5. Re-index as the codebase grows
agent index . --collection myproject
```
