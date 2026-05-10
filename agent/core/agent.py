from typing import Iterator

from ..llm.base import LLMProvider, Message
from ..memory.vectordb import VectorDB
from ..memory.indexer import iter_repo_files, chunk_file, make_chunk_id

REVIEW_SYSTEM = """\
You are an expert code reviewer. Analyze the provided code for:
- Bugs and logic errors
- Security vulnerabilities (injection, auth issues, data exposure, etc.)
- Performance bottlenecks
- Code quality and maintainability issues
- Misleading names or missing error handling

Structure your response:
1. **Summary** — one-line verdict
2. **Issues** — list each issue with severity (Critical / Major / Minor), file/line if known, and a fix
3. **Positives** — what the code does well (keep it brief)

If the code is clean, say so concisely.\
"""

DEBUG_SYSTEM = """\
You are an expert debugger. Given an error and relevant code context:
1. **Root cause** — what exactly is failing and why
2. **Fix** — concrete code change to resolve it
3. **Prevention** — one-line note on how to avoid recurrence

Be direct. No padding.\
"""

EXPLAIN_SYSTEM = """\
You are a senior engineer explaining code to a colleague.
Be concise: what the code does, the key design choices, and any non-obvious behaviour.
Use plain language; no filler.\
"""


class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        db: VectorDB | None = None,
        embed_provider: LLMProvider | None = None,
        collection: str = "default",
    ):
        self.llm = llm
        self.db = db
        # use a separate embedding provider if specified (e.g. local ollama + remote LLM)
        self._embed = embed_provider or llm
        self.collection = collection

    # ------------------------------------------------------------------ #
    #  Public actions                                                       #
    # ------------------------------------------------------------------ #

    def review(self, content: str, filename: str = "", stream: bool = False) -> str | Iterator[str]:
        ctx = self._context_block(content)
        user = f"Review the following code{f' ({filename})' if filename else ''}:\n\n```\n{content}\n```"
        if ctx:
            user += f"\n\n**Related code from the codebase:**\n{ctx}"
        messages = [Message("system", REVIEW_SYSTEM), Message("user", user)]
        return self.llm.stream(messages) if stream else self.llm.complete(messages)

    def debug(self, error: str, hint: str = "", stream: bool = False) -> str | Iterator[str]:
        ctx = self._context_block(f"{error} {hint}")
        user = f"Debug this error:\n\n```\n{error}\n```"
        if ctx:
            user += f"\n\n**Relevant code from the codebase:**\n{ctx}"
        messages = [Message("system", DEBUG_SYSTEM), Message("user", user)]
        return self.llm.stream(messages) if stream else self.llm.complete(messages)

    def explain(self, content: str, filename: str = "", stream: bool = False) -> str | Iterator[str]:
        user = f"Explain this code{f' ({filename})' if filename else ''}:\n\n```\n{content}\n```"
        messages = [Message("system", EXPLAIN_SYSTEM), Message("user", user)]
        return self.llm.stream(messages) if stream else self.llm.complete(messages)

    def chat(self, history: list[Message], stream: bool = False) -> str | Iterator[str]:
        return self.llm.stream(history) if stream else self.llm.complete(history)

    def search(self, query: str, n: int = 5) -> list[dict]:
        return self._get_context(query, n=n)

    def index(self, repo_path: str, progress_cb=None) -> int:
        if self.db is None:
            raise RuntimeError("No VectorDB configured — add memory.db_path to config.yaml")
        total = 0
        for file_path in iter_repo_files(repo_path):
            chunks = chunk_file(file_path)
            if not chunks:
                continue
            ids, docs, embeddings, metadatas = [], [], [], []
            for chunk in chunks:
                try:
                    emb = self._embed.embed(chunk["content"])
                except Exception:
                    continue
                ids.append(make_chunk_id(file_path, chunk))
                docs.append(chunk["content"])
                embeddings.append(emb)
                metadatas.append({
                    "file": str(file_path),
                    "type": chunk["type"],
                    "name": chunk["name"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                })
                total += 1
            if ids:
                self.db.upsert(self.collection, ids, docs, embeddings, metadatas)
            if progress_cb:
                progress_cb(str(file_path), len(chunks))
        return total

    # ------------------------------------------------------------------ #
    #  Internals                                                            #
    # ------------------------------------------------------------------ #

    def _get_context(self, text: str, n: int = 5) -> list[dict]:
        if self.db is None:
            return []
        try:
            emb = self._embed.embed(text)
            return self.db.query(self.collection, emb, n_results=n)
        except Exception:
            return []

    def _context_block(self, text: str, n: int = 3) -> str:
        chunks = self._get_context(text, n=n)
        if not chunks:
            return ""
        parts = []
        for c in chunks:
            m = c["metadata"]
            parts.append(
                f"### {m['file']} (lines {m['start_line']}–{m['end_line']}, {m['type']}: {m['name']})\n"
                f"```\n{c['content']}\n```"
            )
        return "\n\n".join(parts)
