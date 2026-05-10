import ast
import hashlib
from pathlib import Path
from typing import Iterator

IGNORED_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", ".env",
    "node_modules", ".tox", "dist", "build", ".eggs", ".mypy_cache",
    ".pytest_cache", "coverage", ".coverage", "htmlcov",
}

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".cpp", ".c", ".h", ".cc", ".hpp",
    ".rb", ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".toml", ".json",
    ".md", ".txt", ".rst",
    ".sql", ".graphql",
}

# files larger than this are skipped
MAX_FILE_BYTES = 500_000


def iter_repo_files(repo_path: str) -> Iterator[Path]:
    root = Path(repo_path).resolve()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(d in path.parts for d in IGNORED_DIRS):
            continue
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def _chunk_python(path: Path) -> list[dict]:
    source = path.read_text(errors="replace")
    chunks: list[dict] = []
    try:
        tree = ast.parse(source)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = node.lineno - 1
            end = node.end_lineno  # type: ignore[attr-defined]
            content = "\n".join(lines[start:end])
            if len(content.strip()) < 20:
                continue
            chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append({
                "content": content,
                "type": chunk_type,
                "name": node.name,
                "start_line": start + 1,
                "end_line": end,
            })
    except SyntaxError:
        pass
    return chunks


def _chunk_by_lines(content: str, chunk_size: int = 60, overlap: int = 10) -> list[dict]:
    lines = content.splitlines()
    chunks: list[dict] = []
    i = 0
    while i < len(lines):
        end = min(i + chunk_size, len(lines))
        text = "\n".join(lines[i:end])
        if text.strip():
            chunks.append({
                "content": text,
                "type": "chunk",
                "name": f"lines_{i + 1}_{end}",
                "start_line": i + 1,
                "end_line": end,
            })
        i += chunk_size - overlap
    return chunks


def chunk_file(path: Path) -> list[dict]:
    try:
        if path.suffix == ".py":
            chunks = _chunk_python(path)
            if not chunks:
                content = path.read_text(errors="replace")
                chunks = _chunk_by_lines(content)
        else:
            content = path.read_text(errors="replace")
            chunks = _chunk_by_lines(content)
    except Exception:
        return []
    return chunks


def make_chunk_id(file_path: Path, chunk: dict) -> str:
    key = f"{file_path}:{chunk['name']}:{chunk['start_line']}"
    return hashlib.md5(key.encode()).hexdigest()
