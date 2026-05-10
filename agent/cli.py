"""CLI entrypoint: agent <command> [options]"""
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import load_config, build_agent
from .llm.base import Message

console = Console()


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _stream_to_console(tokens, title: str = ""):
    if title:
        console.print(f"[dim]{title}[/dim]")
    result = ""
    for token in tokens:
        print(token, end="", flush=True)
        result += token
    print()
    return result


def _print_md(text: str):
    console.print(Markdown(text))


# ------------------------------------------------------------------ #
#  CLI root                                                            #
# ------------------------------------------------------------------ #

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], max_content_width=100)

@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--config", "-c",
    default=None,
    metavar="PATH",
    help=(
        "Path to a config.yaml file. If omitted, the agent looks for config.yaml "
        "in the current directory, then ~/.config/agent/config.yaml."
    ),
)
@click.option(
    "--collection",
    default=None,
    metavar="NAME",
    help=(
        "Vector DB collection to read from / write to. Collections keep indexed "
        "repos separate (e.g. --collection myproject). Defaults to 'default'."
    ),
)
@click.pass_context
def cli(ctx, config, collection):
    """Local AI agent for code review, debugging, and repo maintenance.

    \b
    Powered by a local LLM via Ollama (default: qwen3:8b) with ChromaDB for
    semantic search and RAG context. Providers are swappable — see config.yaml.

    \b
    Global options must come BEFORE the subcommand:
      agent --collection myproject review src/main.py
      agent --config ~/other-config.yaml review --staged

    \b
    Quick start:
      agent index .                      Index current repo (run once)
      agent review --staged              Review changes before committing
      agent debug --file error.log       Debug a traceback
      agent chat                         Interactive chat session
      agent serve                        Start the REST API on :8000
    """
    ctx.ensure_object(dict)
    cfg = load_config(config)
    ctx.obj["cfg"] = cfg
    ctx.obj["collection"] = collection


def _get_agent(ctx):
    return build_agent(ctx.obj["cfg"], collection=ctx.obj.get("collection"))


# ------------------------------------------------------------------ #
#  review                                                              #
# ------------------------------------------------------------------ #

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("target", required=False, metavar="[FILE]")
@click.option(
    "--diff", "mode", flag_value="diff",
    help="Review all unstaged changes in the current git repo (git diff HEAD).",
)
@click.option(
    "--staged", "mode", flag_value="staged",
    help="Review changes staged for the next commit (git diff --staged). "
         "Ideal as a pre-commit check.",
)
@click.option(
    "--no-stream", is_flag=True,
    help="Wait for the full response before printing instead of streaming tokens.",
)
@click.pass_context
def review(ctx, target, mode, no_stream):
    """Review a file or git diff for bugs, security issues, and code quality.

    \b
    The LLM checks for:
      • Bugs and logic errors
      • Security vulnerabilities (injection, auth, data exposure)
      • Performance bottlenecks
      • Code quality and maintainability issues
      • Misleading names or missing error handling

    \b
    If a --collection is set, relevant code from the indexed repo is
    automatically injected as context so the model understands how the
    file fits into the broader codebase.

    \b
    Input sources (pick one):
      FILE      Path to a source file
      --diff    Unstaged git changes
      --staged  Staged git changes (pre-commit)

    \b
    Examples:
      agent review src/auth/middleware.py
      agent review --staged
      agent review --diff
      agent --collection myproject review src/main.py
      agent review src/main.py --no-stream
    """
    if mode == "diff":
        content = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True).stdout
        filename = "git diff (unstaged)"
    elif mode == "staged":
        content = subprocess.run(["git", "diff", "--staged"], capture_output=True, text=True).stdout
        filename = "git diff (staged)"
    elif target:
        p = Path(target)
        if not p.exists():
            console.print(f"[red]File not found: {target}[/red]")
            sys.exit(1)
        content = p.read_text(errors="replace")
        filename = target
    else:
        console.print("[red]Provide a file path, --diff, or --staged.[/red]")
        console.print("[dim]Run 'agent review --help' for usage.[/dim]")
        sys.exit(1)

    if not content.strip():
        console.print("[yellow]Nothing to review (empty content).[/yellow]")
        return

    agent = _get_agent(ctx)
    console.print(f"[dim]Reviewing with {agent.llm.name} ...[/dim]\n")

    if no_stream:
        result = agent.review(content, filename=filename, stream=False)
        _print_md(result)
    else:
        _stream_to_console(agent.review(content, filename=filename, stream=True))


# ------------------------------------------------------------------ #
#  debug                                                               #
# ------------------------------------------------------------------ #

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("error_input", required=False, metavar="[ERROR]")
@click.option(
    "--file", "-f", "error_file",
    default=None,
    metavar="PATH",
    help="Read the error / traceback from a file instead of passing it inline.",
)
@click.option(
    "--hint",
    default="",
    metavar="TEXT",
    help=(
        "Extra keyword hint to improve vector DB search when the error message "
        "alone is not specific enough. E.g. --hint 'session management'."
    ),
)
@click.option(
    "--no-stream", is_flag=True,
    help="Wait for the full response before printing.",
)
@click.pass_context
def debug(ctx, error_input, error_file, hint, no_stream):
    """Debug an error message or traceback.

    \b
    The LLM will:
      1. Identify the root cause
      2. Provide a concrete code fix
      3. Suggest how to prevent recurrence

    \b
    If a --collection is set, the agent searches your indexed codebase
    for code related to the error and injects it as context automatically.
    Use --hint to steer that search when the error text is too generic.

    \b
    Input sources (pick one):
      ERROR           Inline error string as an argument
      --file PATH     Read from a log file
      stdin           Pipe output from another command

    \b
    Examples:
      agent debug "TypeError: 'NoneType' object is not subscriptable"
      agent debug --file logs/error.log
      python app.py 2>&1 | agent debug
      pytest 2>&1 | agent debug
      agent --collection myproject debug --file crash.log --hint "auth middleware"
    """
    if error_file:
        error = Path(error_file).read_text(errors="replace")
    elif error_input:
        error = error_input
    elif not sys.stdin.isatty():
        error = sys.stdin.read()
    else:
        console.print("[red]Provide an error string, --file PATH, or pipe via stdin.[/red]")
        console.print("[dim]Run 'agent debug --help' for usage.[/dim]")
        sys.exit(1)

    agent = _get_agent(ctx)
    console.print(f"[dim]Debugging with {agent.llm.name} ...[/dim]\n")

    if no_stream:
        result = agent.debug(error, hint=hint, stream=False)
        _print_md(result)
    else:
        _stream_to_console(agent.debug(error, hint=hint, stream=True))


# ------------------------------------------------------------------ #
#  explain                                                             #
# ------------------------------------------------------------------ #

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("target", metavar="FILE")
@click.option(
    "--no-stream", is_flag=True,
    help="Wait for the full response before printing.",
)
@click.pass_context
def explain(ctx, target, no_stream):
    """Explain what a file does in plain language.

    \b
    Useful for:
      • Onboarding into an unfamiliar codebase
      • Understanding code you wrote months ago
      • Getting a quick summary before reviewing or editing

    \b
    The explanation covers what the code does, key design choices,
    and any non-obvious behaviour — without padding or filler.

    \b
    Examples:
      agent explain src/auth/middleware.py
      agent explain src/core/scheduler.py --no-stream
    """
    p = Path(target)
    if not p.exists():
        console.print(f"[red]File not found: {target}[/red]")
        sys.exit(1)

    content = p.read_text(errors="replace")
    agent = _get_agent(ctx)
    console.print(f"[dim]Explaining {target} with {agent.llm.name} ...[/dim]\n")

    if no_stream:
        result = agent.explain(content, filename=target, stream=False)
        _print_md(result)
    else:
        _stream_to_console(agent.explain(content, filename=target, stream=True))


# ------------------------------------------------------------------ #
#  search                                                              #
# ------------------------------------------------------------------ #

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("query", metavar="QUERY")
@click.option(
    "-n", "n_results",
    default=5,
    show_default=True,
    metavar="N",
    help="Number of results to return (1–20).",
)
@click.pass_context
def search(ctx, query, n_results):
    """Semantic search across an indexed repository.

    \b
    Finds code by concept, not just by keyword. Useful for exploring
    unfamiliar codebases or locating logic without knowing the exact
    function name.

    \b
    Requires the repo to be indexed first:
      agent index . --collection myproject

    \b
    Results show file path, line range, chunk type (function/class/chunk),
    and a relevance score (higher = more relevant).

    \b
    Examples:
      agent --collection myproject search "rate limiting logic"
      agent --collection myproject search "JWT token validation" -n 10
      agent --collection myproject search "database connection pooling"
      agent --collection myproject search "how does authentication work"
    """
    agent = _get_agent(ctx)
    results = agent.search(query, n=n_results)

    if not results:
        console.print(
            "[yellow]No results.[/yellow] "
            "Have you indexed a repo yet?\n"
            "[dim]Run: agent index <path> --collection <name>[/dim]"
        )
        return

    for i, r in enumerate(results, 1):
        m = r["metadata"]
        score = 1 - r["distance"]
        console.print(Panel(
            f"[bold]{m['file']}[/bold]  lines {m['start_line']}–{m['end_line']}  "
            f"({m['type']}: [italic]{m['name']}[/italic])  [dim]score {score:.2f}[/dim]\n\n"
            f"```\n{r['content'][:400]}{'...' if len(r['content']) > 400 else ''}\n```",
            title=f"Result {i}",
            border_style="dim",
        ))


# ------------------------------------------------------------------ #
#  index                                                               #
# ------------------------------------------------------------------ #

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument("repo_path", default=".", metavar="[PATH]")
@click.option(
    "--collection", "-C",
    default=None,
    metavar="NAME",
    help=(
        "Collection name to store chunks under. Use a different name per repo "
        "to keep them separate. Defaults to the global --collection or 'default'."
    ),
)
@click.pass_context
def index(ctx, repo_path, collection):
    """Index a repository into the vector database for semantic search and RAG.

    \b
    Walks the repo, chunks source files, embeds each chunk with nomic-embed-text,
    and stores them in ChromaDB. Once indexed, review and debug commands
    automatically retrieve relevant context from the collection.

    \b
    Supported file types:
      .py .js .ts .tsx .jsx .go .rs .java .cpp .c .h
      .rb .sh .yaml .yml .toml .json .md .txt .sql .graphql

    \b
    Python files are chunked by function and class (AST-aware).
    All other files are chunked by line windows with overlap.

    \b
    Safe to re-run — chunks are upserted, not duplicated.
    Re-index whenever the codebase changes significantly.

    \b
    Examples:
      agent index .
      agent index ~/projects/myrepo --collection myrepo
      agent --collection myrepo index ~/projects/myrepo
    """
    cfg = ctx.obj["cfg"]
    col = collection or ctx.obj.get("collection")
    agent = build_agent(cfg, collection=col)

    if agent.db is None:
        console.print("[red]No vector DB configured. Set memory.db_path in config.yaml[/red]")
        sys.exit(1)

    console.print(f"[dim]Indexing {repo_path} into collection '{agent.collection}' ...[/dim]")

    files_done = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Indexing...", total=None)

        def on_file(path: str, n_chunks: int):
            nonlocal files_done
            files_done += 1
            progress.update(task, description=f"[dim]{path}[/dim] ({n_chunks} chunks)")

        total = agent.index(repo_path, progress_cb=on_file)

    console.print(
        f"[green]Done.[/green] Indexed {total} chunks from {files_done} files "
        f"into collection '[bold]{agent.collection}[/bold]'."
    )


# ------------------------------------------------------------------ #
#  chat                                                                #
# ------------------------------------------------------------------ #

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def chat(ctx):
    """Interactive multi-turn chat with the LLM.

    \b
    Maintains full conversation history within the session so the model
    remembers what was said earlier. Good for:
      • Asking questions about architecture or design
      • Exploratory discussions about how to approach a problem
      • Getting code suggestions interactively

    \b
    Controls:
      Type your message and press Enter to send.
      Type 'exit', 'quit', or press Ctrl-C to end the session.

    \b
    Example:
      agent chat
      agent --collection myproject chat
    """
    agent = _get_agent(ctx)
    console.print(f"[bold]Chat[/bold] — {agent.llm.name}  [dim](type 'exit' to quit)[/dim]\n")

    history: list[Message] = []
    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if not user_input:
            continue

        history.append(Message("user", user_input))
        console.print("[bold green]Agent:[/bold green] ", end="")

        response = ""
        for token in agent.llm.stream(history):
            print(token, end="", flush=True)
            response += token
        print()

        history.append(Message("assistant", response))


# ------------------------------------------------------------------ #
#  serve                                                               #
# ------------------------------------------------------------------ #

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--host",
    default=None,
    metavar="HOST",
    help="Host to bind to. Defaults to value in config.yaml (127.0.0.1). "
         "Use 0.0.0.0 to expose on all interfaces.",
)
@click.option(
    "--port",
    default=None,
    type=int,
    metavar="PORT",
    help="Port to listen on. Defaults to value in config.yaml (8000).",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable hot-reload on code changes. For development only.",
)
@click.pass_context
def serve(ctx, host, port, reload):
    """Start the FastAPI REST API server.

    \b
    All CLI features are available as HTTP endpoints. Supports Server-Sent
    Events (SSE) streaming — pass "stream": true in the request body.

    \b
    Endpoints:
      GET  /health          Health check
      POST /review          Review code or a diff
      POST /debug           Debug an error or traceback
      POST /explain         Explain a file
      POST /chat            Multi-turn chat
      POST /search          Semantic search
      POST /index           Index a repository
      GET  /collections     List collections and chunk counts

    \b
    Interactive API docs are served at /docs once the server is running.

    \b
    Examples:
      agent serve
      agent serve --host 0.0.0.0 --port 8080
      agent serve --reload

    \b
    Example API call:
      curl http://localhost:8000/review \\
        -H "Content-Type: application/json" \\
        -d '{"content": "def add(a,b): return a-b", "filename": "math.py"}'
    """
    import uvicorn
    cfg = ctx.obj["cfg"]
    api_cfg = cfg.get("api", {})
    h = host or api_cfg.get("host", "127.0.0.1")
    p = port or api_cfg.get("port", 8000)
    console.print(f"[bold]Starting API server[/bold] on http://{h}:{p}")
    console.print(f"  Docs:   http://{h}:{p}/docs")
    console.print(f"  Health: http://{h}:{p}/health\n")
    uvicorn.run("agent.api:app", host=h, port=p, reload=reload)


# ------------------------------------------------------------------ #
#  models                                                              #
# ------------------------------------------------------------------ #

@cli.command(context_settings=CONTEXT_SETTINGS)
def models():
    """List locally available Ollama models.

    \b
    Shows all models currently installed on your machine via Ollama.
    To change the active model, edit the 'model' field in config.yaml
    or ~/.config/agent/config.yaml.

    \b
    To pull a new model:
      ollama pull qwen3:8b
      ollama pull nomic-embed-text

    \b
    Example:
      agent models
    """
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        console.print("[red]Ollama is not running or not installed.[/red]")
        console.print("[dim]Start it with: ollama serve[/dim]")
        sys.exit(1)
    console.print(result.stdout)


# ------------------------------------------------------------------ #
#  collections                                                         #
# ------------------------------------------------------------------ #

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def collections(ctx):
    """List all vector DB collections and their chunk counts.

    \b
    Each collection holds the indexed chunks of one or more repositories.
    Use a separate collection per project to keep search results scoped.

    \b
    To create a collection, index a repo into it:
      agent index ~/projects/myrepo --collection myrepo

    \b
    To use a collection for search or review:
      agent --collection myrepo search "authentication logic"
      agent --collection myrepo review src/main.py

    \b
    Example:
      agent collections
    """
    cfg = ctx.obj["cfg"]
    mem_cfg = cfg.get("memory", {})
    db_path = mem_cfg.get("db_path")
    if not db_path:
        console.print("[red]memory.db_path not set in config.yaml[/red]")
        sys.exit(1)

    from .memory.vectordb import VectorDB
    db = VectorDB(db_path)
    cols = db.list_collections()
    if not cols:
        console.print(
            "[yellow]No collections found.[/yellow]\n"
            "[dim]Index a repo first: agent index <path> --collection <name>[/dim]"
        )
        return
    for name in cols:
        count = db.collection_count(name)
        console.print(f"  [bold]{name}[/bold]  —  {count} chunks")


if __name__ == "__main__":
    cli()
