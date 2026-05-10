"""CLI entrypoint: agent <command> [options]"""
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

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

@click.group()
@click.option("--config", "-c", default=None, help="Path to config.yaml")
@click.option("--collection", default=None, help="Vector DB collection name")
@click.pass_context
def cli(ctx, config, collection):
    """Local AI agent for code review, debugging, and repo maintenance."""
    ctx.ensure_object(dict)
    cfg = load_config(config)
    ctx.obj["cfg"] = cfg
    ctx.obj["collection"] = collection


def _get_agent(ctx):
    return build_agent(ctx.obj["cfg"], collection=ctx.obj.get("collection"))


# ------------------------------------------------------------------ #
#  review                                                              #
# ------------------------------------------------------------------ #

@cli.command()
@click.argument("target", required=False)
@click.option("--diff", "mode", flag_value="diff", help="Review unstaged git diff")
@click.option("--staged", "mode", flag_value="staged", help="Review staged git diff")
@click.option("--no-stream", is_flag=True, help="Disable streaming output")
@click.pass_context
def review(ctx, target, mode, no_stream):
    """Review a file or git diff.

    \b
    Examples:
      agent review src/main.py
      agent review --diff
      agent review --staged
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
        console.print("[red]Provide a file path, --diff, or --staged[/red]")
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

@cli.command()
@click.argument("error_input", required=False)
@click.option("--file", "-f", "error_file", default=None, help="Read error from a file")
@click.option("--hint", default="", help="Extra context hint for the search")
@click.option("--no-stream", is_flag=True)
@click.pass_context
def debug(ctx, error_input, error_file, hint, no_stream):
    """Debug an error message or traceback.

    \b
    Examples:
      agent debug "TypeError: 'NoneType' object is not subscriptable"
      agent debug --file error.log
      cat error.log | agent debug
    """
    if error_file:
        error = Path(error_file).read_text(errors="replace")
    elif error_input:
        error = error_input
    elif not sys.stdin.isatty():
        error = sys.stdin.read()
    else:
        console.print("[red]Provide an error string, --file, or pipe via stdin[/red]")
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

@cli.command()
@click.argument("target")
@click.option("--no-stream", is_flag=True)
@click.pass_context
def explain(ctx, target, no_stream):
    """Explain what a file does.

    \b
    Example:
      agent explain src/auth/middleware.py
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

@cli.command()
@click.argument("query")
@click.option("-n", default=5, show_default=True, help="Number of results")
@click.pass_context
def search(ctx, query, n):
    """Semantic search across indexed repositories.

    \b
    Example:
      agent search "authentication middleware"
    """
    agent = _get_agent(ctx)
    results = agent.search(query, n=n)

    if not results:
        console.print("[yellow]No results — have you indexed a repo yet? Run: agent index <path>[/yellow]")
        return

    for i, r in enumerate(results, 1):
        m = r["metadata"]
        score = 1 - r["distance"]
        console.print(Panel(
            f"[bold]{m['file']}[/bold]  lines {m['start_line']}–{m['end_line']}  "
            f"({m['type']}: {m['name']})  [dim]score {score:.2f}[/dim]\n\n"
            f"```\n{r['content'][:400]}{'...' if len(r['content']) > 400 else ''}\n```",
            title=f"Result {i}",
            border_style="dim",
        ))


# ------------------------------------------------------------------ #
#  index                                                               #
# ------------------------------------------------------------------ #

@cli.command()
@click.argument("repo_path", default=".")
@click.option("--collection", "-C", default=None, help="Override collection name")
@click.pass_context
def index(ctx, repo_path, collection):
    """Index a repository into the vector database.

    \b
    Example:
      agent index ~/projects/myrepo
      agent index . --collection myrepo
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

    console.print(f"[green]Done.[/green] Indexed {total} chunks from {files_done} files "
                  f"into collection '{agent.collection}'.")


# ------------------------------------------------------------------ #
#  chat                                                                #
# ------------------------------------------------------------------ #

@cli.command()
@click.pass_context
def chat(ctx):
    """Interactive chat with the LLM (Ctrl-C or 'exit' to quit).

    \b
    Example:
      agent chat
    """
    agent = _get_agent(ctx)
    console.print(f"[bold]Chat[/bold] — {agent.llm.name}  (type 'exit' to quit)\n")

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
#  models                                                              #
# ------------------------------------------------------------------ #

@cli.command()
def models():
    """List locally available Ollama models."""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        console.print("[red]Ollama not running or not installed.[/red]")
        sys.exit(1)
    console.print(result.stdout)


# ------------------------------------------------------------------ #
#  collections                                                         #
# ------------------------------------------------------------------ #

@cli.command()
@click.pass_context
def collections(ctx):
    """List vector DB collections and their chunk counts."""
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
        console.print("[yellow]No collections found.[/yellow]")
        return
    for name in cols:
        count = db.collection_count(name)
        console.print(f"  [bold]{name}[/bold]  —  {count} chunks")


# ------------------------------------------------------------------ #
#  serve                                                               #
# ------------------------------------------------------------------ #

@cli.command()
@click.option("--host", default=None, help="Bind host (default from config)")
@click.option("--port", default=None, type=int, help="Bind port (default from config)")
@click.option("--reload", is_flag=True, help="Enable hot-reload (dev mode)")
@click.pass_context
def serve(ctx, host, port, reload):
    """Start the FastAPI REST API server.

    \b
    Example:
      agent serve
      agent serve --host 0.0.0.0 --port 8080
    """
    import uvicorn
    cfg = ctx.obj["cfg"]
    api_cfg = cfg.get("api", {})
    h = host or api_cfg.get("host", "127.0.0.1")
    p = port or api_cfg.get("port", 8000)
    console.print(f"[bold]Starting API server[/bold] on http://{h}:{p}")
    console.print(f"  Docs: http://{h}:{p}/docs")
    uvicorn.run("agent.api:app", host=h, port=p, reload=reload)


if __name__ == "__main__":
    cli()
