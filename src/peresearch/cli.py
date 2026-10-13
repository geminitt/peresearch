"""peresearch command line.

    peresearch                                  open the TUI; configure services with /model and folders with /folders
    peresearch --project research               open the TUI with this project's conversations
    peresearch setup                            optional terminal setup for scripted workflows
    peresearch chat                             compatibility alias for the TUI
    peresearch ask "question"                   one question to the agent: your files first, then the web
    peresearch find "query"                     search files; no chat-model call (configured search endpoints may receive text)
    peresearch add ~/notes ~/projects/x/docs    declare folders peresearch may read (also A in the TUI /folders window)
    peresearch remove ~/notes                   stop reading a folder (its chunks leave the index)
    peresearch folders                          list the declared folders
    peresearch index                            (re)index the declared folders, on this machine
"""

import argparse
import getpass
import os
import re
import signal
import sys
from pathlib import Path

from peresearch import guard, settings
from peresearch.workspace import Workspace, summary


def _print(text: str = "") -> None:
    print(guard.sanitize(text))


def cmd_add(args):
    ws = Workspace()
    for p in args.paths:
        try:
            ws.add(p)
        except ValueError as e:
            sys.exit(str(e))
    _print("folders: " + ", ".join(map(str, ws.folders())) + "\nrun `peresearch index` (or open `peresearch chat`)")


def cmd_remove(args):
    ws = Workspace()
    for p in args.paths:
        ws.remove(p)
    _print("folders: " + (", ".join(map(str, ws.folders())) or "(none)") + "\nrun `peresearch index` to drop their chunks")


def cmd_folders(args):
    _print("\n".join(map(str, guard.roots())) or "(no folders declared; use `peresearch add`)")


STOPPED = 130                                  # the shell's status for a command ended by Ctrl+C (128 + SIGINT)


def _interrupt(signum, frame):
    # One Ctrl+C reaches the process twice under `pixi run` (the terminal signals the whole group and pixi forwards
    # it again, measured): the first stops the command, a later one would break the clean stop with a traceback.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise KeyboardInterrupt


def _update(ws) -> int:
    """Update the index; Ctrl+C keeps what was committed (the update commits as it goes) and says so."""
    try:
        _print(summary(ws.update()))
    except KeyboardInterrupt:
        print("stopped: what was saved is kept (an update saves its work every minute); "
              "`peresearch index` goes on from there", file=sys.stderr)
        return STOPPED
    return 0


def cmd_index(args):
    if not guard.roots():
        sys.exit("no folders declared; use `peresearch add`")
    return _update(Workspace())


def _check_rerank_depth():
    """A bad PERESEARCH_RERANK_DEPTH stops a command before it starts, with the setting's name and range."""
    from peresearch.zetokrag.search import rerank_depth

    try:
        rerank_depth()
    except ValueError as e:
        raise SystemExit(f"{e}; it is set in the environment or in {guard.home() / 'settings.env'}") from None


def cmd_find(args):
    from peresearch.zetokrag.index import Index
    from peresearch.zetokrag.search import Searcher

    _check_rerank_depth()

    searcher = Searcher(Index())
    searcher.allowed = lambda path: guard.allowed(Path(path))
    hits, stale, total = searcher.find(args.query, k=args.k, page=args.page)
    if total == 0:
        _print("Nothing in your files matched.")
        return
    first, last = (args.page - 1) * args.k + 1, min(args.page * args.k, total)
    _print(f"results {first}–{last} of {total}" if first <= total else f"no results on page {args.page} of {total}")
    for h in hits:
        _print(f"\n{h.where}  [{'not reranked' if h.score is None else f'{h.score:.2f}'}]"
               + (f"  §{h.section}" if h.section else "")
               + ("  (file changed since indexing; excerpt still present)" if h.file_changed else ""))
        _print("  " + h.text.replace("\n", "\n  "))
        if h.copies:
            _print(f"  (also in: {', '.join(h.copies)})")
    if stale:
        _print(f"\n{stale} excerpts dropped: their files changed; run `peresearch index`")
    if last < total:
        _print(f"more: --page {args.page + 1}")


def make_agent(project: str = "default", on_event=None, index=None):
    """The agent with its real parts: the local index, the web providers that have keys, the scholarly sources,
    the model endpoint."""
    from peresearch.assembly import _assemble

    return _assemble(project=project, on_event=on_event, index=index)


def cmd_ask(args):
    from peresearch.tui import explain, render

    _check_rerank_depth()

    def event(kind, detail):
        print(guard.sanitize(f"· {kind}: {detail}" if detail else f"· {kind}"), file=sys.stderr)

    try:
        agent = make_agent(args.project, event)
        agent.toolbox.ask = ask_in_terminal if sys.stdin.isatty() else None     # nobody to ask: nothing outside read
        answer = agent.ask(args.question)
    except Exception as e:                    # what went wrong and what to do, not a traceback
        print(re.sub(r"</?sub>|\*\*|`", "", explain(e)).replace("/retry asks again. ", "").lstrip("> "), file=sys.stderr)
        return 1
    _print(render(answer))


def ask_in_terminal(tool: str, path) -> str:
    """The command line's answer to "may the agent read this path outside the declared folders?"."""
    folder = path if path.is_dir() else path.parent
    reply = input(guard.sanitize(f"peresearch wants to {tool} {path}, outside your declared folders. "
                                 f"Allow [o]nce, [s]ession ({folder}), or [N]o? ")).strip().lower()
    return {"o": "once", "once": "once", "s": "session", "session": "session"}.get(reply, "no")


def _shown(path) -> str:
    """A path as people write it: under the home folder, with ~."""
    home = Path.home()
    return f"~/{Path(path).relative_to(home).as_posix()}" if Path(path).is_relative_to(home) else str(path)


def _model() -> str:
    url = settings.get("PERESEARCH_LLM_URL")
    return url.split("//")[-1].split("/")[0] if url else "not set: /model chooses it"


def welcome(project: str) -> str:
    """What the interface opens with, as Claude Code does: the model, the folders, and where to go next."""
    folders = sorted(_shown(p) for p in guard.outermost(guard.roots()))
    shown = (", ".join(folders[:3]) + (f" and {len(folders) - 3} more" if len(folders) > 3 else "") if folders
             else "none yet: /add <folder> lets peresearch read one")
    lines = ["✻ peresearch · research over your files and the web", "", f"  model    {_model()}",
             f"  folders  {shown}"]
    if project != "default":
        lines.append(f"  project  {project}")
    return "\n".join(lines + ["", "/help for the commands · /status for the setup and the index"])


def status(project: str, index, reason: str | None = None) -> str:
    """/status (Markdown): the model, the search services, the folders, what the index holds and what it left out by
    reason; with a reason, the files left out for it."""
    from peresearch.workspace import PROBLEMS, count

    label = lambda r: PROBLEMS.get("data_file" if r == "data" else r, r)          # noqa: E731
    held = index.coverage()
    if reason is not None:
        if not held.get(reason):
            return f"No files left out for `{reason}`. The reasons: " + ", ".join(f"`{r}`" for r in sorted(held)) + "."
        paths, total = index.left_out(reason, limit=50)
        more = [f"- … and {total - len(paths):,} more"] if total > len(paths) else []
        return "\n".join([f"**{total:,} {label(reason)}**"] + [f"- {_shown(p)}" for p in paths] + more)
    server = settings.get("PERESEARCH_SEARCH_MCP")
    if server.startswith(("http://", "https://")):          # a key may sit in its query: never shown
        from urllib.parse import urlsplit
        parts = urlsplit(server)
        server = f"{parts.scheme}://{parts.netloc}{parts.path}"
    papers = (["OpenAlex"] if settings.get("OPENALEX_API_KEY") else []) + ["Semantic Scholar", "arXiv"]
    files, chunks = index.held()
    lines = [f"- **Model** {_model()}",
             f"- **Search** web: {guard.redact(server)[0] if server else 'none (add a search MCP server in /settings)'}"
             " · papers: "
             f"{', '.join(papers)} · Wikipedia",
             f"- **Project** {project} (its conversations: /resume)",
             "- **Folders** " + (", ".join(sorted(_shown(p) for p in guard.outermost(guard.roots()))) or "none"),
             f"- **Index** {count(files, 'file')} · {count(chunks, 'chunk')}"]
    if held:
        lines += ["", "**Left out of the search by meaning** (grep and read still reach them, protected paths "
                      "excepted):"]
        lines += [f"- {n:,} {label(r)} · `/status {r}`" for r, n in sorted(held.items(), key=lambda x: (-x[1], x[0]))]
    return "\n".join(lines)


def cmd_chat(args):
    from peresearch.tui import Chat

    _check_rerank_depth()
    ws = Workspace()
    app = Chat(lambda on_event: make_agent(args.project, on_event, ws.index), welcome=lambda: welcome(args.project),
         status=lambda reason=None: status(args.project, ws.index, reason), workspace=ws,
         project=args.project)
    app.setup_on_start = True
    app.run()


KEYS = settings.FIELDS


def cmd_setup(args, ask=input, secret=getpass.getpass):
    """Choose folders, index them, and store the settings (settings.env, readable by you only)."""
    ws = Workspace()
    _print("Folders peresearch may read: " + (", ".join(map(str, ws.folders())) or "none yet"))
    while True:
        path = ask("Add a folder (empty to go on): ").strip()
        if not path:
            break
        try:
            _print(f"  added {ws.add(path)}")
        except ValueError as e:
            _print(f"  {e}")
    if ws.folders():
        _print("Indexing (only new or changed files)…")
        if _update(ws):
            return STOPPED
    path = guard.home() / "settings.env"
    current = settings.saved()
    _print(f"\nSettings go to {path}. Enter keeps the current value.")
    for name, label, hidden in KEYS:
        shown = "(set)" if current.get(name) and hidden else current.get(name, "")
        value = (secret if hidden else ask)(f"{label} [{shown}]: ").strip()
        if value:
            current[name] = value
    settings.save(current)
    if settings.get("PERESEARCH_LLM_URL"):
        _print("\nChecking the model…")
        print_check()
    print_endpoints()
    _print("Saved. Start with `peresearch chat`.")


def print_check() -> int:
    """The model check (llm.check_model) on the model in the settings, printed; 0 when it calls tools."""
    from peresearch.llm import LLM, check_model
    try:
        works, lines = check_model(LLM())
    except Exception as e:
        _print(f"  the model could not be checked: {type(e).__name__}: {e}"[:400])
        return 1
    for line in lines:
        _print("  " + line)
    _print("  → it works with the agent" if works else
           "  → it does not call tools: the agent cannot search your files or the web with it")
    return 0 if works else 1


def print_endpoints() -> None:
    """The search's endpoints (zetokrag/embed.check): what is set, that it answers, where passages go; none, what that
    costs."""
    from peresearch.zetokrag import embed

    _print("\nChecking the search's endpoints…")
    try:
        lines = embed.check(*embed.from_settings())
    except Exception as e:                          # a bad URL in the settings: said, never a traceback
        lines = [f"the endpoints could not be checked: {type(e).__name__}: {e}"[:400]]
    lines += web_check()
    for line in lines:
        _print("  " + line)


def web_check() -> list[str]:
    """The web search's line(s) (web.check), the server started for it stopped after."""
    from peresearch import web

    try:
        searcher = web.McpSearch.from_settings()
    except ValueError as e:                         # PERESEARCH_SEARCH_MCP_KEY not NAME=value: said
        return [f"web search: {e}"]
    try:
        return web.check(searcher)
    finally:
        if searcher is not None:
            searcher.close()


def cmd_check(args):
    """Check the model in the settings (how it calls tools, where its reasoning comes back, its context, its speed)
    and the search's endpoints."""
    code = print_check()
    print_endpoints()
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(prog="peresearch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="default", help="conversation history to use in the TUI")
    ap.set_defaults(fn=cmd_chat)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    sub.add_parser("check", help="check the model in the settings").set_defaults(fn=cmd_check)
    for name, fn in [("add", cmd_add), ("remove", cmd_remove)]:
        p = sub.add_parser(name)
        p.add_argument("paths", nargs="+")
        p.set_defaults(fn=fn)
    sub.add_parser("folders").set_defaults(fn=cmd_folders)
    sub.add_parser("index").set_defaults(fn=cmd_index)
    p = sub.add_parser("find")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--page", type=int, default=1)
    p.set_defaults(fn=cmd_find)
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--project", default="default", help="conversation history to use and extend")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("chat")
    p.add_argument("--project", default="default", help="conversation history to use and extend")
    p.set_defaults(fn=cmd_chat)
    args = ap.parse_args(argv)
    guard.ensure_home()                        # every command, the interface too: the user's alone
    try:                                       # the user's own config.json, before any command reads it
        from peresearch import files
        files.config_patterns(guard.config(), Path.home())
    except guard.ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    previous = signal.signal(signal.SIGINT, _interrupt)
    try:
        code = args.fn(args) or 0              # a command's failure becomes the exit status
    except KeyboardInterrupt:                  # SIGINT stays ignored: the command ends here
        print("stopped", file=sys.stderr)
        return STOPPED
    if signal.getsignal(signal.SIGINT) is _interrupt:     # no Ctrl+C came (a command may have caught one itself)
        signal.signal(signal.SIGINT, previous)
    return code


if __name__ == "__main__":
    sys.exit(main())
