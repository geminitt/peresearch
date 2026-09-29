"""peresearch command line.

    peresearch setup                            first run: choose folders, index them, save the model and search keys
    peresearch chat                             the full-screen terminal interface (TUI); updates the index first
    peresearch ask "question"                   one question to the agent: your files first, then the web
    peresearch find "query"                     search your files only; no model call, nothing leaves
    peresearch add ~/notes ~/projects/x/docs    declare folders peresearch may read (also /add in the TUI)
    peresearch remove ~/notes                   stop reading a folder (its chunks leave the index)
    peresearch folders                          list the declared folders
    peresearch index                            (re)index the declared folders, on this machine
"""

import argparse
import getpass
import os
import re
import sys

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


def cmd_index(args):
    if not guard.roots():
        sys.exit("no folders declared; use `peresearch add`")
    _print(summary(Workspace().update()))


def cmd_find(args):
    from peresearch.zetokrag.index import Index
    from peresearch.zetokrag.search import Searcher

    hits, verdict, stale = Searcher(Index()).find(args.query, k=args.k)
    _print({"enough": "Your files cover this.", "partial": "Your files touch on this.",
            "none": "Nothing in your files seems to answer this."}[verdict])
    for h in hits:
        _print(f"\n{h.where}  [{h.score:.2f}]" + (f"  §{h.section}" if h.section else "")
               + ("  (file changed since indexing; excerpt still present)" if h.file_changed else ""))
        _print("  " + h.text.replace("\n", "\n  "))
    if stale:
        _print(f"\n{stale} excerpts dropped: their files changed; run `peresearch index`")


def make_agent(project: str = "default", on_event=None, index=None):
    """The agent with its real parts: the local index, the web providers that have keys, the model endpoint."""
    from peresearch.agent import Agent
    from peresearch.llm import LLM
    from peresearch.tools import Toolbox
    from peresearch.web import Web
    from peresearch.zetokrag.index import Index
    from peresearch.zetokrag.search import Searcher

    on_event = on_event or (lambda kind, detail: None)
    llm = LLM(on_wait=lambda msg: on_event("wait", msg))
    return Agent(llm, Toolbox(Searcher(index or Index()), Web()), project=project, on_event=on_event)


def cmd_ask(args):
    from peresearch.tui import explain, render

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


def heading(project: str) -> str:
    url = settings.get("PERESEARCH_LLM_URL")
    model = url.split("//")[-1].split("/")[0] if url else "no model endpoint set"
    web = [n for n, k in (("Tavily", "TAVILY_API_KEY"), ("Exa", "EXA_API_KEY")) if settings.get(k)]
    from peresearch.llm import Budget
    from peresearch.tools import outermost

    budget = Budget.from_settings(url) if url else None      # only an endpoint billed by the second has one
    cost = f" · today ≤ ${budget.spent():.2f} of ${budget.cap:.2f}" if budget else ""
    return (f"peresearch · project {project} · {len(outermost(guard.roots()))} folders · model {model}{cost} · "
            f"web {' → '.join(web) if web else 'off (no search key)'}")


def cmd_chat(args):
    from peresearch.tui import Chat

    ws = Workspace()
    Chat(lambda on_event: make_agent(args.project, on_event, ws.index), lambda: heading(args.project),
         workspace=ws, project=args.project).run()


KEYS = [("PERESEARCH_LLM_URL", "model endpoint (…/v1), e.g. http://localhost:8000/v1 or the URL `modal deploy` prints", False),
        ("PERESEARCH_LLM_KEY", "model key (a Modal proxy token wk-….ws-…; empty for a local server)", True),
        ("TAVILY_API_KEY", "Tavily key (free; empty to skip)", True),
        ("EXA_API_KEY", "Exa key (free credits; empty to skip)", True)]


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
        _print(summary(ws.update()))
    path = guard.home() / "settings.env"
    current = settings.saved()
    _print(f"\nSettings go to {path}. Enter keeps the current value.")
    for name, label, hidden in KEYS:
        shown = "(set)" if current.get(name) and hidden else current.get(name, "")
        value = (secret if hidden else ask)(f"{label} [{shown}]: ").strip()
        if value:
            current[name] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("".join(f"{k}={v}\n" for k, v in current.items()))
    os.chmod(path, 0o600)
    _print("Saved. Start with `peresearch chat`.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="peresearch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    for name, fn in [("add", cmd_add), ("remove", cmd_remove)]:
        p = sub.add_parser(name)
        p.add_argument("paths", nargs="+")
        p.set_defaults(fn=fn)
    sub.add_parser("folders").set_defaults(fn=cmd_folders)
    sub.add_parser("index").set_defaults(fn=cmd_index)
    p = sub.add_parser("find")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(fn=cmd_find)
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--project", default="default", help="conversation history to use and extend")
    p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("chat")
    p.add_argument("--project", default="default", help="conversation history to use and extend")
    p.set_defaults(fn=cmd_chat)
    args = ap.parse_args(argv)
    return args.fn(args) or 0                  # a command's failure becomes the exit status


if __name__ == "__main__":
    sys.exit(main())
