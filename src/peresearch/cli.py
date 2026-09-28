"""peresearch command line.

    peresearch add ~/notes ~/projects/x/docs    declare folders peresearch may read
    peresearch remove ~/notes                   stop reading a folder (its chunks leave the index)
    peresearch folders                          list the declared folders
    peresearch index                            (re)index the declared folders, on this machine
    peresearch find "query"                     search your files only; no model call, nothing leaves
"""

import argparse
import sys
from pathlib import Path

from peresearch import guard


def _print(text: str = "") -> None:
    print(guard.sanitize(text))


def cmd_add(args):
    new = [Path(p).expanduser().resolve() for p in args.paths]
    for p in new:
        if not p.is_dir():
            sys.exit(f"not a folder: {p}")
        if guard.denied(p):
            sys.exit(f"refused: {p} matches a protected pattern (keys, tokens, credentials)")
    guard.set_roots(guard.roots() + new)
    _print("folders: " + ", ".join(map(str, guard.roots())))


def cmd_remove(args):
    drop = {Path(p).expanduser().resolve() for p in args.paths}
    guard.set_roots([r for r in guard.roots() if r not in drop])
    _print("folders: " + (", ".join(map(str, guard.roots())) or "(none)") + "\nrun `peresearch index` to drop their chunks")


def cmd_folders(args):
    _print("\n".join(map(str, guard.roots())) or "(no folders declared; use `peresearch add`)")


def cmd_index(args):
    from peresearch.zetokrag.index import Index

    if not guard.roots():
        sys.exit("no folders declared; use `peresearch add`")
    r = Index().update()
    _print(f"{r.added} added · {r.changed} changed · {r.unchanged} unchanged · {r.removed} removed · "
           f"{r.chunks} chunks · {r.denied} protected paths skipped")
    for status, paths in sorted(r.problems.items()):
        label = {"needs_ocr": "need OCR (no text layer)", "too_large": "too large", "binary": "binary",
                 "error": "could not be read", "large_folder": "folders skipped as datasets"}.get(status, status)
        _print(f"{len(paths)} {label}:")
        for p in paths[:20]:
            _print(f"  {p}")
    if r.withheld:
        _print(f"{sum(r.withheld.values())} chunks withheld because they look like credentials:")
        for p, n in list(r.withheld.items())[:20]:
            _print(f"  {p} ({n})")


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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="peresearch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
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
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
