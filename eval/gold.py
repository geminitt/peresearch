"""The gold question set for ZetokRAG, built on a frozen copy of the owner's files.

    pixi run python eval/gold.py freeze --exclude '*/projects/minisafe/*'   # copy the indexed files once
    pixi run python eval/gold.py index                                      # index the copy (GPU, real models)
    pixi run python eval/gold.py pool [qid ...]                             # candidates for each question
    pixi run python eval/gold.py check                                      # every rule below; exit 1 on a fault

Everything lives in $PERESEARCH_HOME/eval/gold/ (private: it quotes the owner's files), never in the repository:
corpus/ (the frozen files, under their path relative to the home folder), manifest.json (path, size, sha256),
index/, questions.jsonl, pools/<id>.json and judgments.jsonl.

Method (approved by the owner, 2026-10-02):
- Question types after CRAG (Yang et al., 2024) and the plan: factoid, conditional, set, comparison, aggregation,
  multi_hop, false_premise, unanswerable and follow_up; the question's language is a field of its own, so a
  question in another language than its evidence is cross-lingual. A false-premise question is answerable: its
  answer says the premise is wrong, with the passage that shows it.
- Evidence is a passage — file, unit (line, page or cell), start, end and a verbatim quote — not an index chunk, so
  the set outlives a change of chunking. Each quote is checked against the frozen file.
- Every passage that any of four searches returns for a question (TREC-style pooling: grep for the question's probe
  terms, BM25, the dense embedder and full ZetokRAG, each to depth POOL_DEPTH) is judged: 2 answers the question,
  1 helps answer it, 0 does not. An unanswerable question keeps the probes that were searched and all-zero
  judgments as its evidence of absence.
- Questions are written by the assistant, so they lean on the wording of their sources; the word overlap between a
  question and its quotes is reported so that easy, copied questions show.
"""

import argparse
import fnmatch
import hashlib
import json
import shutil
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))     # eval/tokenization.py
from tokenization import legacy_tokenize  # noqa: E402

from peresearch import guard
from peresearch.zetokrag import core, parse
from peresearch.zetokrag.index import Index

TYPES = ("factoid", "conditional", "set", "comparison", "aggregation", "multi_hop", "false_premise", "unanswerable",
         "follow_up")
LANGS = ("en", "vi", "zh", "ko", "ja", "hi")
POOL_DEPTH = 10
GRADES = (0, 1, 2)


def gold_home() -> Path:
    return guard.home() / "eval" / "gold"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- freezing -----------------------------------------------------------------------------------------------------

def freeze(live: Index, home: Path, user: Path, exclude: list[str]) -> list[dict]:
    """Copy every file the live index read (status ok) under `home`/corpus, at its path relative to `user`, except
    those matching an `exclude` pattern; write the manifest, and the files gone since indexing to missing.json.
    Done once: a second freeze would move the ground the labels stand on, so it is refused. The copy is made in a
    folder of its own and renamed at the end, so a failure leaves no half corpus."""
    corpus = home / "corpus"
    if corpus.exists():
        raise SystemExit(f"{corpus} exists: the corpus is frozen once")
    paths = sorted(p for (p,) in live.read("SELECT path FROM files WHERE status='ok'")
                   if not any(fnmatch.fnmatch(p, pat) for pat in exclude))
    partial = home / "corpus.partial"
    shutil.rmtree(partial, ignore_errors=True)
    manifest, missing = [], []
    try:
        for p in paths:
            src = Path(p)
            rel = src.relative_to(user)
            if not src.is_file():
                missing.append(rel.as_posix())
                continue
            dst = partial / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            manifest.append({"file": rel.as_posix(), "size": dst.stat().st_size, "sha256": sha(dst)})
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    (home / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (home / "missing.json").write_text(json.dumps(missing, indent=1))
    partial.rename(corpus)
    return manifest


# --- passages -----------------------------------------------------------------------------------------------------

def norm(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def units(path: Path) -> tuple[str, dict[int, str]]:
    """A file's unit kind and the text of each numbered unit, as the parser reads it (the index reads the same)."""
    doc = parse.read(path)
    out: dict[int, str] = {}
    kind = doc.blocks[0].unit if doc.blocks else "line"
    for b in doc.blocks:
        if b.unit == "line":
            for i, line in enumerate(b.text.split("\n")):
                out[b.start + i] = line
        else:
            out[b.start] = (out[b.start] + "\n" + b.text) if b.start in out else b.text
    return kind, out


def passage(corpus: Path, ev: dict) -> str | None:
    """The text of an evidence passage in the frozen file, or None if the file or the span is not there."""
    f = corpus / ev["file"]
    if not f.is_file():
        return None
    kind, text = units(f)
    if kind != ev["unit"] or any(n not in text for n in range(ev["start"], ev["end"] + 1)):
        return None
    return "\n".join(text[n] for n in range(ev["start"], ev["end"] + 1))


def locate(corpus: Path, file: str, quote: str) -> dict | None:
    """The smallest span of consecutive units of `file` whose text holds `quote` verbatim (the one ending first),
    as an evidence passage; None if the quote is not in the file. Spans are found by code so that no line is ever
    counted by hand."""
    f = corpus / file
    if not f.is_file():
        return None
    kind, text = units(f)
    q = norm(quote)
    runs, numbers = [], sorted(text)
    for n in numbers:                                   # a span cannot cross a missing unit (an empty cell, a page)
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])

    def joined(run, i, j):
        return "\n".join(norm(text[n]) for n in run[i:j + 1])
    best = None
    for run in runs:
        for j in range(len(run)):
            if q in joined(run, 0, j):
                i = max(i for i in range(j + 1) if q in joined(run, i, j))
                if best is None or run[j] < best[1]:
                    best = (run[i], run[j])
                break
    if best is None:
        return None
    return {"file": file, "unit": kind, "start": best[0], "end": best[1], "quote": quote}


def overlap(question: str, quotes: list[str]) -> float:
    """Share of the question's words found in its quotes (the cut is fixed, as in eval/personal.py)."""
    q = set(legacy_tokenize(question, folded=False))
    e = set(w for t in quotes for w in legacy_tokenize(t, folded=False))
    return len(q & e) / max(len(q), 1)


# --- pooling ------------------------------------------------------------------------------------------------------

def pool(index: Index, searcher, question: str, probes: list[str], corpus: Path, depth: int = POOL_DEPTH) -> list[dict]:
    """The union of what four searches return for a question, each to `depth`: grep for every probe term (case
    folded, in index order), BM25, the dense embedder and full ZetokRAG. One entry per chunk, with every search
    that found it."""
    ids, bm, vecs = index.snapshot()
    rows = {r[0]: r for r in index.rows()}
    found: dict[int, set] = {}

    def add(chunk_ids, how):
        for i in chunk_ids:
            found.setdefault(int(i), set()).add(how)

    for term in probes:
        t = term.casefold()
        add([i for i in ids if t in rows[int(i)][6].casefold()][:depth], "grep")
    if bm is not None:
        add(ids[core.top(bm.scores(question), depth)], "bm25")
        q = index.embedder.queries([question])[0].astype("float32")
        add(ids[core.top(vecs @ q, depth)], "dense")
        ranked, _ = searcher.rank(question)
        add([r[0] for r in ranked[:depth]], "zetokrag")
    out = []
    for i in sorted(found):
        _, path, section, unit, start, end, text, _ = rows[i]
        out.append({"chunk": i, "file": Path(path).relative_to(corpus).as_posix(), "unit": unit, "start": start,
                    "end": end, "section": section, "text": text, "found_by": sorted(found[i])})
    return out


def pool_query(q: dict) -> str:
    """What the searches are given for a question: a follow-up is read with the earlier questions of its
    conversation, as an agent would, or its passages would never come up."""
    return " ".join([h["q"] for h in q.get("history", [])] + [q["question"]])


def key(e: dict) -> tuple:
    return e["file"], e["unit"], e["start"], e["end"]


# --- recording judgments and evidence ----------------------------------------------------------------------------

def judge(home: Path, qid: str, grades: dict[int, int]) -> list[dict]:
    """Judge every pooled passage of a question: the grade given by its pool position in `grades`, 0 for the rest
    (a judge reads every passage and names only those that help). Replaces the question's earlier judgments."""
    entries = json.loads((home / "pools" / f"{qid}.json").read_text())
    if not set(grades) <= set(range(len(entries))) or not set(grades.values()) <= set(GRADES):
        raise ValueError(f"{qid}: positions {sorted(grades)} or grades {sorted(set(grades.values()))} out of range")
    mine = [{"id": qid, **{k: e[k] for k in ("file", "unit", "start", "end")}, "grade": grades.get(k, 0)}
            for k, e in enumerate(entries)]
    path = home / "judgments.jsonl"
    rest = [j for j in load_jsonl(path) if j["id"] != qid]
    path.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in rest + mine))
    return mine


def add_evidence(home: Path, qid: str, file: str, quote: str, grade: int) -> dict:
    """Add a passage to a question's evidence, located by its quote in the frozen file."""
    if grade not in (1, 2):
        raise ValueError("evidence is graded 1 (helps) or 2 (answers)")
    ev = locate(home / "corpus", file, quote)
    if ev is None:
        raise ValueError(f"{qid}: quote not found in {file}: {quote[:60]!r}")
    ev["grade"] = grade
    path = home / "questions.jsonl"
    qs = load_jsonl(path)
    q = next(q for q in qs if q["id"] == qid)
    if ev not in q.setdefault("evidence", []):
        q["evidence"].append(ev)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in qs))
    return ev


# --- checking -----------------------------------------------------------------------------------------------------

def check(questions: list[dict], corpus: Path, pools: dict[str, list[dict]], judgments: list[dict],
          test_share: tuple[float, float] = (0.25, 0.4)) -> tuple[list[str], dict]:
    """Every rule of the set; returns (faults, summary). An empty fault list means the set may be used."""
    faults, seen = [], set()
    judged: dict[str, dict[tuple, int]] = {}
    for j in judgments:
        if j.get("grade") not in GRADES:
            faults.append(f"{j.get('id')}: judgment grade {j.get('grade')!r} not in {GRADES}")
        judged.setdefault(j["id"], {})[key(j)] = j.get("grade")
    overlaps = {}
    for q in questions:
        qid = q.get("id")
        where = f"{qid}"
        if qid in seen:
            faults.append(f"{where}: duplicate id")
        seen.add(qid)
        for field, allowed in (("split", ("dev", "test")), ("type", TYPES), ("lang", LANGS)):
            if q.get(field) not in allowed:
                faults.append(f"{where}: {field} {q.get(field)!r} not in {allowed}")
        if not str(q.get("question", "")).strip():
            faults.append(f"{where}: no question")
        if not q.get("probes"):
            faults.append(f"{where}: no probe terms (the grep part of the pool)")
        if q.get("type") == "follow_up" and not q.get("history"):
            faults.append(f"{where}: a follow_up needs the earlier turns in `history`")
        answerable = q.get("answerable")
        if answerable is not (q.get("type") != "unanswerable"):
            faults.append(f"{where}: answerable={answerable!r} does not fit type {q.get('type')!r}")
        evidence = q.get("evidence", [])
        if answerable:
            if not str(q.get("answer") or "").strip():
                faults.append(f"{where}: answerable but no answer")
            if not any(e.get("grade") == 2 for e in evidence):
                faults.append(f"{where}: answerable but no passage graded 2")
        elif q.get("type") == "unanswerable" and (q.get("answer") is not None or any(e.get("grade") == 2 for e in evidence)):
            faults.append(f"{where}: unanswerable yet has an answer or an answering passage")
        quotes = []
        for e in evidence:
            text = passage(corpus, e)
            if text is None:
                faults.append(f"{where}: {e.get('file')} {e.get('unit')} {e.get('start')}-{e.get('end')} not found")
            elif not e.get("quote") or norm(e["quote"]) not in norm(text):
                faults.append(f"{where}: quote not verbatim in {e['file']}:{e['start']}-{e['end']}")
            if e.get("grade") not in (1, 2):
                faults.append(f"{where}: evidence grade {e.get('grade')!r} not 1 or 2")
            quotes.append(e.get("quote", ""))
        if qid not in pools:
            faults.append(f"{where}: not pooled")
        else:
            missing = [key(p) for p in pools[qid] if key(p) not in judged.get(qid, {})]
            if missing:
                faults.append(f"{where}: {len(missing)} pooled passages not judged")
            if q.get("type") == "unanswerable" and any(g == 2 for g in judged.get(qid, {}).values()):
                faults.append(f"{where}: unanswerable but a pooled passage was judged to answer it")
        if quotes:
            overlaps[qid] = overlap(q.get("question", ""), quotes)
    summary = {"questions": len(questions), "by_type": {}, "by_lang": {}, "overlap": overlaps}
    for t in TYPES:
        of_type = [q for q in questions if q.get("type") == t]
        test = sum(q.get("split") == "test" for q in of_type)
        summary["by_type"][t] = {"dev": len(of_type) - test, "test": test}
        if of_type and not test:
            faults.append(f"type {t}: no question in the test split")
    for lang in LANGS:
        summary["by_lang"][lang] = sum(q.get("lang") == lang for q in questions)
    if questions:
        share = sum(q.get("split") == "test" for q in questions) / len(questions)
        if not test_share[0] <= share <= test_share[1]:
            faults.append(f"test share {share:.2f} outside {test_share}")
    return faults, summary


# --- command line -------------------------------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gold")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--exclude", action="append", default=[])
    sub.add_parser("index")
    p = sub.add_parser("pool")
    p.add_argument("ids", nargs="*")
    sub.add_parser("check")
    j = sub.add_parser("judge")
    j.add_argument("id")
    j.add_argument("grades", nargs="*", help="pool position=grade, e.g. 5=2 3=1; every other passage gets 0")
    e = sub.add_parser("evidence")
    e.add_argument("id")
    e.add_argument("file")
    e.add_argument("grade", type=int)
    e.add_argument("quote")
    args = ap.parse_args(argv)
    home = gold_home()
    home.mkdir(parents=True, exist_ok=True)
    corpus = home / "corpus"
    if args.cmd == "freeze":
        manifest = freeze(Index(guard.home()), home, Path.home(), args.exclude)
        print(f"froze {len(manifest)} files, {sum(m['size'] for m in manifest) / 1e6:.1f} MB")
        return 0
    if args.cmd == "index":
        r = Index(home / "index").update([corpus])
        print(f"{r.added} added · {r.chunks} chunks")
        return 0
    if args.cmd == "judge":
        mine = judge(home, args.id, {int(k): int(v) for k, v in (g.split("=") for g in args.grades)})
        print(f"{args.id}: {len(mine)} judged, {sum(m['grade'] > 0 for m in mine)} helpful")
        return 0
    if args.cmd == "evidence":
        ev = add_evidence(home, args.id, args.file, args.quote, args.grade)
        print(f"{args.id}: {ev['file']} {ev['unit']} {ev['start']}-{ev['end']} grade {ev['grade']}")
        return 0
    questions = load_jsonl(home / "questions.jsonl")
    if args.cmd == "pool":
        from peresearch.zetokrag.search import Searcher
        index = Index(home / "index")
        searcher = Searcher(index)
        (home / "pools").mkdir(exist_ok=True)
        for q in questions:
            if args.ids and q["id"] not in args.ids:
                continue
            entries = pool(index, searcher, pool_query(q), q["probes"], corpus.resolve())
            (home / "pools" / f"{q['id']}.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1))
            print(f"{q['id']}: {len(entries)} passages")
        return 0
    pools = {p.stem: json.loads(p.read_text()) for p in (home / "pools").glob("*.json")} if (home / "pools").exists() else {}
    faults, summary = check(questions, corpus, pools, load_jsonl(home / "judgments.jsonl"))
    o = sorted(summary["overlap"].values())
    print(f"{summary['questions']} questions; by type {summary['by_type']}; by language {summary['by_lang']}")
    if o:
        print(f"question-to-quote word overlap: median {o[len(o) // 2]:.2f}, max {o[-1]:.2f}, "
              f"over 0.6: {sum(x > 0.6 for x in o)}")
    print("\n".join(faults) or "no faults")
    return 1 if faults else 0


if __name__ == "__main__":
    sys.exit(main())
