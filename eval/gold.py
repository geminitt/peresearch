"""The gold question set for ZetokRAG, built on a frozen copy of the owner's files.

    pixi run python eval/gold.py freeze --exclude '*/projects/minisafe/*'   # copy the indexed files once
    pixi run python eval/gold.py index                                      # index the copy (GPU, real models)
    pixi run python eval/gold.py pool [qid ...]                             # candidates for each question
    pixi run python eval/gold.py check                                      # every rule below; exit 1 on a fault
    pixi run python eval/gold.py index --into indexes/<name>                # an index of the copy for one setting
    pixi run python eval/gold.py evaluate <label> --index indexes/<name>    # score the dev split (test: once, last)
    pixi run python eval/gold.py compare <label-a> <label-b>                # noise: two runs, question by question

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
import math
import os
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
    assert len(manifest) + len(missing) == len(paths), "every indexed file is copied or listed as missing"
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
    assert q in "\n".join(norm(text[n]) for n in range(best[0], best[1] + 1)), "the located span lacks the quote"
    return {"file": file, "unit": kind, "start": best[0], "end": best[1], "quote": quote}


def overlap(question: str, quotes: list[str]) -> float:
    """Share of the question's words found in its quotes (the first design's word cut, kept fixed)."""
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
    assert all(int(i) in rows for i in ids), "the snapshot names a chunk the index does not hold"
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
        _, path, section, unit, start, end, text, *_ = rows[i]
        out.append({"chunk": i, "file": Path(path).relative_to(corpus).as_posix(), "unit": unit, "start": start,
                    "end": end, "section": section, "text": text, "found_by": sorted(found[i])})
    return out


# --- indexing the copy --------------------------------------------------------------------------------------------

def index_corpus(home: Path, into: str):
    """Index the frozen copy into home/`into`. The copy lives in peresearch's own folder, which the walk never enters
    for a user; while this index is built, peresearch's own folder is the index's (the copy is walked, the index
    and its extracted texts are not), then the setting is put back."""
    before = os.environ.get("PERESEARCH_HOME")
    target = home / into
    os.environ["PERESEARCH_HOME"] = str(target)
    try:
        return Index(target).update([home / "corpus"])
    finally:
        os.environ.pop("PERESEARCH_HOME") if before is None else os.environ.__setitem__("PERESEARCH_HOME", before)


# --- the ruler ----------------------------------------------------------------------------------------------------

def score(ranked: list[dict], judged: list[dict], k: int = 10, groups: list[int] | None = None) -> dict | None:
    """A question's ranked list against its judged passages, by overlapping lines (file, unit, start, end), so that
    any chunking can be scored: nDCG@k with the grade as gain, each judged passage credited once, to the first chunk
    that reaches it (a second chunk on it gains nothing); recall@3, @k and @30 of the passages graded 2; and the
    share of the top k that reaches no judged passage (unjudged: pooling bias to watch). None when nothing judged
    has a grade above 0 (an unanswerable question: scored elsewhere).

    A ranked chunk reaches a passage through its own lines or those of the near-copies search folded into it
    (`also`); judged passages that are near-copies of each other (`groups`, from copy_groups) are one passage:
    credited once, with their best grade, and counted once in recall."""
    groups = list(range(len(judged))) if groups is None else groups
    assert len(groups) == len(judged), "one group a judged passage"
    best_of: dict[int, int] = {}
    for g, j in zip(groups, judged):
        best_of[g] = max(best_of.get(g, 0), j["grade"])
    best = sorted(best_of.values(), reverse=True)[:k]
    if not any(best):
        return None

    def overlap(c, j):
        return c["file"] == j["file"] and c["unit"] == j["unit"] and c["start"] <= j["end"] and j["start"] <= c["end"]

    def reaches(c, j):
        return overlap(c, j) or any(overlap(a, j) for a in c.get("also", ()))

    credited, dcg, unjudged = set(), 0.0, 0
    for r, c in enumerate(ranked[:k], start=1):
        hit = {i for i, j in enumerate(judged) if reaches(c, j)}
        unjudged += not hit
        # a group's best grade, whichever copy was reached: the ideal ranking holds that best (it credited the copy's
        # own grade, so finding the same text in a lower-graded copy scored less; 8 of 74 dev questions, 2026-10-05)
        dcg += max((best_of[groups[i]] for i in hit if groups[i] not in credited), default=0) / math.log2(r + 1)
        credited |= {groups[i] for i in hit}
    idcg = sum(g / math.log2(r + 1) for r, g in enumerate(best, start=1))
    answers = {groups[i] for i, j in enumerate(judged) if j["grade"] == 2}

    def recall(n):
        reached = {groups[i] for c in ranked[:n] for i, j in enumerate(judged) if groups[i] in answers and reaches(c, j)}
        return len(reached) / len(answers) if answers else None
    out = {f"ndcg@{k}": dcg / idcg, "recall@3": recall(3), f"recall@{k}": recall(k), "recall@30": recall(30),
           f"unjudged@{k}": unjudged / len(ranked[:k]) if ranked[:k] else 0.0}
    assert 0 <= out[f"ndcg@{k}"] <= 1 + 1e-9, f"nDCG {out[f'ndcg@{k}']} outside [0, 1]"
    return out


def copy_groups(judged: list[dict], corpus: Path) -> list[int]:
    """A group number per judged passage: passages whose texts are near-copies (search.near_copy) share one."""
    from peresearch.zetokrag.search import near_copy
    texts = [passage(corpus, j) or "" for j in judged]
    group = list(range(len(judged)))
    for i in range(len(judged)):
        for j in range(i):
            if group[j] == j and texts[i] and texts[j] and near_copy(texts[i], texts[j]):
                group[i] = j
                break
    return group


def judged_passages(q: dict, judgments: list[dict]) -> list[dict]:
    """A question's judged passages, with each evidence passage that no judged one already covers."""
    mine = [j for j in judgments if j["id"] == q["id"]]
    for e in q.get("evidence", []):
        if not any(j["file"] == e["file"] and j["unit"] == e["unit"] and j["start"] <= e["end"] and e["start"] <= j["end"]
                   and j["grade"] >= e["grade"] for j in mine):
            mine.append({**{k: e[k] for k in ("file", "unit", "start", "end")}, "grade": e["grade"]})
    return mine


def evaluate(searcher, questions: list[dict], judgments: list[dict], corpus: Path, out: Path) -> dict:
    """Score every answerable question's ranked list (Searcher.rank, before any cut) and write one line per
    question to `out`; the means over the scored questions."""
    rows = []
    with open(out, "w", encoding="utf-8") as f:
        for q in questions:
            ranked, scores, copies = searcher.candidates(pool_query(q))         # what the model is shown
            where = lambda r: {"file": Path(r[1]).relative_to(corpus).as_posix(), "unit": r[3], "start": r[4], "end": r[5]}
            got = [{**where(r), "also": [where(c) for c in copies.get(i, [])]} for i, r in enumerate(ranked)]
            judged = judged_passages(q, judgments)
            s = score(got, judged, groups=copy_groups(judged, corpus)) if q["answerable"] else None   # answerable only
            line = {"id": q["id"], "lang": q["lang"], "type": q["type"], "answerable": q["answerable"],
                    # null when the first is not reranked (a depth under N_RERANK): NaN is not JSON
                    "best": (None if math.isnan(scores[0]) else float(scores[0])) if len(scores) else 0.0, "metrics": s,
                    "ranked": [[g["file"], g["unit"], g["start"], g["end"]] for g in got],
                    "also": {str(i): [[a["file"], a["unit"], a["start"], a["end"]] for a in g["also"]]
                             for i, g in enumerate(got) if g["also"]}}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(line)
    scored = [r["metrics"] for r in rows if r["metrics"]]
    return {k: sum(m[k] for m in scored if m[k] is not None) / max(1, sum(m[k] is not None for m in scored))
            for k in ("ndcg@10", "recall@3", "recall@10", "recall@30", "unjudged@10")} | {"scored": len(scored)}


def paired(new: list[float], old: list[float], resamples: int = 10_000, seed: int = 0) -> dict:
    """Per-question differences new − old: their mean, a 95% interval from a paired bootstrap (the questions drawn
    again with replacement, the mean taken each time), and how many questions won, tied and lost."""
    import random
    d = [a - b for a, b in zip(new, old, strict=True)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(resamples))
    return {"mean": sum(d) / len(d), "low": means[int(0.025 * resamples)], "high": means[int(0.975 * resamples) - 1],
            "wins": sum(x > 1e-9 for x in d), "ties": sum(abs(x) <= 1e-9 for x in d), "losses": sum(x < -1e-9 for x in d)}


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


def split(questions: list[dict], share: float = 0.3, seed: int = 0) -> list[dict]:
    """Put `share` of each language's questions in the test split (rounded), drawn with a seed, then make sure every
    question type has one there (swapping within the language), the rest in dev. Done once: a second draw would let
    the test split be chosen after looking at it, so it is refused."""
    import random
    if any(q.get("split") == "test" for q in questions):
        raise SystemExit("a test split exists already: it is drawn once")
    test = set()
    for lang in LANGS:
        ids = sorted(q["id"] for q in questions if q.get("lang") == lang)
        if ids:
            test |= set(random.Random(f"{seed}:{lang}").sample(ids, max(1, round(share * len(ids)))))
    by_id = {q["id"]: q for q in questions}
    for t in TYPES:
        of_type = sorted(q["id"] for q in questions if q.get("type") == t)
        if not of_type or any(i in test for i in of_type):
            continue
        for i in of_type:                       # swap a question of this type in for one of a type with spares
            lang = by_id[i]["lang"]
            spare = [j for j in sorted(test) if by_id[j]["lang"] == lang
                     and sum(by_id[k]["type"] == by_id[j]["type"] for k in test) > 1]
            if spare:
                test.discard(spare[0])
                test.add(i)
                break
    for q in questions:
        q["split"] = "test" if q["id"] in test else "dev"
    assert sum(q["split"] == "test" for q in questions) == len(test), "a test id that names no question"
    return questions


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
    ix = sub.add_parser("index")
    ix.add_argument("--into", default="index", help="the index folder under eval/gold")
    ix.add_argument("--chunking", default=None, help="a preset of zetokrag.chunk.PRESETS (default: chunk.DEFAULT)")
    ix.add_argument("--context", default=None, help="what precedes a chunk when ranked: one of index.CONTEXTS "
                                                    "(default: index.CONTEXT)")
    p = sub.add_parser("pool")
    p.add_argument("ids", nargs="*")
    sub.add_parser("check")
    ev = sub.add_parser("evaluate", help="score ZetokRAG on the dev split (the test split once, at the end)")
    ev.add_argument("label", help="a name for this run, e.g. baseline-2026-10-03")
    ev.add_argument("--index", default="index", help="index folder under eval/gold (built by `index --into`)")
    ev.add_argument("--split", default="dev", choices=["dev", "test"])
    cmp_ = sub.add_parser("compare", help="two runs question by question: the noise between identical runs")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    sub.add_parser("split")
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
        from peresearch.zetokrag import chunk
        chunk.DEFAULT = chunk.PRESETS[args.chunking] if args.chunking else chunk.DEFAULT
        if args.context:
            import peresearch.zetokrag.index as index_module
            assert args.context in index_module.CONTEXTS, f"context {args.context!r} not in {index_module.CONTEXTS}"
            index_module.CONTEXT = args.context
        r = index_corpus(home, args.into)
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
    if args.cmd == "split":
        qs = split(questions)
        (home / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
        print(f"test {sum(q['split'] == 'test' for q in qs)} of {len(qs)}")
        return 0
    if args.cmd == "evaluate":
        from peresearch.zetokrag.search import N_RERANK, Searcher
        runs = home / "runs"
        runs.mkdir(exist_ok=True)
        picked = [q for q in questions if q["split"] == args.split]
        # the ruler reranks all N_RERANK, whatever this machine's PERESEARCH_RERANK_DEPTH
        means = evaluate(Searcher(Index(home / args.index), depth=N_RERANK), picked, load_jsonl(home / "judgments.jsonl"),
                         corpus.resolve(), runs / f"{args.label}.jsonl")
        (runs / f"{args.label}.summary.json").write_text(json.dumps({"index": args.index, "split": args.split,
                                                                    **means}, indent=1))
        print(f"{args.label}: " + " · ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                             for k, v in means.items()))
        return 0
    if args.cmd == "compare":
        a, b = ({r["id"]: r for r in load_jsonl(home / "runs" / f"{x}.jsonl")} for x in (args.a, args.b))
        same_list = sum(a[i]["ranked"] == b[i]["ranked"] for i in a if i in b)
        diffs = [abs(a[i]["metrics"]["ndcg@10"] - b[i]["metrics"]["ndcg@10"]) for i in a
                 if i in b and a[i]["metrics"] and b[i]["metrics"]]
        print(f"{same_list}/{len(a)} questions ranked identically; nDCG@10 changed on {sum(d > 1e-9 for d in diffs)} "
              f"of {len(diffs)}, by {max(diffs, default=0):.4f} at most, {sum(diffs) / max(1, len(diffs)):.4f} on average")
        both = [i for i in a if i in b and a[i]["metrics"] and b[i]["metrics"]]
        for m in ("ndcg@10", "recall@10", "recall@30"):
            pairs = [(b[i]["metrics"][m], a[i]["metrics"][m]) for i in both
                     if a[i]["metrics"][m] is not None and b[i]["metrics"][m] is not None]
            r = paired([x for x, _ in pairs], [y for _, y in pairs])
            print(f"{args.b} − {args.a}, {m}: {r['mean']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}] 95% paired "
                  f"bootstrap; {r['wins']} better, {r['ties']} same, {r['losses']} worse of {len(pairs)}")
        return 0
    if args.cmd == "pool":
        from peresearch.zetokrag.search import N_RERANK, Searcher
        index = Index(home / "index")
        searcher = Searcher(index, depth=N_RERANK)
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
