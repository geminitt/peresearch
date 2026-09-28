"""ZetokRAG on the user's own files: the reviewed question set in $PERESEARCH_HOME/eval/personal.jsonl.

    pixi run python eval/personal.py ~/projects/monodist ~/class ...     # folders to index for the run

The question set and the index stay outside the repository (they hold excerpts of private files); only the
aggregate numbers are printed. Two ways of scoring a hit:
- strict: it comes from the labelled file and overlaps the labelled lines (pages, cells);
- content: its words overlap the labelled passage by at least 80% (Jaccard), which also credits the same
  passage kept in another file (a solution notebook, a text export).
Each labelled passage counts once, at its best rank.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

from peresearch import guard
from peresearch.zetokrag import search
from peresearch.zetokrag.core import tokenize
from peresearch.zetokrag.index import Index
from peresearch.zetokrag.search import Searcher


def strict(row, r) -> bool:
    _, path, _, unit, start, end, _, _ = row
    return r["path"] == path and r["unit"] == unit and start <= r["end"] and r["start"] <= end


def jaccard(a: str, b: str) -> float:
    x, y = set(tokenize(a, folded=False)), set(tokenize(b, folded=False))
    return len(x & y) / max(len(x | y), 1)


def gains(rows, relevant, texts, mode) -> list[int]:
    """1 at the first rank where each labelled passage is found, 0 elsewhere."""
    found, out = set(), []
    for row in rows[:10]:
        g = 0
        for k, r in enumerate(relevant):
            ok = strict(row, r) or (mode == "content" and jaccard(row[6], texts[k]) >= 0.8)
            if ok and k not in found:
                found.add(k)
                g = 1
                break
        out.append(g)
    return out


def main(folders: list[str]) -> None:
    home = guard.home() / "eval"
    items = [json.loads(line) for line in open(home / "personal.jsonl")]
    idx = Index(home / "index", embedder=None)
    report = idx.update([Path(f).expanduser().resolve() for f in folders])
    print(f"indexed {report.chunks} chunks from {report.added + report.changed + report.unchanged} files")
    s = Searcher(idx)
    chunks = {(c["path"], c["unit"], c["start"], c["end"]): c["text"]
              for c in map(json.loads, open(home / "chunks.jsonl"))}
    ans, neg = {"strict": [], "content": []}, []
    for it in items:
        rows, scores = s.rank(it["question"])
        best = float(scores[0]) if len(scores) else 0.0
        if not it["answerable"]:
            neg.append(best)
            continue
        texts = [chunks.get((r["path"], r["unit"], r["start"], r["end"]), "") for r in it["relevant"]]
        for mode in ans:
            g = gains(rows, it["relevant"], texts, mode)
            dcg = sum(x / math.log2(i + 2) for i, x in enumerate(g))
            ideal = sum(1 / math.log2(i + 2) for i in range(min(len(it["relevant"]), 10)))
            ans[mode].append({"r1": g[0] == 1 if g else False, "r5": any(g[:5]), "r10": any(g), "ndcg": dcg / ideal,
                              "best": best})
    for mode, a in ans.items():
        print(f"answerable {len(a)} ({mode}): hit@1 {np.mean([x['r1'] for x in a]):.1%}, hit@5 {np.mean([x['r5'] for x in a]):.1%}, "
              f"hit@10 {np.mean([x['r10'] for x in a]):.1%}, nDCG@10 {np.mean([x['ndcg'] for x in a]):.3f}")
    ans = ans["strict"]
    print(f"best score: answerable min {min(x['best'] for x in ans):.3f}, median {np.median([x['best'] for x in ans]):.3f}; "
          f"unanswerable max {max(neg):.3f}, median {np.median(neg):.3f}")
    a_best = np.array([a["best"] for a in ans])
    auroc = float(np.mean([(x > y) + 0.5 * (x == y) for x in a_best for y in neg]))
    print(f"not answerable {len(neg)}: AUROC of the best score {auroc:.3f}")
    for name, tau in (("PARTIAL", search.PARTIAL), ("ENOUGH", search.ENOUGH)):
        print(f"  at {name} = {tau:.3f}: rejected {np.mean(np.array(neg) < tau):.1%} of unanswerable, "
              f"{np.mean(a_best < tau):.1%} of answerable")


if __name__ == "__main__":
    main(sys.argv[1:])
