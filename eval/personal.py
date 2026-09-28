"""ZetokRAG on the user's own files: the reviewed question set in $PERESEARCH_HOME/eval/personal.jsonl.

    pixi run python eval/personal.py ~/projects/monodist ~/class ...     # folders to index for the run
    pixi run python eval/personal.py --embedder bge-m3 ~/notes ...        # another embedder, own index
    pixi run python eval/personal.py compare    # nine retrieval methods on the same chunks (both indexes built)

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
from peresearch.zetokrag import index as index_module
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


def main(folders: list[str], embedder: str = index_module.EMBEDDER) -> None:
    home = guard.home() / "eval"
    items = [json.loads(line) for line in open(home / "personal.jsonl")]
    from peresearch.zetokrag.models import Embedder

    where = home / ("index" if embedder == index_module.EMBEDDER else f"index-{embedder}")
    idx = Index(where, embedder=Embedder(embedder))
    print(f"embedder: {embedder}")
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
            ans[mode].append({"id": it["id"], "r1": g[0] == 1 if g else False, "r5": any(g[:5]), "r10": any(g),
                              "ndcg": dcg / ideal, "best": best})
    for mode, a in ans.items():
        print(f"answerable {len(a)} ({mode}): hit@1 {np.mean([x['r1'] for x in a]):.1%}, hit@5 {np.mean([x['r5'] for x in a]):.1%}, "
              f"hit@10 {np.mean([x['r10'] for x in a]):.1%}, nDCG@10 {np.mean([x['ndcg'] for x in a]):.3f}")
    (home / f"scores-{embedder}.json").write_text(json.dumps({"answerable": ans, "unanswerable": neg}))
    ans = ans["strict"]
    print(f"best score: answerable min {min(x['best'] for x in ans):.3f}, median {np.median([x['best'] for x in ans]):.3f}; "
          f"unanswerable max {max(neg):.3f}, median {np.median(neg):.3f}")
    a_best = np.array([a["best"] for a in ans])
    auroc = float(np.mean([(x > y) + 0.5 * (x == y) for x in a_best for y in neg]))
    print(f"not answerable {len(neg)}: AUROC of the best score {auroc:.3f}")
    for name, tau in (("PARTIAL", search.PARTIAL), ("ENOUGH", search.ENOUGH)):
        print(f"  at {name} = {tau:.3f}: rejected {np.mean(np.array(neg) < tau):.1%} of unanswerable, "
              f"{np.mean(a_best < tau):.1%} of answerable")


# --- the same questions for other retrieval methods -------------------------------------------------------

def exact_interval(k: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson 95% interval for k successes out of n, by bisection on the binomial tails (no scipy)."""
    from math import comb

    def at_most(p, kk):                       # P(X <= kk), X ~ Binomial(n, p); decreasing in p
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(kk + 1))

    def solve(f, increasing):                 # the p in [0, 1] where f(p) = 0.025
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (f(mid) < 0.025) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - at_most(p, k - 1), increasing=True)    # P(X >= k) = 0.025
    upper = 1.0 if k == n else solve(lambda p: at_most(p, k), increasing=False)           # P(X <= k) = 0.025
    return lower, upper


def compare(n_boot: int = 10_000) -> None:
    """Rank the 85 questions with nine retrieval methods over the same chunks; ZetokRAG minus each, paired."""
    from peresearch.zetokrag import core
    from peresearch.zetokrag.index import embed_text
    from peresearch.zetokrag.models import Embedder, Reranker, free_gpu

    home = guard.home() / "eval"
    items = [json.loads(line) for line in open(home / "personal.jsonl")]
    qwen_idx, bge_idx = Index(home / "index"), Index(home / "index-bge-m3")
    rows = qwen_idx.rows()
    assert [r[1:7] for r in rows] == [r[1:7] for r in bge_idx.rows()], "the two indexes hold different chunks"
    texts = [embed_text(r[2], r[6]) for r in rows]
    questions = [it["question"] for it in items]
    vecs = {"qwen": qwen_idx.dense()[1].astype(np.float32), "bge": bge_idx.dense()[1].astype(np.float32)}
    qvecs = {}
    for key, name in (("qwen", "qwen3-embedding-0.6b"), ("bge", "bge-m3")):
        e = Embedder(name)
        qvecs[key] = e.queries(questions).astype(np.float32)
        del e
        free_gpu()
    bm = {False: core.BM25(texts, folded=False), True: core.BM25(texts, folded=True)}
    rr = Reranker()

    def lexical(q, rule):
        return bm[rule == "fold" or (rule == "auto" and not core.accented(q))].scores(q)

    def rerank(q, cand):
        head = cand[:search.N_RERANK]
        s = rr.scores(q, [texts[i] for i in head])
        order = np.argsort(-s, kind="stable")
        return np.concatenate([head[order], cand[search.N_RERANK:]]), float(s.max()) if len(s) else 0.0

    methods = {
        "bm25": lambda qi, q: (core.top(lexical(q, "plain"), 100), None),
        "bm25-auto": lambda qi, q: (core.top(lexical(q, "auto"), 100), None),
        "dense-bge-m3": lambda qi, q: (core.top(vecs["bge"] @ qvecs["bge"][qi], 100), None),
        "dense-qwen3": lambda qi, q: (core.top(vecs["qwen"] @ qvecs["qwen"][qi], 100), None),
        "hybrid (no rerank)": lambda qi, q: (core.fuse_minmax(lexical(q, "auto"), vecs["qwen"] @ qvecs["qwen"][qi],
                                                               search.RHO, search.N_FIRST)[0], None),
        "dense-bge-m3+rerank": lambda qi, q: rerank(q, core.top(vecs["bge"] @ qvecs["bge"][qi], 100)),
        "dense-qwen3+rerank": lambda qi, q: rerank(q, core.top(vecs["qwen"] @ qvecs["qwen"][qi], 100)),
        "bm25-auto+rerank": lambda qi, q: rerank(q, core.top(lexical(q, "auto"), 100)),
        "zetokrag": lambda qi, q: rerank(q, core.fuse_minmax(lexical(q, "auto"), vecs["qwen"] @ qvecs["qwen"][qi],
                                                             search.RHO, search.N_FIRST)[0]),
    }
    results = {}
    for name, fn in methods.items():
        ndcg, hits, best_a, best_n = [], [], [], []
        for qi, it in enumerate(items):
            order, best = fn(qi, it["question"])
            if not it["answerable"]:
                best_n.append(best)
                continue
            g = gains([rows[i] for i in order], it["relevant"], [""] * len(it["relevant"]), "strict")
            ideal = sum(1 / math.log2(i + 2) for i in range(min(len(it["relevant"]), 10)))
            ndcg.append(sum(x / math.log2(i + 2) for i, x in enumerate(g)) / ideal)
            hits.append((g[:1] == [1], any(g[:5]), any(g)))
            best_a.append(best)
        results[name] = {"ndcg": np.array(ndcg), "hits": np.array(hits), "best_a": best_a, "best_n": best_n}
    rng = np.random.default_rng(0)
    z = results["zetokrag"]["ndcg"]
    print(f"\n{'method':22s} {'hit@1':>6s} {'hit@5':>6s} {'hit@10':>7s} {'nDCG@10':>8s}   ZetokRAG minus it [95% CI]")
    for name, r in results.items():
        d = z - r["ndcg"]
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cmp = "" if name == "zetokrag" else f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
        h = r["hits"].mean(0)
        print(f"{name:22s} {h[0]:6.1%} {h[1]:6.1%} {h[2]:7.1%} {r['ndcg'].mean():8.3f}   {cmp}")
    a, n = np.array(results["zetokrag"]["best_a"]), np.array(results["zetokrag"]["best_n"])
    for label, tau in (("PARTIAL", search.PARTIAL), ("ENOUGH", search.ENOUGH)):
        kr, kf = int((n < tau).sum()), int((a < tau).sum())
        lo_r, hi_r = exact_interval(kr, len(n))
        lo_f, hi_f = exact_interval(kf, len(a))
        print(f"{label} {tau}: unanswerable rejected {kr}/{len(n)} [{lo_r:.1%}, {hi_r:.1%}], "
              f"answerable rejected {kf}/{len(a)} [{lo_f:.1%}, {hi_f:.1%}]  (exact 95% intervals)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["compare"]:
        compare()
    elif args[:1] == ["--embedder"]:
        main(args[2:], args[1])
    else:
        main(args)
