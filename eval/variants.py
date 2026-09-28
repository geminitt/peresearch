"""Which ZetokRAG configuration? Variants compared on the same benchmark corpora (eval/retrieval.py must have
run first: document embeddings are reused from runs/retrieval/).

    pixi run python eval/variants.py run        # 200 queries per corpus, fixed seed; resumable
    pixi run python eval/variants.py report     # results/variants.md

Variants (all rerank their top 30 with bge-reranker-v2-m3):
  fold   : BM25 over folded tokens + BGE-M3            (the configuration the first benchmark measured)
  plain  : plain BM25 + BGE-M3
  auto   : folded BM25 only when the query is typed without diacritics, else plain BM25; + BGE-M3
  qwen   : plain BM25 + Qwen3-Embedding-0.6B
  qwen-auto: auto BM25 + Qwen3-Embedding-0.6B
The Vietnamese corpora are also run with their queries stripped of diacritics (typed as "hoc may"), which is
where folding should help and plain BM25 should fail.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import DATASETS, RUNS, auroc, load, log, metrics, own_documents  # noqa: E402

from peresearch.zetokrag import core  # noqa: E402

OUT = RUNS.parent / "variants"
N_QUERIES, N_FIRST, N_RERANK, RHO = 200, 100, 30, 0.5
VARIANTS = {"fold": ("fold", "bge-m3"), "plain": ("plain", "bge-m3"), "auto": ("auto", "bge-m3"),
            "qwen": ("plain", "qwen3-embedding-0.6b"), "qwen-auto": ("auto", "qwen3-embedding-0.6b")}


def accented(text: str) -> bool:
    return core.fold(text) != text


def run_corpus(name: str, rr, embedders) -> None:
    path = OUT / f"{name}.json"
    if path.exists():
        return
    doc_ids, texts, qs, qtexts, rel = load(name)
    pick = sorted(np.random.default_rng(1).choice(len(qs), min(N_QUERIES, len(qs)), replace=False).tolist())
    qs, qtexts = [qs[i] for i in pick], [qtexts[i] for i in pick]
    rels = [rel[q] for q in qs]
    own = own_documents(name, doc_ids, qs)
    bm = {"plain": core.BM25(texts, folded=False), "fold": core.BM25(texts, folded=True)}
    docs = {m: np.load(RUNS / name / f"docs-{m}.npy").astype(np.float32) for m in embedders}
    out = {"queries": qs, "runs": {}}
    settings = [("as-typed", qtexts)] + ([("no-diacritics", [core.fold(q) for q in qtexts])] if DATASETS[name][3] == "vi" else [])
    t = time.time()
    for setting, queries in settings:
        qvec = {m: e.queries(queries).astype(np.float32) for m, e in embedders.items()}
        for variant, (lex, model) in VARIANTS.items():
            per_query, best = [], []
            for qi, q in enumerate(queries):
                kind = lex if lex != "auto" else ("plain" if accented(q) else "fold")
                lexical = bm[kind].scores(q)
                dense = docs[model] @ qvec[model][qi]
                if own[qi] is not None:          # ArguAna: the query's own document (see retrieval.py)
                    lexical[own[qi]] = dense[own[qi]] = -np.inf
                ids, _ = core.fuse_minmax(lexical, dense, RHO, N_FIRST)
                head = ids[:N_RERANK]
                order, s = core.rerank(ids, rr(q, head, texts))
                per_query.append(metrics([doc_ids[i] for i in order], rels[qi]))
                best.append(float(s.max()) if len(s) else 0.0)
            out["runs"][f"{setting}/{variant}"] = {"per_query": per_query, "best": best}
            log(name, setting, variant, f"nDCG@10 {np.mean([m['ndcg@10'] for m in per_query]):.4f}", f"{time.time() - t:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))


def run(names):
    from peresearch.zetokrag.models import Embedder, Reranker

    reranker = Reranker()
    cache: dict[tuple[str, int], float] = {}

    def rr(query, head, texts):
        todo = [int(i) for i in head if (query, int(i)) not in cache]
        for i, s in zip(todo, reranker.scores(query, [texts[i] for i in todo])):
            cache[(query, i)] = float(s)
        return np.array([cache[(query, int(i))] for i in head], dtype=np.float32)

    embedders = {m: Embedder(m) for m in ("bge-m3", "qwen3-embedding-0.6b")}
    for name in names:
        run_corpus(name, rr, embedders)
        cache.clear()


def report(n_boot: int = 10_000) -> str:
    runs = {n: json.loads((OUT / f"{n}.json").read_text()) for n in DATASETS if (OUT / f"{n}.json").exists()}
    rng = np.random.default_rng(0)
    lines = ["# ZetokRAG variants", "", f"{len(runs)} corpora, {N_QUERIES} queries each (fixed seed). nDCG@10; "
             "the interval is a paired bootstrap over queries within each corpus, averaged over corpora.", ""]
    for setting, langs in (("as-typed", ["en", "vi"]), ("no-diacritics", ["vi"])):
        lines += [f"## Queries {setting.replace('-', ' ')}", "", "| variant | " + " | ".join(langs) + " |",
                  "|---|" + "---|" * len(langs)]
        for v in VARIANTS:
            cells = []
            for lang in langs:
                sel = [r["runs"][f"{setting}/{v}"]["per_query"] for n, r in runs.items()
                       if DATASETS[n][3] == lang and f"{setting}/{v}" in r["runs"]]
                cells.append(f"{100 * np.mean([np.mean([m['ndcg@10'] for m in s]) for s in sel]):.1f}" if sel else "")
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
        lines += ["", f"Paired differences, {setting.replace('-', ' ')} (points):", "",
                  "| comparison | " + " | ".join(langs) + " |", "|---|" + "---|" * len(langs)]
        for a, b in [("auto", "fold"), ("auto", "plain"), ("qwen-auto", "auto"), ("qwen", "plain")]:
            cells = []
            for lang in langs:
                diffs = [np.array([x["ndcg@10"] - y["ndcg@10"] for x, y in zip(r["runs"][f"{setting}/{a}"]["per_query"],
                                                                            r["runs"][f"{setting}/{b}"]["per_query"])])
                         for n, r in runs.items() if DATASETS[n][3] == lang and f"{setting}/{a}" in r["runs"]]
                if not diffs:
                    cells.append("")
                    continue
                boot = np.mean([[d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)] for d in diffs], axis=0)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                cells.append(f"{100 * np.mean([d.mean() for d in diffs]):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
            lines.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    (Path(__file__).parent.parent / "results" / "variants.md").write_text(text)
    return text


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        run(sys.argv[2:] or list(DATASETS))
        print(report())
    elif sys.argv[1:] == ["report"]:
        print(report())
    else:
        sys.exit(__doc__)
