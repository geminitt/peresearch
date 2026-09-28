"""How fast ZetokRAG is on this machine, per corpus, with BGE-M3 and with Qwen3-Embedding-0.6B.

    pixi run python eval/speed.py          # results/speed.md; run alone (another GPU job skews the timings)

Indexing: 1,024 documents drawn from each corpus (fixed seed) are embedded after one warm-up batch; the rate
is documents per second, with the mean document length in model tokens (cut at 512). Querying: 100 queries per
corpus go through the whole ZetokRAG path — query embedding, BM25 over the corpus, min-max fusion, reranking the
top 30 — and each stage is timed; reported is the median and the 90th percentile over queries. The embedding
matrices are the ones eval/retrieval.py saved, so only the query-time work is done here.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import DATASETS, RESULTS, RUNS, load, log  # noqa: E402

from peresearch.zetokrag import core, search  # noqa: E402

OUT = RUNS.parent / "speed"
MODELS = ["bge-m3", "qwen3-embedding-0.6b"]
N_DOCS, N_QUERIES = 1024, 100


def gpu_sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_corpus(name, embedders, reranker) -> None:
    import torch

    path = OUT / f"{name}.json"
    if path.exists():
        return
    doc_ids, texts, qs, qtexts, rel = load(name)
    rng = np.random.default_rng(0)
    sample = [texts[i] for i in rng.choice(len(texts), min(N_DOCS, len(texts)), replace=False)]
    queries = [qtexts[i] for i in rng.choice(len(qtexts), min(N_QUERIES, len(qtexts)), replace=False)]
    out = {"documents": len(texts), "index": {}, "query": {}}
    for m, e in embedders.items():
        tok = e.model.tokenizer
        lengths = [min(len(tok(t, add_special_tokens=True)["input_ids"]), 512) for t in sample]
        e.documents(sample[:32])                                    # warm-up
        gpu_sync()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        e.documents(sample)
        gpu_sync()
        dt = time.perf_counter() - t
        out["index"][m] = {"docs_per_s": len(sample) / dt, "mean_tokens": float(np.mean(lengths)),
                           "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20}
        log(name, m, f"{len(sample) / dt:.0f} docs/s")
    bm = {False: core.BM25(texts, folded=False), True: core.BM25(texts, folded=True)}
    docs = {m: np.load(RUNS / name / f"docs-{m}.npy").astype(np.float32) for m in MODELS}
    for m, e in embedders.items():
        stages = {"embed": [], "bm25": [], "dense+fusion": [], "rerank": [], "total": []}
        for q in queries:
            t0 = time.perf_counter()
            qv = e.queries([q])[0].astype(np.float32)
            gpu_sync()
            t1 = time.perf_counter()
            lexical = bm[not core.accented(q)].scores(q)
            t2 = time.perf_counter()
            cand, _ = core.fuse_minmax(lexical, docs[m] @ qv, search.RHO, search.N_FIRST)
            t3 = time.perf_counter()
            reranker.scores(q, [texts[i] for i in cand[:search.N_RERANK]])
            gpu_sync()
            t4 = time.perf_counter()
            for k, v in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)):
                stages[k].append(1000 * v)
        out["query"][m] = {k: {"median_ms": float(np.median(v)), "p90_ms": float(np.percentile(v, 90))}
                           for k, v in stages.items()}
        log(name, m, f"query median {out['query'][m]['total']['median_ms']:.0f} ms")
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))


def report(out_path: Path = RESULTS / "speed.md") -> str:
    runs = {n: json.loads((OUT / f"{n}.json").read_text()) for n in DATASETS if (OUT / f"{n}.json").exists()}
    import subprocess

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    lines = ["# ZetokRAG speed", "", f"One laptop GPU ({gpu}), fp16, batch 32, texts cut at 512 tokens. "
             f"Indexing: {N_DOCS} sampled documents per corpus; querying: {N_QUERIES} queries per corpus, "
             "median (90th percentile) in milliseconds.", "",
             "## Indexing (documents per second)", "",
             "| corpus | docs | mean tokens | BGE-M3 | Qwen3-Embedding-0.6B | ratio |", "|---|---:|---:|---:|---:|---:|"]
    for n, r in runs.items():
        b, q = r["index"]["bge-m3"], r["index"]["qwen3-embedding-0.6b"]
        lines.append(f"| {n} | {r['documents']} | {b['mean_tokens']:.0f} | {b['docs_per_s']:.0f} | "
                     f"{q['docs_per_s']:.0f} | {b['docs_per_s'] / q['docs_per_s']:.1f}x |")
    b_all = np.mean([r["index"]["bge-m3"]["docs_per_s"] for r in runs.values()])
    q_all = np.mean([r["index"]["qwen3-embedding-0.6b"]["docs_per_s"] for r in runs.values()])
    lines += [f"| **mean** | | | {b_all:.0f} | {q_all:.0f} | {b_all / q_all:.1f}x |", "",
              "## Answering one query with ZetokRAG (ms)", "",
              "| corpus | embedder | embed query | BM25 | dense + fusion | rerank top 30 | total |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for n, r in runs.items():
        for m in MODELS:
            s = r["query"][m]
            cell = lambda k: f"{s[k]['median_ms']:.0f} ({s[k]['p90_ms']:.0f})"
            lines.append(f"| {n} | {m} | {cell('embed')} | {cell('bm25')} | {cell('dense+fusion')} | "
                         f"{cell('rerank')} | {cell('total')} |")
    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    return text


def main():
    from peresearch.zetokrag.models import Embedder, Reranker

    embedders = {m: Embedder(m) for m in MODELS}
    reranker = Reranker()
    for name in DATASETS:               # the corpora this machine has embedded (a worker may hold only some)
        if all((RUNS / name / f"docs-{m}.npy").exists() for m in MODELS):
            run_corpus(name, embedders, reranker)
    print(report())


if __name__ == "__main__":
    report() if sys.argv[1:] == ["report"] else main()
