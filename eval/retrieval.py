"""Benchmark ZetokRAG against other retrieval methods on public English and Vietnamese corpora.

    pixi run python eval/retrieval.py run            # resumable; every finished step is cached in runs/
    pixi run python eval/retrieval.py report         # results/retrieval.md

Corpora: five BEIR sets (English, from mteb), their Vietnamese translations from VN-MTEB (GreenNode), six
more Vietnamese sets from VN-MTEB (named "nano-*" there, though each corpus holds about 100k documents), and
Zalo legal retrieval (native Vietnamese). Query sets above 1,000 are subsampled with a fixed seed. Methods share one first
stage per view, and every reranked method reranks the top 30 of its own first stage with the same model.

Besides ranking quality, "does the corpus hold an answer at all?" is measured the way RGB measures negative
rejection: every query is also run with its relevant documents removed from the corpus, and the reranker's
best score must separate the two cases (AUROC; the threshold is fixed on the other corpora).
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from peresearch.zetokrag import core

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "retrieval"
N_FIRST, N_RERANK, RHO = 100, 30, 0.5
MAX_QUERIES = 1000   # larger query sets are subsampled (fixed seed); corpora are always used whole

DATASETS = {  # name: (repo, layout, split, language)
    **{n: (f"mteb/{n}", "mteb", "test", "en") for n in ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs"]},
    **{f"{n}-vn": (f"GreenNode/{n}-vn", "greennode", "test", "vi")
       for n in ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs"]},
    **{f"nano-{n}-vn": (f"GreenNode/nano-{n}-vn", "greennode", "test", "vi")
       for n in ["nq", "hotpotqa", "fever", "dbpedia", "climate-fever"]},
    "nano-msmarco-vn": ("GreenNode/nano-msmarco-vn", "greennode", "dev", "vi"),
    "zalo-legal-vn": ("GreenNode/zalo-ai-legal-text-retrieval-vn", "greennode", "test", "vi"),
}
# In ArguAna each query is itself an argument from the corpus. As in BEIR's evaluation, the query's own document
# is excluded from every ranking; otherwise every method finds the query itself first and is scored wrong.
SELF_IN_CORPUS = {"arguana", "arguana-vn"}
EMBEDDERS = ["bge-m3", "multilingual-e5-large", "qwen3-embedding-0.6b"]
METHODS = {  # name: (first stage, reranked?)
    "bm25": ("bm25", False),
    "bm25-fold": ("bm25-fold", False),
    "dense-bge-m3": ("dense-bge-m3", False),
    "dense-multilingual-e5-large": ("dense-multilingual-e5-large", False),
    "dense-qwen3-embedding-0.6b": ("dense-qwen3-embedding-0.6b", False),
    "hybrid-rrf": ("hybrid-rrf", False),
    "hybrid-minmax": ("hybrid-minmax", False),
    "bm25-fold+rerank": ("bm25-fold", True),
    "dense-bge-m3+rerank": ("dense-bge-m3", True),
    "zetokrag": ("hybrid-minmax", True),        # BM25-fold + BGE-M3, min-max fusion, bge-reranker-v2-m3
}


# --- data -------------------------------------------------------------------------------------------------

def load(name: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo, layout, split, _ = DATASETS[name]
    get = lambda f: hf_hub_download(repo, f, repo_type="dataset")
    if layout == "mteb":
        corpus = pd.read_json(get("corpus.jsonl"), lines=True, dtype=False)
        queries = pd.read_json(get("queries.jsonl"), lines=True, dtype=False)
        qrels = pd.read_json(get(f"qrels/{split}.jsonl"), lines=True, dtype=False)
    else:
        corpus = pd.read_parquet(get(f"corpus/{split}-00000-of-00001.parquet"))
        queries = pd.read_parquet(get(f"queries/{split}-00000-of-00001.parquet"))
        qrels = pd.read_parquet(get(f"qrels/{split}-00000-of-00001.parquet"))
    cid = "_id" if "_id" in corpus else "id"
    qid = "_id" if "_id" in queries else "id"
    title = corpus["title"].fillna("") if "title" in corpus else ""
    texts = (title + "\n" + corpus["text"].fillna("")).str.strip() if "title" in corpus else corpus["text"].fillna("")
    doc_ids = corpus[cid].astype(str).tolist()
    rel: dict[str, dict[str, int]] = {}
    for q, d, s in zip(qrels["query-id"].astype(str), qrels["corpus-id"].astype(str), qrels["score"]):
        if int(s) > 0:
            rel.setdefault(q, {})[d] = int(s)
    known = set(doc_ids)
    rel = {q: {d: s for d, s in r.items() if d in known} for q, r in rel.items()}
    rel = {q: r for q, r in rel.items() if r}
    qtext = dict(zip(queries[qid].astype(str), queries["text"]))
    qs = sorted(q for q in rel if q in qtext)
    if len(qs) > MAX_QUERIES:
        qs = sorted(np.random.default_rng(0).choice(qs, MAX_QUERIES, replace=False).tolist())
    return doc_ids, texts.tolist(), qs, [qtext[q] for q in qs], {q: rel[q] for q in qs}


# --- metrics ----------------------------------------------------------------------------------------------

def metrics(ranked: list[str], rel: dict[str, int]) -> dict[str, float]:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sum(s / math.log2(i + 2) for i, s in enumerate(sorted(rel.values(), reverse=True)[:10]))
    mrr = next((1 / (i + 1) for i, d in enumerate(ranked[:10]) if d in rel), 0.0)
    return {"ndcg@10": dcg / ideal, "recall@10": len(set(ranked[:10]) & rel.keys()) / len(rel),
            "recall@100": len(set(ranked[:100]) & rel.keys()) / len(rel), "mrr@10": mrr}


# --- run --------------------------------------------------------------------------------------------------

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def own_documents(name, doc_ids, qs) -> list:
    """Per query, the index of the query's own document when the corpus contains it (else None)."""
    if name not in SELF_IN_CORPUS:
        return [None] * len(qs)
    pos = {d: i for i, d in enumerate(doc_ids)}
    return [pos.get(q) for q in qs]


def exclude(per_query, own) -> list:
    """Scores with each query's own document pushed to the bottom."""
    for scores, i in zip(per_query, own):
        if i is not None:
            scores[i] = -np.inf
    return per_query


def first_stages(name, doc_ids, texts, qtexts, out: Path, own) -> dict:
    """Top N_FIRST (index, score) per query for every first-stage view; cached."""
    from peresearch.zetokrag.models import Embedder, free_gpu

    path = out / "first_stage.npz"
    if path.exists():
        z = np.load(path)
        return {k: (z[k + ":ids"], z[k + ":scores"]) for k in {k.split(":")[0] for k in z.files}}
    # full score arrays are kept only for the two views the fusions need; the rest shrink to their top N
    full, result = {}, {}

    def keep(view, per_query):
        ids = np.stack([core.top(sc, N_FIRST) for sc in per_query])
        result[view] = (ids, np.stack([sc[i] for sc, i in zip(per_query, ids)]))

    for folded, view in ((False, "bm25"), (True, "bm25-fold")):
        bm = core.BM25(texts, folded=folded)
        per_query = exclude([bm.scores(q) for q in qtexts], own)
        keep(view, per_query)
        if folded:
            full["bm25-fold"] = per_query
        log(name, view, "done")
    for model in EMBEDDERS:
        emb_path = out / f"docs-{model}.npy"
        emb = Embedder(model)
        if emb_path.exists():
            docs = np.load(emb_path)
        else:
            t, parts = time.time(), []
            for a in range(0, len(texts), 4096):    # in slices, so a stall shows in the log
                parts.append(emb.documents(texts[a:a + 4096]))
                log(name, model, f"{a + len(parts[-1])}/{len(texts)} docs, {time.time() - t:.0f}s")
            docs = np.concatenate(parts)
            np.save(emb_path, docs)
            log(name, model, f"{len(texts)} docs encoded in {time.time() - t:.0f}s")
        queries = emb.queries(qtexts)
        del emb
        free_gpu()
        per_query = exclude(list((queries.astype(np.float32) @ docs.astype(np.float32).T)), own)
        keep(f"dense-{model}", per_query)
        if model == "bge-m3":
            full["bge-m3"] = per_query
    for view, fuse in (("hybrid-minmax", lambda a, b: core.fuse_minmax(a, b, RHO, N_FIRST)),
                       ("hybrid-rrf", lambda a, b: core.fuse_rrf(a, b, 60, N_FIRST))):
        pairs = [fuse(a, b) for a, b in zip(full["bm25-fold"], full["bge-m3"])]
        width = max(len(i) for i, _ in pairs)
        ids = np.full((len(pairs), width), -1)
        scores = np.full((len(pairs), width), -np.inf, dtype=np.float32)
        for r, (i, s) in enumerate(pairs):
            ids[r, :len(i)], scores[r, :len(s)] = i, s
        result[view] = (ids, scores)
    np.savez(path, **{f"{k}:ids": v[0] for k, v in result.items()}, **{f"{k}:scores": v[1] for k, v in result.items()})
    return result


def rerank_all(name, texts, qtexts, rels, doc_ids, stages, out: Path) -> dict:
    """Reranked orders for every reranked method, plus the answerable / unanswerable best scores."""
    path = out / "rerank.json"
    if path.exists():
        return json.loads(path.read_text())
    from peresearch.zetokrag.models import Reranker, free_gpu

    rr = Reranker()
    cache: list[dict[int, float]] = [{} for _ in qtexts]

    def score(qi, idx):
        todo = [i for i in idx if i not in cache[qi]]
        for i, s in zip(todo, rr.scores(qtexts[qi], [texts[i] for i in todo])):
            cache[qi][i] = float(s)
        return np.array([cache[qi][i] for i in idx], dtype=np.float32)

    result = {"orders": {}, "answerable": [], "unanswerable": []}
    t = time.time()
    for view in sorted({v for v, r in METHODS.values() if r}):
        ids = stages[view][0]
        orders = []
        for qi in range(len(qtexts)):
            if qi and qi % 200 == 0:   # keeps the log alive, so a real stall is told apart from slow work
                log(name, "reranking", view, f"{qi}/{len(qtexts)}", f"{time.time() - t:.0f}s")
            row = [int(i) for i in ids[qi] if i >= 0]
            head = row[:N_RERANK]
            reordered, _ = core.rerank(np.array(row), score(qi, head))
            orders.append(reordered.tolist())
        result["orders"][view] = orders
        log(name, "reranked", view, f"{time.time() - t:.0f}s")
    # negative rejection: the same query with its relevant documents gone from the corpus
    pos = {d: i for i, d in enumerate(doc_ids)}
    ids = stages["hybrid-minmax"][0]
    for qi in range(len(qtexts)):
        gone = {pos[d] for d in rels[qi]}
        row = [int(i) for i in ids[qi] if i >= 0]
        result["answerable"].append(float(score(qi, row[:N_RERANK]).max()))
        kept = [i for i in row if i not in gone][:N_RERANK]
        result["unanswerable"].append(float(score(qi, kept).max()) if kept else 0.0)
    log(name, "negative rejection pass", f"{time.time() - t:.0f}s")
    del rr
    free_gpu()
    path.write_text(json.dumps(result))
    return result


def run_dataset(name: str) -> None:
    out = RUNS / name
    done = out / "metrics.json"
    if done.exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    doc_ids, texts, qs, qtexts, rel = load(name)
    log(name, f"{len(texts)} documents, {len(qs)} queries")
    own = own_documents(name, doc_ids, qs)
    stages = first_stages(name, doc_ids, texts, qtexts, out, own)
    rels = [rel[q] for q in qs]
    rr = rerank_all(name, texts, qtexts, rels, doc_ids, stages, out)
    per_method = {}
    for method, (view, reranked) in METHODS.items():
        orders = rr["orders"][view] if reranked else [[int(i) for i in row if i >= 0] for row in stages[view][0]]
        # the own document never reaches a ranking; the filter only guards the metric
        per_method[method] = [metrics([doc_ids[i] for i in o if i != x], r) for o, r, x in zip(orders, rels, own)]
    done.write_text(json.dumps({"queries": qs, "documents": len(texts), "per_query": per_method,
                                "answerable": rr["answerable"], "unanswerable": rr["unanswerable"]}))
    log(name, "zetokrag nDCG@10", round(float(np.mean([m["ndcg@10"] for m in per_method["zetokrag"]])), 4))


# --- report -----------------------------------------------------------------------------------------------

def auroc(pos: list[float], neg: list[float]) -> float:
    """P(answerable score > unanswerable score), ties counting half."""
    p, n = np.asarray(pos), np.asarray(neg)
    return float(((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def report(out_path: Path = ROOT / "results" / "retrieval.md", n_boot: int = 10_000) -> str:
    runs = {n: json.loads((RUNS / n / "metrics.json").read_text())
            for n in DATASETS if (RUNS / n / "metrics.json").exists()}
    rng = np.random.default_rng(0)
    lines = ["# Retrieval benchmark", "",
             f"{len(runs)} corpora. First stage keeps the top {N_FIRST}; reranked methods rerank their top "
             f"{N_RERANK} with bge-reranker-v2-m3; min-max fusion uses rho = {RHO}. Texts are cut at 512 tokens.", ""]
    for metric in ["ndcg@10", "recall@10", "recall@100", "mrr@10"]:
        lines += [f"## {metric}", "", "| corpus | lang | docs | queries | " + " | ".join(METHODS) + " |",
                  "|---|---|---:|---:|" + "---:|" * len(METHODS)]
        for n, r in runs.items():
            vals = [100 * np.mean([m[metric] for m in r["per_query"][k]]) for k in METHODS]
            best = max(vals)
            lines.append(f"| {n} | {DATASETS[n][3]} | {r['documents']} | {len(r['queries'])} | " +
                         " | ".join(f"**{v:.1f}**" if v == best else f"{v:.1f}" for v in vals) + " |")
        for lang in ["en", "vi", "all"]:
            sel = [r for n, r in runs.items() if lang == "all" or DATASETS[n][3] == lang]
            if sel:
                vals = [100 * np.mean([np.mean([m[metric] for m in r["per_query"][k]]) for r in sel]) for k in METHODS]
                lines.append(f"| **mean ({lang})** | | | | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
        lines.append("")
    # paired bootstrap: ZetokRAG minus each method, nDCG@10, macro mean over corpora
    lines += ["## ZetokRAG minus each method (nDCG@10, points, macro mean over corpora)", "",
              "Queries are resampled within each corpus (paired: the same queries for both methods), then corpus "
              f"means are averaged; {n_boot} resamples, percentile 95% interval.", "",
              "| method | en | vi | all |", "|---|---|---|---|"]
    for k in METHODS:
        if k == "zetokrag":
            continue
        cells = []
        for lang in ["en", "vi", "all"]:
            sel = [r for n, r in runs.items() if lang == "all" or DATASETS[n][3] == lang]
            if not sel:
                cells.append("")
                continue
            diffs = [np.array([a["ndcg@10"] - b["ndcg@10"] for a, b in zip(r["per_query"]["zetokrag"], r["per_query"][k])])
                     for r in sel]
            point = np.mean([d.mean() for d in diffs])
            boot = np.mean([[d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)] for d in diffs], axis=0)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            cells.append(f"{100 * point:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    # negative rejection
    lines += ["", "## Does the corpus hold an answer? (ZetokRAG's best reranker score)", "",
              "Each query is run twice: as is, and with its relevant documents removed. AUROC = chance that the "
              "answerable run scores higher. The threshold for a corpus is set on all *other* corpora so that 90% "
              "of their answerable queries pass; rejection = unanswerable runs below it, false rejection = "
              "answerable runs below it.", "",
              "| corpus | AUROC | threshold | rejection | false rejection |", "|---|---:|---:|---:|---:|"]
    for n, r in runs.items():
        others = [s for m, o in runs.items() if m != n for s in o["answerable"]]
        tau = float(np.quantile(others, 0.10)) if others else 0.5
        a, u = np.array(r["answerable"]), np.array(r["unanswerable"])
        lines.append(f"| {n} | {auroc(r['answerable'], r['unanswerable']):.3f} | {tau:.3f} | "
                     f"{100 * (u < tau).mean():.1f}% | {100 * (a < tau).mean():.1f}% |")
    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(text)
    return text


if __name__ == "__main__":
    if sys.argv[1:] == ["run"] or sys.argv[1:2] == ["run"]:
        for name in (sys.argv[2:] or list(DATASETS)):
            run_dataset(name)
        print(report())
    elif sys.argv[1:] == ["report"]:
        print(report())
    else:
        sys.exit(__doc__)
