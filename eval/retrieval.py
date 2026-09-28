"""Benchmark ZetokRAG against other retrieval methods on public English and Vietnamese corpora.

    pixi run python eval/retrieval.py run [corpus ...]      # resumable; every finished step is cached
    pixi run python eval/retrieval.py report                # results/retrieval.md
    python eval/retrieval.py run --shard 0/4                # one of four workers (see kaggle/)

Corpora: five BEIR sets (English, from mteb), their Vietnamese translations from VN-MTEB (GreenNode), six
more Vietnamese sets from VN-MTEB (named "nano-*" there, though each corpus holds about 100k documents), and
Zalo legal retrieval (native Vietnamese), each pinned to a dataset revision. Corpora are used whole; query sets
above 1,000 are subsampled with a fixed seed (with all queries the confidence intervals of the conclusions
narrowed by only 0.01-0.07 points, as the smallest corpora dominate them). Every Vietnamese corpus is run
twice: with its queries as written, and with the same queries stripped of diacritics ("hoc may"), the way they
are often typed. Methods share one first stage per view, and every reranked method reranks the top 30 of its
own first stage with the same model.

Besides ranking quality, "does the corpus hold an answer at all?" is measured the way RGB measures negative
rejection: every query is also run with its relevant documents removed from the corpus, and the reranker's
best score must separate the two cases (AUROC; the threshold is fixed on the other corpora).
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from peresearch.zetokrag import core

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("PERESEARCH_RUNS", ROOT / "runs")) / "retrieval"
RESULTS = Path(os.environ.get("PERESEARCH_RESULTS", ROOT / "results"))
N_FIRST, N_RERANK, RHO = 100, 30, 0.5
QUERY_BATCH = 256
MAX_QUERIES = 1000   # per corpus, subsampled with seed 0

DATASETS = {  # name: (repo, layout, split, language)
    **{n: (f"mteb/{n}", "mteb", "test", "en") for n in ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs"]},
    **{f"{n}-vn": (f"GreenNode/{n}-vn", "greennode", "test", "vi")
       for n in ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs"]},
    **{f"nano-{n}-vn": (f"GreenNode/nano-{n}-vn", "greennode", "test", "vi")
       for n in ["nq", "hotpotqa", "fever", "dbpedia", "climate-fever"]},
    "nano-msmarco-vn": ("GreenNode/nano-msmarco-vn", "greennode", "dev", "vi"),
    "zalo-legal-vn": ("GreenNode/zalo-ai-legal-text-retrieval-vn", "greennode", "test", "vi"),
}
REVISIONS = {  # Hugging Face dataset commits, so a changed dataset cannot change the benchmark silently
    "scifact": "cf10ab6856b15b0e670ef8ae5dae4e266c12d035",
    "nfcorpus": "52ac3f19d3449632d9f00aab0ad34a110fc03816",
    "fiqa": "5e59eeb3a7df6b85882112b747008547c21587ea",
    "arguana": "6c1bcf74b13dfd823aff056b79d4d93e702f19c7",
    "scidocs": "490848228d0a9ca7a7244f5e77d8fe33e6df6974",
    "scifact-vn": "3fe01c71905b40964f70fa93f255505fd5009ea5",
    "nfcorpus-vn": "3a0a6496015fd44ac99d39f994e6a6f15b068ec1",
    "fiqa-vn": "d8a8236d8fb09789467121124f7d379fb330e24c",
    "arguana-vn": "137122e4a56c03399d31bce35a045f0034242a4c",
    "scidocs-vn": "2314d706e594b92b12999bb15159a45bc4cb8949",
    "nano-nq-vn": "1ad4d6556fe0e5314994839089ce070fb0db8b19",
    "nano-hotpotqa-vn": "f4de19a2fae1a582de114e5bcd178bb262183113",
    "nano-fever-vn": "457ca6b058ed19b28f2359e2d816d7527af6bef8",
    "nano-dbpedia-vn": "bbc3259bc63bf1e250d7034024092cc3230d5850",
    "nano-climate-fever-vn": "1852e852f07403d4529a8520d52b91ff6d57869b",
    "nano-msmarco-vn": "f149369c82ec228b05b0f6677699ab4bfbab73f6",
    "zalo-legal-vn": "12d76d4d04b94ceada970fcfbe7fec20bfa97389",
}
# In ArguAna each query is itself an argument from the corpus. As in BEIR's evaluation, the query's own document
# is excluded from every ranking; otherwise every method finds the query itself first and is scored wrong.
SELF_IN_CORPUS = {"arguana", "arguana-vn"}
EMBEDDERS = ["bge-m3", "multilingual-e5-large", "qwen3-embedding-0.6b"]
# BM25 rules: "plain" matches words as typed, "fold" matches diacritic-free forms, "auto" folds only queries
# typed without diacritics (ZetokRAG's rule).
METHODS = {  # name: (first stage, reranked?)
    "bm25": ("bm25", False),
    "bm25-fold": ("bm25-fold", False),
    "bm25-auto": ("bm25-auto", False),
    "dense-bge-m3": ("dense-bge-m3", False),
    "dense-multilingual-e5-large": ("dense-multilingual-e5-large", False),
    "dense-qwen3-embedding-0.6b": ("dense-qwen3-embedding-0.6b", False),
    "hybrid-rrf": ("hybrid-rrf", False),                          # RRF of BM25-auto and Qwen3-Embedding
    "hybrid-minmax": ("hybrid-auto-qwen", False),                 # ZetokRAG without the reranker
    "bm25-auto+rerank": ("bm25-auto", True),
    "dense-bge-m3+rerank": ("dense-bge-m3", True),
    "dense-qwen3+rerank": ("dense-qwen3-embedding-0.6b", True),
    "zetokrag-v0": ("hybrid-fold-bge", True),                     # the first configuration: BM25-fold + BGE-M3
    "zetokrag-bge": ("hybrid-auto-bge", True),                    # ZetokRAG with BGE-M3
    "zetokrag": ("hybrid-auto-qwen", True),                       # ZetokRAG as used: BM25-auto + Qwen3-Embedding
}
SETTINGS = {"as-typed": lambda q: q, "no-diacritics": core.fold}
FUSIONS = {  # view: (kind, BM25 view, embedder)
    "hybrid-fold-bge": ("minmax", "bm25-fold", "bge-m3"),
    "hybrid-auto-bge": ("minmax", "bm25-auto", "bge-m3"),
    "hybrid-auto-qwen": ("minmax", "bm25-auto", "qwen3-embedding-0.6b"),
    "hybrid-rrf": ("rrf", "bm25-auto", "qwen3-embedding-0.6b"),
}


# --- data -------------------------------------------------------------------------------------------------

def load(name: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo, layout, split, _ = DATASETS[name]
    get = lambda f: hf_hub_download(repo, f, repo_type="dataset", revision=REVISIONS[name])
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


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(answerable score > unanswerable score), ties counting half."""
    p, n = np.sort(np.asarray(pos, dtype=float)), np.asarray(neg, dtype=float)
    right, left = np.searchsorted(p, n, side="right"), np.searchsorted(p, n, side="left")
    return float(((len(p) - right).sum() + 0.5 * (right - left).sum()) / (len(p) * len(n)))


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


def document_embeddings(name, model, texts, base: Path) -> np.ndarray:
    """Embeddings of every document, saved slice by slice so an interrupted run resumes."""
    from peresearch.zetokrag.models import Embedder, free_gpu, gpu_memory

    path = base / f"docs-{model}.npy"
    if path.exists():
        return np.load(path)
    step, parts, emb, t = 16384, [], None, time.time()
    for a in range(0, len(texts), step):
        part = base / f"docs-{model}.{a:08d}.npy"
        if not part.exists():
            emb = emb or Embedder(model)
            np.save(part, emb.documents(texts[a:a + step]))
            log(name, model, f"{min(a + step, len(texts))}/{len(texts)} docs, {time.time() - t:.0f}s;", gpu_memory())
        parts.append(part)
    docs = np.concatenate([np.load(x) for x in parts])
    np.save(path, docs)
    for x in parts:
        x.unlink()
    del emb
    free_gpu()
    return docs


def first_stages(name, texts, qtexts, out: Path, own, base: Path) -> dict:
    """Top N_FIRST (ids, scores) per query for every first-stage view; cached. Queries are processed in batches
    and full score vectors are never kept, so memory stays flat for corpora with 6,000 queries."""
    import torch

    from peresearch.zetokrag.models import Embedder, free_gpu, gpu_memory

    path = out / "first_stage.npz"
    if path.exists():
        z = np.load(path)
        return {k: (z[k + ":ids"], z[k + ":scores"]) for k in {k.split(":")[0] for k in z.files}}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs = {m: document_embeddings(name, m, texts, base) for m in EMBEDDERS}
    qvecs = {}
    for m in EMBEDDERS:
        e = Embedder(m)
        qvecs[m] = e.queries(qtexts)
        log(name, m, f"{len(qtexts)} queries;", gpu_memory())
        del e
        free_gpu()
    docs = {m: torch.from_numpy(v).to(device) for m, v in docs.items()}
    bm = {"bm25": core.BM25(texts, folded=False), "bm25-fold": core.BM25(texts, folded=True)}
    views = ["bm25", "bm25-fold", "bm25-auto", *[f"dense-{m}" for m in EMBEDDERS], *FUSIONS]
    kept = {v: ([], []) for v in views}

    def keep(view, i, sc):
        kept[view][0].append(i)
        kept[view][1].append(np.asarray(sc, dtype=np.float32))

    t = time.time()
    for a in range(0, len(qtexts), QUERY_BATCH):
        batch = range(a, min(a + QUERY_BATCH, len(qtexts)))
        dense = {m: (docs[m] @ torch.from_numpy(qvecs[m][a:batch.stop]).to(device).T).float().T.cpu().numpy()
                 for m in EMBEDDERS}
        for j, qi in enumerate(batch):
            q, x = qtexts[qi], own[qi]
            full = {"bm25": bm["bm25"].scores(q), "bm25-fold": bm["bm25-fold"].scores(q)}
            full["bm25-auto"] = full["bm25"] if core.accented(q) else full["bm25-fold"]
            for m in EMBEDDERS:
                full[m] = dense[m][j]
            if x is not None:
                for sc in full.values():
                    sc[x] = -np.inf
            for v in ("bm25", "bm25-fold", "bm25-auto"):
                i = core.top(full[v], N_FIRST)
                keep(v, i, full[v][i])
            for m in EMBEDDERS:
                i = core.top(full[m], N_FIRST)
                keep(f"dense-{m}", i, full[m][i])
            for view, (kind, lex, emb) in FUSIONS.items():
                i, sc = core.fuse_minmax(full[lex], full[emb], RHO, N_FIRST) if kind == "minmax" \
                    else core.fuse_rrf(full[lex], full[emb], 60, N_FIRST)
                keep(view, i, sc)
        log(name, "first stage", f"{batch.stop}/{len(qtexts)} queries, {time.time() - t:.0f}s")
    del docs
    free_gpu()
    result = {}
    for v, (ids, scores) in kept.items():
        width = max(len(i) for i in ids)
        I = np.full((len(ids), width), -1)
        S = np.full((len(ids), width), -np.inf, dtype=np.float32)
        for r, (i, sc) in enumerate(zip(ids, scores)):
            I[r, :len(i)], S[r, :len(sc)] = i, sc
        result[v] = (I, S)
    np.savez(path, **{f"{k}:ids": v[0] for k, v in result.items()}, **{f"{k}:scores": v[1] for k, v in result.items()})
    return result


def rerank_all(name, texts, qtexts, rels, doc_ids, stages, out: Path) -> dict:
    """Reranked orders for every reranked view, plus ZetokRAG's answerable / unanswerable best scores.

    Reranker scores are saved after every view (rerank_scores.json), so a restart resumes where it stopped."""
    path = out / "rerank.json"
    if path.exists():
        return json.loads(path.read_text())
    from peresearch.zetokrag.models import Reranker, free_gpu, gpu_memory

    rr = Reranker()
    cache_path = out / "rerank_scores.json"
    cache: list[dict[int, float]] = [{int(k): v for k, v in d.items()} for d in json.loads(cache_path.read_text())] \
        if cache_path.exists() else [{} for _ in qtexts]

    def score(qi, idx):
        todo = [i for i in idx if i not in cache[qi]]
        for i, sc in zip(todo, rr.scores(qtexts[qi], [texts[i] for i in todo])):
            cache[qi][i] = float(sc)
        return np.array([cache[qi][i] for i in idx], dtype=np.float32)

    result = {"orders": {}, "answerable": [], "unanswerable": []}
    t = time.time()
    for view in sorted({v for v, r in METHODS.values() if r}):
        ids = stages[view][0]
        orders = []
        for qi in range(len(qtexts)):
            if qi and qi % 500 == 0:   # keeps the log alive, so a real stall is told apart from slow work
                log(name, "reranking", view, f"{qi}/{len(qtexts)}", f"{time.time() - t:.0f}s")
            row = [int(i) for i in ids[qi] if i >= 0]
            reordered, _ = core.rerank(np.array(row), score(qi, row[:N_RERANK]))
            orders.append(reordered.tolist())
        result["orders"][view] = orders
        cache_path.write_text(json.dumps([{str(k): v for k, v in d.items()} for d in cache]))
        log(name, "reranked", view, f"{time.time() - t:.0f}s;", gpu_memory())
    # negative rejection: the same query with its relevant documents gone from the corpus
    pos = {d: i for i, d in enumerate(doc_ids)}
    ids = stages["hybrid-auto-qwen"][0]
    for qi in range(len(qtexts)):
        gone = {pos[d] for d in rels[qi]}
        row = [int(i) for i in ids[qi] if i >= 0]
        result["answerable"].append(float(score(qi, row[:N_RERANK]).max()))
        kept = [i for i in row if i not in gone][:N_RERANK]
        result["unanswerable"].append(float(score(qi, kept).max()) if kept else 0.0)
    cache_path.write_text(json.dumps([{str(k): v for k, v in d.items()} for d in cache]))
    log(name, "negative rejection pass", f"{time.time() - t:.0f}s")
    del rr
    free_gpu()
    path.write_text(json.dumps(result))
    return result


def manifest(name: str, setting: str, n_docs: int, n_queries: int) -> dict:
    """What produced a result: code, data and model revisions, packages, hardware."""
    import platform
    from importlib.metadata import version

    import torch

    from peresearch.zetokrag.models import EMBEDDERS as SPECS, RERANKER

    return {"corpus": name, "setting": setting, "dataset": DATASETS[name][0], "dataset_revision": REVISIONS[name],
            "documents": n_docs, "queries": n_queries, "max_queries": MAX_QUERIES,
            "commit": os.environ.get("PERESEARCH_COMMIT", "unknown"),
            "models": {**{m: SPECS[m]["revision"] for m in EMBEDDERS}, "bge-reranker-v2-m3": RERANKER["revision"]},
            "packages": {p: version(p) for p in ("torch", "transformers", "sentence-transformers", "bm25s", "numpy")},
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "n_first": N_FIRST, "n_rerank": N_RERANK, "rho": RHO, "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def run_dataset(name: str, setting: str = "as-typed") -> None:
    base = RUNS / name
    out = base / setting
    done = out / "metrics.json"
    if done.exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    doc_ids, texts, qs, qtexts, rel = load(name)
    qtexts = [SETTINGS[setting](q) for q in qtexts]
    log(name, setting, f"{len(texts)} documents, {len(qs)} queries")
    (out / "manifest.json").write_text(json.dumps(manifest(name, setting, len(texts), len(qs)), indent=1))
    own = own_documents(name, doc_ids, qs)
    stages = first_stages(name, texts, qtexts, out, own, base)
    rels = [rel[q] for q in qs]
    rr = rerank_all(name, texts, qtexts, rels, doc_ids, stages, out)
    per_method = {}
    for method, (view, reranked) in METHODS.items():
        orders = rr["orders"][view] if reranked else [[int(i) for i in row if i >= 0] for row in stages[view][0]]
        # the own document never reaches a ranking; the filter only guards the metric
        per_method[method] = [metrics([doc_ids[i] for i in o if i != x], r) for o, r, x in zip(orders, rels, own)]
    done.write_text(json.dumps({"queries": qs, "documents": len(texts), "per_query": per_method,
                                "answerable": rr["answerable"], "unanswerable": rr["unanswerable"]}))
    log(name, setting, "zetokrag nDCG@10", round(float(np.mean([m["ndcg@10"] for m in per_method["zetokrag"]])), 4))


def jobs(names=None) -> list[tuple[str, str]]:
    names = names or list(DATASETS)
    return [(n, st) for n in names for st in SETTINGS if st == "as-typed" or DATASETS[n][3] == "vi"]


# Rough relative cost of a corpus (document embeddings, then reranking every query), to split the corpora
# across workers evenly; a Vietnamese corpus counts twice (both query settings).
# Estimated T4 minutes: embedding the documents with the three models (once per corpus) plus reranking about
# one second per query (per query setting). From the Kaggle smoke run on SciFact.
COST = {"scifact": 20, "nfcorpus": 12, "fiqa": 80, "arguana": 30, "scidocs": 55, "scifact-vn": 16, "nfcorpus-vn": 10,
        "fiqa-vn": 75, "arguana-vn": 30, "scidocs-vn": 45, "nano-nq-vn": 120, "nano-hotpotqa-vn": 120,
        "nano-fever-vn": 125, "nano-dbpedia-vn": 135, "nano-climate-fever-vn": 120, "nano-msmarco-vn": 115,
        "zalo-legal-vn": 110}


def shard(k: int, n: int) -> list[str]:
    """Corpora of worker k of n: largest first onto the least loaded worker. Both settings of a corpus stay
    with one worker, as they share its document embeddings."""
    load_, owner = [0.0] * n, {}
    for name in sorted(DATASETS, key=lambda c: -COST[c]):
        w = int(np.argmin(load_))
        owner[name] = w
        load_[w] += COST[name]
    return [c for c in DATASETS if owner[c] == k]


# --- report -----------------------------------------------------------------------------------------------

def report(out_path: Path = RESULTS / "retrieval.md", n_boot: int = 10_000) -> str:
    rng = np.random.default_rng(0)
    lines = ["# Retrieval benchmark", "",
             f"Every query of every corpus. First stage keeps the top {N_FIRST}; reranked methods rerank their top {N_RERANK} with "
             f"bge-reranker-v2-m3; min-max fusion uses rho = {RHO}. Texts are cut at 512 tokens. In ArguAna the "
             "query's own document is excluded from every ranking, as in BEIR. `zetokrag` is the configuration in "
             "use (BM25-auto + Qwen3-Embedding-0.6B); `zetokrag-bge` swaps in BGE-M3; `zetokrag-v0` is the first "
             "configuration (BM25 on diacritic-free words for every query + BGE-M3).", ""]
    for setting in SETTINGS:
        runs = {n: json.loads((RUNS / n / setting / "metrics.json").read_text())
                for n in DATASETS if (RUNS / n / setting / "metrics.json").exists()}
        if not runs:
            continue
        langs = ["en", "vi", "all"] if setting == "as-typed" else ["vi"]
        lines += [f"# Queries {setting.replace('-', ' ')} ({len(runs)} corpora)", ""]
        for metric in ["ndcg@10", "recall@10", "recall@100", "mrr@10"]:
            lines += [f"## {metric}", "", "| corpus | lang | docs | queries | " + " | ".join(METHODS) + " |",
                      "|---|---|---:|---:|" + "---:|" * len(METHODS)]
            for n, r in runs.items():
                vals = [100 * np.mean([m[metric] for m in r["per_query"][k]]) for k in METHODS]
                best = max(vals)
                lines.append(f"| {n} | {DATASETS[n][3]} | {r['documents']} | {len(r['queries'])} | " +
                             " | ".join(f"**{v:.1f}**" if v == best else f"{v:.1f}" for v in vals) + " |")
            for lang in langs:
                sel = [r for n, r in runs.items() if lang == "all" or DATASETS[n][3] == lang]
                if sel:
                    vals = [100 * np.mean([np.mean([m[metric] for m in r["per_query"][k]]) for r in sel]) for k in METHODS]
                    lines.append(f"| **mean ({lang})** | | | | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
            lines.append("")
        lines += ["## ZetokRAG minus each method (nDCG@10, points, macro mean over corpora)", "",
                  "Queries are resampled within each corpus (paired: the same queries for both methods), then corpus "
                  f"means are averaged; {n_boot} resamples, percentile 95% interval.", "",
                  "| method | " + " | ".join(langs) + " |", "|---|" + "---|" * len(langs)]
        for k in METHODS:
            if k == "zetokrag":
                continue
            cells = []
            for lang in langs:
                sel = [r for n, r in runs.items() if lang == "all" or DATASETS[n][3] == lang]
                diffs = [np.array([a["ndcg@10"] - b["ndcg@10"] for a, b in zip(r["per_query"]["zetokrag"], r["per_query"][k])])
                         for r in sel]
                if not diffs:
                    cells.append("")
                    continue
                point = np.mean([d.mean() for d in diffs])
                boot = np.mean([[d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)] for d in diffs], axis=0)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                cells.append(f"{100 * point:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines += ["", "## Does the corpus hold an answer? (ZetokRAG's best reranker score)", "",
                  "Each query is run twice: as is, and with its relevant documents removed. AUROC = chance that the "
                  "answerable run scores higher. The threshold for a corpus is set on all *other* corpora so that 90% "
                  "of their answerable queries pass; rejection = unanswerable runs below it, false rejection = "
                  "answerable runs below it.", "",
                  "| corpus | AUROC | threshold | rejection | false rejection |", "|---|---:|---:|---:|---:|"]
        for n, r in runs.items():
            others = [x for m, o in runs.items() if m != n for x in o["answerable"]]
            tau = float(np.quantile(others, 0.10)) if others else 0.5
            a, u = np.array(r["answerable"]), np.array(r["unanswerable"])
            lines.append(f"| {n} | {auroc(r['answerable'], r['unanswerable']):.3f} | {tau:.3f} | "
                         f"{100 * (u < tau).mean():.1f}% | {100 * (a < tau).mean():.1f}% |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["run"]:
        names = [a for a in args[1:] if a in DATASETS]
        if "--shard" in args:
            k, n = map(int, args[args.index("--shard") + 1].split("/"))
            names = shard(k, n)
        failed = []
        for name, setting in jobs(names):
            try:
                run_dataset(name, setting)
            except Exception:   # the next corpus still runs; what finished is cached, a rerun resumes
                import traceback

                from peresearch.zetokrag.models import gpu_memory

                log(name, setting, "FAILED;", gpu_memory())
                traceback.print_exc()
                failed.append(f"{name}/{setting}")
            if failed and failed[-1] == f"{name}/{setting}":   # outside `except`: its frames hold the models
                from peresearch.zetokrag.models import free_gpu

                free_gpu()
        print(report())
        if failed:
            sys.exit(f"failed: {' '.join(failed)}")
    elif args == ["report"]:
        print(report())
    else:
        sys.exit(__doc__)
