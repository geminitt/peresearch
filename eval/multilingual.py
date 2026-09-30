"""Which ranking core should ZetokRAG use? One scoring pass, every variant analysed afterwards on the CPU.

    pixi run python eval/multilingual.py stats [set ...]    # sizes and text lengths (CPU): runs/multilingual/stats.json
    pixi run python eval/multilingual.py overlap [set ...]  # question words found in their relevant documents
    pixi run python eval/multilingual.py speed              # a small timing run of each model on the local GPU
    pixi run python eval/multilingual.py lexical [set ...]  # stage 1, CPU: BM25 views -> <set>/<setting>/lexical.npz
    pixi run python eval/multilingual.py scores [set ...]   # stage 2, GPU: dense, sparse, reranker (not MLDR)
    pixi run python eval/multilingual.py analyze [set ...]  # stage 3, CPU: every variant -> results/multilingual.md

The six target languages (the owner's choice, 2026-09-30): English, Vietnamese, Chinese, Korean, Japanese, Hindi.
Each has several test sets from different domains, pinned to a dataset commit. Queries above 1,000 are
subsampled with seed 0, as in eval/retrieval.py.
"""

import gzip
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("PERESEARCH_RUNS", ROOT / "runs")) / "multilingual"
MAX_QUERIES = 1000

sys.path.insert(0, str(Path(__file__).parent))     # eval/tokenization.py, imported by name (its process pool needs it)
_spec = importlib.util.spec_from_file_location("retrieval", Path(__file__).parent / "retrieval.py")
retrieval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrieval)
log, retry = retrieval.log, retrieval.retry

MIRACL = ("mteb/MIRACLRetrievalHardNegatives", "332a9acb49f5e83d5397683f79d23e588f685916")
# (revision, qrels revision, corpus file, queries file, qrels file): file names written out, so that loading works
# offline (listing a repository's files needs the network)
CMTEB = {"DuRetrieval": ("a1a333e290fe30b10f3f56498e3a0d911a693ced", "497b7bd1bbb25cb3757ff34d95a8be50a3de2279",
                         "data/corpus-00000-of-00001-19b9e924cb33e4d5.parquet",
                         "data/queries-00000-of-00001-7c7edb40be6b560c.parquet",
                         "data/dev-00000-of-00001-d3c385852a7c0c9d.parquet"),
         "CmedqaRetrieval": ("cd540c506dae1cf9e9a59c3e06f42030d54e7301", "279d737f36c731c8ff6e2b055f31fe02216fa23d",
                             "data/corpus-00000-of-00001-a3949861f65a3226.parquet",
                             "data/queries-00000-of-00001-daeedab899d3c839.parquet",
                             "data/dev-00000-of-00001-57fb84a4aceaa695.parquet")}
JMTEB = ("sbintuitions/JMTEB", "6064d6d00e3eccca0f9673621022b409f79d1a18")
MLDR = ("Shitao/MLDR", "d67138e705d963e346253a80e59676ddb418810a")

# name: (language, loader, arguments, what it is)
SETS = {
    "miracl-en": ("en", "mteb", (*MIRACL, "en-", "dev"), "Wikipedia, questions written by native speakers"),
    **{n: ("en", "gate1", (n,), d) for n, d in [
        ("scifact", "scientific claims"), ("nfcorpus", "medical questions"), ("fiqa", "financial questions"),
        ("arguana", "counterarguments"), ("scidocs", "citation prediction")]},
    **{n: ("vi", "gate1", (n,), d) for n, d in [
        ("zalo-legal-vn", "legal questions, written in Vietnamese"), ("nano-hotpotqa-vn", "multi-hop, Wikipedia (translated)"),
        ("nano-msmarco-vn", "web search (translated)"), ("fiqa-vn", "financial questions (translated)"),
        ("scifact-vn", "scientific claims (translated)")]},
    "miracl-zh": ("zh", "mteb", (*MIRACL, "zh-", "dev"), "Wikipedia"),
    "duretrieval-zh": ("zh", "cmteb", ("DuRetrieval",), "Baidu web search"),
    "cmedqa-zh": ("zh", "cmteb", ("CmedqaRetrieval",), "medical questions and answers"),
    "miracl-ja": ("ja", "mteb", (*MIRACL, "ja-", "dev"), "Wikipedia"),
    "jagovfaqs-ja": ("ja", "jmteb", ("jagovfaqs_22k",), "government FAQs"),
    "nlp-journal-title-abs-ja": ("ja", "jmteb", ("nlp_journal_title_abs",), "paper title to its abstract"),
    "mldr-ja": ("ja", "mldr", ("ja",), "long documents (Wikipedia)"),
    "miracl-ko": ("ko", "mteb", (*MIRACL, "ko-", "dev"), "Wikipedia"),
    "ko-strategyqa": ("ko", "mteb", ("mteb/Ko-StrategyQA", "315b000dfa35ba4713d2a266d204e68f8ac80687", "", "dev"),
                      "multi-hop questions"),
    "autorag-ko": ("ko", "mteb", ("mteb/AutoRAGRetrieval", "43b817937708cb10ba519f86edb4f6885a1631a4", "", "test"),
                   "finance and public-sector documents"),
    "mldr-ko": ("ko", "mldr", ("ko",), "long documents (Wikipedia)"),
    "miracl-hi": ("hi", "mteb", (*MIRACL, "hi-", "dev"), "Wikipedia"),
    "wikipedia-hi": ("hi", "mteb", ("mteb/WikipediaRetrievalMultilingual", "757426bd952ed691a381c46719bf78cd149041e7",
                                    "hi-", "test"), "Wikipedia, another source than MIRACL"),
    "xpqa-hi": ("hi", "mteb", ("mteb/XPQARetrieval", "fc4624be978945a0ceebbc4b85737258fe26330b", "hin-hin-", "test"),
                "product questions (e-commerce)"),
    "mldr-hi": ("hi", "mldr", ("hi",), "long documents (Wikipedia)"),
}


def _file(repo, rev, path):
    from huggingface_hub import hf_hub_download

    return retry(lambda: hf_hub_download(repo, path, repo_type="dataset", revision=rev))


def _finish(doc_ids, texts, rel, qtext):
    """Keep the judged documents that exist and the queries that have one; subsample queries with seed 0."""
    known = set(doc_ids)
    rel = {q: {d: s for d, s in r.items() if d in known} for q, r in rel.items()}
    qs = sorted(q for q, r in rel.items() if r and q in qtext)
    if len(qs) > MAX_QUERIES:
        qs = sorted(np.random.default_rng(0).choice(qs, MAX_QUERIES, replace=False).tolist())
    return doc_ids, texts, qs, [qtext[q] for q in qs], [rel[q] for q in qs]


def load(name: str):
    """(document ids, document texts, query ids, query texts, relevance per query) of a set."""
    import pandas as pd

    lang, kind, args, _ = SETS[name]
    if kind == "gate1":
        doc_ids, texts, qs, qtexts, rel = retrieval.load(*args)
        return doc_ids, texts, qs, qtexts, [rel[q] for q in qs]
    if kind == "mteb":
        repo, rev, prefix, split = args
        part = lambda p: pd.read_parquet(_file(repo, rev, f"{prefix}{p}/{split}-00000-of-00001.parquet"))
        corpus, queries, qrels = part("corpus"), part("queries"), part("qrels")
        text = corpus["text"].fillna("")
        if "title" in corpus:
            text = (corpus["title"].fillna("") + "\n" + text).str.strip()
        rel: dict = {}
        for q, d, s in zip(qrels["query-id"].astype(str), qrels["corpus-id"].astype(str), qrels["score"]):
            if int(s) > 0:
                rel.setdefault(q, {})[d] = int(s)
        cid, qid = ("_id" if "_id" in corpus else "id"), ("_id" if "_id" in queries else "id")
        return _finish(corpus[cid].astype(str).tolist(), text.tolist(), rel,
                       dict(zip(queries[qid].astype(str), queries["text"])))
    if kind == "cmteb":
        (task,) = args
        rev, qrev, cfile, qfile, rfile = CMTEB[task]
        corpus = pd.read_parquet(_file(f"C-MTEB/{task}", rev, cfile))
        queries = pd.read_parquet(_file(f"C-MTEB/{task}", rev, qfile))
        qrels = pd.read_parquet(_file(f"C-MTEB/{task}-qrels", qrev, rfile))
        rel = {}
        for q, d, s in zip(qrels["qid"].astype(str), qrels["pid"].astype(str), qrels["score"]):
            if int(s) > 0:
                rel.setdefault(q, {})[d] = int(s)
        return _finish(corpus["id"].astype(str).tolist(), corpus["text"].fillna("").tolist(), rel,
                       dict(zip(queries["id"].astype(str), queries["text"])))
    if kind == "jmteb":
        (task,) = args
        corpus = pd.read_parquet(_file(*JMTEB, f"data/{task}-corpus/corpus.parquet"))
        queries = pd.read_parquet(_file(*JMTEB, f"data/{task}-query/test.parquet"))
        text = corpus["text"].fillna("")
        if "title" in corpus:
            text = (corpus["title"].fillna("") + "\n" + text).str.strip()
        qids = queries["qid"].astype(str) if "qid" in queries else pd.Series(range(len(queries))).astype(str)
        rel, qtext = {}, {}
        for q, t, docs in zip(qids, queries["query"], queries["relevant_docs"]):
            docs = [docs] if isinstance(docs, (str, int, np.integer)) else list(docs)
            rel[q] = {str(d): 1 for d in docs}
            qtext[q] = t
        return _finish(corpus["docid"].astype(str).tolist(), text.tolist(), rel, qtext)
    if kind == "mldr":
        (lg,) = args
        doc_ids, texts = [], []
        with gzip.open(_file(*MLDR, f"mldr-v1.0-{lg}/corpus.jsonl.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                doc_ids.append(str(r["docid"]))
                texts.append(r["text"])
        rel, qtext = {}, {}
        with gzip.open(_file(*MLDR, f"mldr-v1.0-{lg}/test.jsonl.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rel[r["query_id"]] = {str(p["docid"]): 1 for p in r["positive_passages"]}
                qtext[r["query_id"]] = r["query"]
        return _finish(doc_ids, texts, rel, qtext)
    raise KeyError(kind)


def stats(names) -> dict:
    """Sizes, and text lengths in Qwen3-Embedding tokens: how much of each set the 512-token cut drops."""
    from transformers import AutoTokenizer

    from peresearch.zetokrag.models import EMBEDDERS as SPECS

    spec = SPECS["qwen3-embedding-0.6b"]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    path = RUNS / "stats.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        if name in out:
            continue
        doc_ids, texts, qs, qtexts, rels = load(name)
        n = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
        qn = [len(x) for x in tok(qtexts, add_special_tokens=False)["input_ids"]]
        out[name] = {"language": SETS[name][0], "documents": len(texts), "queries": len(qs),
                     "relevant_per_query": float(np.mean([len(r) for r in rels])),
                     "document_tokens": {"mean": float(np.mean(n)), "p50": float(np.percentile(n, 50)),
                                         "p90": float(np.percentile(n, 90)), "max": int(max(n)), "total": int(sum(n))},
                     "share_over_512": float(np.mean([x > 512 for x in n])),
                     "tokens_cut_by_512": float(sum(max(0, x - 512) for x in n) / max(1, sum(n))),
                     "query_tokens_mean": float(np.mean(qn))}
        log(name, out[name])
        RUNS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))
    return out


def speed(n_docs: int = 1536, n_queries: int = 48, depth: int = 64) -> dict:
    """A small timing run on the local GPU: document tokens per second of each embedder, and query-document pairs
    per second of the reranker, all at 512 tokens, on texts drawn from every target language."""
    import torch

    from peresearch.zetokrag.models import Embedder, Reranker, free_gpu, gpu_memory

    rng = np.random.default_rng(0)
    picks = ["miracl-en", "fiqa", "zalo-legal-vn", "nano-msmarco-vn", "miracl-zh", "cmedqa-zh", "miracl-ja",
             "jagovfaqs-ja", "miracl-ko", "ko-strategyqa", "miracl-hi", "wikipedia-hi"]
    docs, queries = [], []
    for name in picks:
        _, texts, _, qtexts, _ = load(name)
        docs += [texts[i] for i in rng.choice(len(texts), n_docs // len(picks), replace=False)]
        queries += [qtexts[i] for i in rng.choice(len(qtexts), n_queries // len(picks), replace=False)]
    out = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "documents": len(docs)}
    for name in ["qwen3-embedding-0.6b", "bge-m3"]:
        e = Embedder(name)
        tok = e.model.tokenizer
        tokens = sum(min(512, len(x)) for x in tok(docs, add_special_tokens=True)["input_ids"])
        e.documents(docs[:64])                                  # warm-up
        torch.cuda.synchronize()
        t = time.time()
        e.documents(docs)
        torch.cuda.synchronize()
        out[name] = {"tokens_per_second": tokens / (time.time() - t), "memory": gpu_memory()}
        log(name, out[name])
        del e
        free_gpu()
    r = Reranker()
    pairs = [(q, docs[i]) for q in queries for i in rng.choice(len(docs), depth, replace=False)]
    r.scores(pairs[0][0], [d for _, d in pairs[:32]])            # warm-up
    torch.cuda.synchronize()
    t = time.time()
    for k in range(0, len(pairs), depth):
        r.scores(pairs[k][0], [d for _, d in pairs[k:k + depth]])
    torch.cuda.synchronize()
    out["bge-reranker-v2-m3"] = {"pairs_per_second": len(pairs) / (time.time() - t), "memory": gpu_memory()}
    log("bge-reranker-v2-m3", out["bge-reranker-v2-m3"])
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "speed.json").write_text(json.dumps(out, indent=1))
    return out

# --- scoring: every view's top 100 per query, saved; the reranker's scores on the union the analysis needs ---------

TOP = 100
RHO = 0.5                     # ZetokRAG's min-max fusion weight
SPARSE_WEIGHT = 0.3           # BGE-M3's own dense + sparse mix (Chen et al. 2024, BGE M3-Embedding, s = s_dense + 0.3 s_lex)
FUSED_DEPTH = 50              # the reranker scores the top 50 of ZetokRAG's fusion, to test depths 30 and 50 ...
OTHER_DEPTH = 30              # ... and the top 30 of each other candidate list
LEXICAL = ["current", "unspaced-bigrams", "unspaced-xlmr", "unspaced-hangul-xlmr", "xlmr-whole"]
CANDIDATE = "unspaced-xlmr"   # the lexical view of the second fusion the reranker covers (best single index on MIRACL)
_CUTS = {"unspaced-bigrams": "words+marks+bigrams", "unspaced-xlmr": "words+marks+xlmr",
         "unspaced-hangul-xlmr": "words+marks+hangul+xlmr", "xlmr-whole": "xlmr"}
BGE = ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181")


def settings(name: str) -> list[str]:
    """Vietnamese queries are also run stripped of diacritics, as in gate 1."""
    return ["as-typed", "no-diacritics"] if SETS[name][0] == "vi" else ["as-typed"]


def queries_for(qtexts: list[str], setting: str) -> list[str]:
    from peresearch.zetokrag import core

    return [core.fold(q) for q in qtexts] if setting == "no-diacritics" else list(qtexts)


def _atomic_save(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _pack(lists) -> dict:
    """Ragged lists of (id, score) per query as padded arrays (-1 ids, -inf scores)."""
    ids = np.full((len(lists), TOP), -1, dtype=np.int32)
    sc = np.full((len(lists), TOP), -np.inf, dtype=np.float32)
    for r, (i, s) in enumerate(lists):
        ids[r, :len(i)], sc[r, :len(s)] = i[:TOP], s[:TOP]
    return {"ids": ids, "scores": sc}


def _unpack(ids, scores) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(i[i >= 0], s[i >= 0]) for i, s in zip(ids, scores)]


def lexical_terms(variant: str, texts: list[str], folded: bool) -> list[list[str]]:
    """The terms of each text: `current` is ZetokRAG's tokenize; the others cut as in eval/tokenization.py, then
    ZetokRAG's folding rule (a term without its combining marks is added next to it) for the folded index."""
    import tokenization

    from peresearch.zetokrag import core

    if variant == "current":
        return tokenization.cut_all("words-folded" if folded else "words", texts)
    terms = tokenization.cut_all(_CUTS[variant], texts)
    if not folded:
        return terms
    out = []
    for ts in terms:
        row = []
        for t in ts:
            row.append(t)
            bare = core.fold(t)
            if bare != t:
                row.append(bare)
        out.append(row)
    return out


def _term_ids(variant: str, texts: list[str], folded: bool, chunk: int = 1000):
    """The corpus as word ids (int32 per document) and the vocabulary, cut `chunk` texts at a time: MLDR's long
    documents as Python strings would take several GB (mldr-ja: ~10^8 terms)."""
    vocab: dict[str, int] = {}
    ids = []
    for a in range(0, len(texts), chunk):
        for terms in lexical_terms(variant, texts[a:a + chunk], folded):
            ids.append(np.fromiter((vocab.setdefault(t, len(vocab)) for t in (terms or ["∅"])), dtype=np.int32))
    return ids, vocab


def lexical_top(variant: str, texts: list[str], qtexts: list[str], own: list | None = None,
                low_memory: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """BM25 top 100 per query with ZetokRAG's rule: queries typed with diacritics search the plain index, the
    others the folded one. As in ZetokRAG, the top 100 of the full score vector: when fewer documents share a term
    with the query, documents with score 0 fill the list (they set the minimum of the min-max normalization)."""
    import bm25s

    from peresearch.zetokrag import core
    from tokenization import WORKERS

    out: list = [None] * len(qtexts)
    own = own or [None] * len(qtexts)
    for folded in (False, True):
        pick = [k for k, q in enumerate(qtexts) if core.accented(q) != folded]
        if not pick:
            continue
        model = bm25s.BM25()
        if low_memory:
            model.index(_term_ids(variant, texts, folded), show_progress=False)
        else:
            model.index([t or ["∅"] for t in lexical_terms(variant, texts, folded)], show_progress=False)
        qterms = lexical_terms(variant, [qtexts[k] for k in pick], folded)
        ids, sc = model.retrieve([t or ["∅"] for t in qterms], k=min(TOP + 1, len(texts)), show_progress=False,
                                 n_threads=WORKERS)
        for k, i, s in zip(pick, ids, sc):
            keep = i != own[k] if own[k] is not None else np.ones(len(i), bool)   # ArguAna: the query's own text
            out[k] = (i[keep][:TOP].astype(np.int32), s[keep][:TOP].astype(np.float32))
    return out


def run_lexical(names) -> list[str]:
    """Stage 1 (CPU): the BM25 views of every set and query setting, saved to <set>/<setting>/lexical.npz."""
    import traceback

    failed = []
    for name in names:
        try:
            todo = [st for st in settings(name) if not (RUNS / name / st / "lexical.npz").exists()]
            if not todo:
                continue
            doc_ids, texts, qs, qtexts, rels = load(name)
            own = retrieval.own_documents(name, doc_ids, qs)
            for st in todo:
                t = time.time()
                qt = queries_for(qtexts, st)
                arrays = {}
                for v in LEXICAL:
                    tv = time.time()
                    packed = _pack(lexical_top(v, texts, qt, own, low_memory=name.startswith("mldr")))
                    arrays[f"{v}:ids"], arrays[f"{v}:scores"] = packed["ids"], packed["scores"]
                    log(name, st, v, f"{time.time() - tv:.0f}s")
                (RUNS / name / st).mkdir(parents=True, exist_ok=True)
                _atomic_save(RUNS / name / st / "lexical.npz", **arrays)
                log(name, st, "lexical", f"{time.time() - t:.0f}s")
        except Exception:
            log(name, "lexical FAILED")
            traceback.print_exc()
            failed.append(name)
    return failed


class BgeM3:
    """BGE-M3's dense vector (the CLS state, unit length) and its sparse lexical weights (ReLU of a linear head
    on every token, the largest weight per token id, special tokens left out), from one forward pass, as
    FlagEmbedding's BGEM3FlagModel computes them; checked against the model card's printed weights."""

    def __init__(self, batch_size: int = 32):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        from peresearch.zetokrag.models import _device

        self.device = _device()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(BGE[0], revision=BGE[1])
        self.model = AutoModel.from_pretrained(BGE[0], revision=BGE[1], dtype=dtype).to(self.device).eval()
        self.head = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.head.load_state_dict(torch.load(hf_hub_download(BGE[0], "sparse_linear.pt", revision=BGE[1]),
                                             map_location="cpu"))
        self.head = self.head.to(self.device, dtype).eval()
        self.special = {self.tok.cls_token_id, self.tok.eos_token_id, self.tok.pad_token_id, self.tok.unk_token_id}
        self.batch_size = batch_size

    def encode(self, texts: list[str]):
        """(dense float16 [n, 1024], sparse CSR [n, vocab] float32) for the texts, cut at 512 tokens."""
        import scipy.sparse as sp
        import torch

        order = np.argsort([-len(t) for t in texts], kind="stable")   # similar lengths per batch: less padding
        dense = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float16)
        rows, cols, vals = [], [], []
        for a in range(0, len(texts), self.batch_size):
            idx = order[a:a + self.batch_size]
            b = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=512,
                         return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.model(**b).last_hidden_state
                d = torch.nn.functional.normalize(h[:, 0].float(), dim=-1)
                w = torch.relu(self.head(h)).squeeze(-1).float()
            dense[idx] = d.cpu().numpy().astype(np.float16)
            for r, (ids, ws) in zip(idx, zip(b["input_ids"].cpu().numpy(), w.cpu().numpy())):
                best: dict[int, float] = {}
                for i, x in zip(ids.tolist(), ws.tolist()):
                    if x > 0 and i not in self.special and x > best.get(i, 0.0):
                        best[i] = x
                rows += [r] * len(best)
                cols += list(best)
                vals += list(best.values())
        sparse = sp.csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(len(texts), len(self.tok)))
        return dense, sparse


def _embeddings(name: str, model: str, texts: list[str], base: Path):
    """Document vectors, saved in slices so an interrupted run resumes. Qwen3's are ZetokRAG's Embedder, the same
    file eval/retrieval.py writes (so gate 1's can be reused); BGE-M3's come with its sparse weights."""
    import scipy.sparse as sp

    from peresearch.zetokrag.models import Embedder, free_gpu, gpu_memory

    if model == "qwen3-embedding-0.6b":
        return retrieval.document_embeddings(name, model, texts, base)
    dense_path, sparse_path = base / "docs-bge-m3-dense.npy", base / "docs-bge-m3-sparse.npz"
    if dense_path.exists() and sparse_path.exists():
        return np.load(dense_path), sp.load_npz(sparse_path)
    step, enc, t, parts = 16384, None, time.time(), []
    for a in range(0, len(texts), step):
        part_d, part_s = base / f"docs-bge-m3-dense.{a:08d}.npy", base / f"docs-bge-m3-sparse.{a:08d}.npz"
        if not (part_d.exists() and part_s.exists()):
            enc = enc or BgeM3()
            d, s_ = enc.encode(texts[a:a + step])
            np.save(part_d, d)
            sp.save_npz(part_s, s_)
            log(name, "bge-m3", f"{min(a + step, len(texts))}/{len(texts)} docs, {time.time() - t:.0f}s;", gpu_memory())
        parts.append((part_d, part_s))
    dense = np.concatenate([np.load(d) for d, _ in parts])
    sparse = sp.vstack([sp.load_npz(s_) for _, s_ in parts]).tocsr()
    np.save(dense_path, dense)
    sp.save_npz(sparse_path, sparse)
    for d, s_ in parts:
        d.unlink()
        s_.unlink()
    del enc
    free_gpu()
    return dense, sparse


def _top(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from peresearch.zetokrag import core

    i = core.top(scores, TOP)
    return i.astype(np.int32), scores[i].astype(np.float32)


def fuse(a: tuple, b: tuple, rho: float = RHO, guard: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """ZetokRAG's min-max fusion computed from two views' top 100 (ids, scores): the same as core.fuse_minmax
    on the full score vectors, which normalizes each view over its own top 100 (a text missing from a view gets 0).
    Note: when BM25 matches nothing, all its scores are 0 and core.minmax maps them to 1, so 100 arbitrary
    documents get rho each. `guard` drops a lexical view whose best score is 0 (the fix under test)."""
    from peresearch.zetokrag import core

    fused: dict[int, float] = {}
    for weight, (ids, sc) in ((rho, a), (1 - rho, b)):
        if guard and weight == rho and (len(sc) == 0 or float(np.max(sc)) <= 0):
            continue
        if len(ids):
            for i, x in zip(ids, core.minmax(np.asarray(sc, dtype=np.float32))):
                fused[int(i)] = fused.get(int(i), 0.0) + weight * float(x)
    order = sorted(fused, key=lambda i: -fused[i])
    return np.array(order, dtype=np.int32), np.array([fused[i] for i in order], dtype=np.float32)


def rerank_union(views: dict) -> list[set[int]]:
    """Per query, the documents the reranker must score: the top 50 of ZetokRAG's fusion as used, and the top 30
    of Qwen3 alone, BGE-M3 dense+sparse and the fusion with the candidate tokenization."""
    out = []
    for k in range(len(views["qwen3"])):
        u = set(fuse(views["lexical:current"][k], views["qwen3"][k])[0][:FUSED_DEPTH].tolist())
        u |= set(fuse(views[f"lexical:{CANDIDATE}"][k], views["qwen3"][k])[0][:OTHER_DEPTH].tolist())
        for v in ("qwen3", "bge-m3-hybrid"):
            u |= set(views[v][k][0][:OTHER_DEPTH].tolist())
        out.append(u)
    return out


def load_views(name: str, setting: str) -> dict:
    views = {}
    for fname, prefix in (("lexical.npz", "lexical:"), ("dense.npz", "")):
        z = np.load(RUNS / name / setting / fname)
        for key in {k.split(":ids")[0] for k in z.files if k.endswith(":ids")}:
            views[prefix + key] = _unpack(z[f"{key}:ids"], z[f"{key}:scores"])
    return views


def run_scores(names) -> list[str]:
    """Stage 2 (GPU): Qwen3 and BGE-M3 (dense, sparse, dense + 0.3 sparse) top 100 per query, then the reranker
    on `rerank_union`; needs stage 1's lexical.npz. Everything is saved per set and setting, the reranker every
    100 queries, so a rerun resumes."""
    import traceback

    import torch

    from peresearch.zetokrag.models import Embedder, Reranker, free_gpu, gpu_memory

    retrieval.preflight()
    failed = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in names:
        try:
            if all((RUNS / name / st / "rerank.npz").exists() for st in settings(name)):
                continue
            base = RUNS / name
            doc_ids, texts, qs, qtexts, rels = load(name)
            own = retrieval.own_documents(name, doc_ids, qs)
            log(name, f"{len(texts)} documents, {len(qs)} queries;", gpu_memory())
            q3_docs = _embeddings(name, "qwen3-embedding-0.6b", texts, base)
            bge_dense, bge_sparse = _embeddings(name, "bge-m3", texts, base)
            sparse_t = bge_sparse.T.tocsr()
            for st in settings(name):
                out = base / st
                qt = queries_for(qtexts, st)
                if not (out / "dense.npz").exists():
                    e = Embedder("qwen3-embedding-0.6b")
                    qv = e.queries(qt)
                    del e
                    free_gpu()
                    bd, bs = BgeM3().encode(qt)
                    free_gpu()
                    lists = {"qwen3": [], "bge-m3-dense": [], "bge-m3-sparse": [], "bge-m3-hybrid": []}
                    D3 = torch.from_numpy(q3_docs).to(device)
                    DB = torch.from_numpy(bge_dense).to(device)
                    for a in range(0, len(qt), 256):
                        s3 = (D3 @ torch.from_numpy(qv[a:a + 256]).to(device).T).float().T.cpu().numpy()
                        sb = (DB @ torch.from_numpy(bd[a:a + 256]).to(device).T).float().T.cpu().numpy()
                        ss = (bs[a:a + 256] @ sparse_t).toarray().astype(np.float32)
                        for j in range(len(s3)):
                            x = own[a + j]
                            if x is not None:                       # ArguAna: never rank the query's own text
                                s3[j][x] = sb[j][x] = -np.inf
                                ss[j][x] = 0.0
                            lists["qwen3"].append(_top(s3[j]))
                            lists["bge-m3-dense"].append(_top(sb[j]))
                            lists["bge-m3-sparse"].append(_top(np.where(ss[j] > 0, ss[j], -np.inf)))
                            lists["bge-m3-hybrid"].append(_top(sb[j] + SPARSE_WEIGHT * ss[j]))
                    del D3, DB
                    free_gpu()
                    arrays = {}
                    for v, l in lists.items():
                        p = _pack(l)
                        arrays[f"{v}:ids"], arrays[f"{v}:scores"] = p["ids"], p["scores"]
                    _atomic_save(out / "dense.npz", **arrays)
                    log(name, st, "dense views saved")
                if (out / "rerank.npz").exists():
                    continue
                union = rerank_union(load_views(name, st))
                part = out / "rerank.part.npz"
                done_ids, done_scores, start = [], [], 0
                if part.exists():
                    z = np.load(part, allow_pickle=False)
                    start = int(z["queries_done"])
                    done_ids, done_scores = list(np.split(z["ids"], z["offsets"][1:-1])), \
                        list(np.split(z["scores"], z["offsets"][1:-1]))
                    done_ids, done_scores = done_ids[:start], done_scores[:start]
                rr, t = Reranker(), time.time()
                for k in range(start, len(qt)):
                    ids = np.array(sorted(union[k]), dtype=np.int32)
                    done_ids.append(ids)
                    done_scores.append(rr.scores(qt[k], [texts[i] for i in ids]).astype(np.float32))
                    if (k + 1) % 100 == 0 or k + 1 == len(qt):
                        offsets = np.cumsum([0] + [len(x) for x in done_ids])
                        _atomic_save(part, ids=np.concatenate(done_ids), scores=np.concatenate(done_scores),
                                     offsets=offsets, queries_done=np.array(k + 1))
                        log(name, st, "reranked", f"{k + 1}/{len(qt)} queries, {time.time() - t:.0f}s")
                os.replace(part, out / "rerank.npz")
                del rr
                free_gpu()
        except Exception:
            log(name, "scores FAILED;", gpu_memory())
            traceback.print_exc()
            failed.append(name)
        free_gpu()
    return failed


# --- analysis (CPU): every variant from the saved scores ----------------------------------------------------------

def rrf(a: tuple, b: tuple, k: int = 60) -> tuple[np.ndarray, np.ndarray]:
    fused: dict[int, float] = {}
    for ids, _ in (a, b):
        for r, i in enumerate(ids, start=1):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (k + r)
    order = sorted(fused, key=lambda i: -fused[i])
    return np.array(order, dtype=np.int32), np.array([fused[i] for i in order], dtype=np.float32)


def reranked(order: np.ndarray, cached: dict[int, float], depth: int) -> list[int] | None:
    """The first `depth` documents reordered by the reranker's score, the rest after them in their order (as
    core.rerank); None when a document of the head was not scored (the variant is not measurable then)."""
    head = [int(i) for i in order[:depth]]
    if any(i not in cached for i in head):
        return None
    ranked = [head[j] for j in np.argsort([-cached[i] for i in head], kind="stable")]
    return ranked + [int(i) for i in order[depth:]]


def variants(views: dict, cached: list[dict] | None) -> dict[str, list]:
    """Name -> per-query ranking (document indices) of every variant analysed."""
    n = len(next(iter(views.values())))
    out: dict[str, list] = {}
    for v in LEXICAL:
        out[f"bm25 {v}"] = [views[f"lexical:{v}"][k][0] for k in range(n)]
    if "qwen3" not in views:
        return out                                  # MLDR: BM25 only
    for v in ("qwen3", "bge-m3-dense", "bge-m3-sparse", "bge-m3-hybrid"):
        out[v] = [views[v][k][0] for k in range(n)]
    for v in LEXICAL:
        for rho in (0.3, 0.5, 0.7):
            out[f"fusion {v} rho={rho}"] = [fuse(views[f"lexical:{v}"][k], views["qwen3"][k], rho)[0] for k in range(n)]
        out[f"rrf {v}"] = [rrf(views[f"lexical:{v}"][k], views["qwen3"][k])[0] for k in range(n)]
        out[f"fusion {v} rho=0.5 guarded"] = [fuse(views[f"lexical:{v}"][k], views["qwen3"][k], guard=True)[0]
                                              for k in range(n)]
    if cached is None:
        return out
    plans = {"zetokrag (as used)": ("fusion current rho=0.5", 30),
             "zetokrag, rerank top 50": ("fusion current rho=0.5", 50),
             f"zetokrag with bm25 {CANDIDATE}": (f"fusion {CANDIDATE} rho=0.5", 30),
             "qwen3 + rerank": ("qwen3", 30), "bge-m3-hybrid + rerank": ("bge-m3-hybrid", 30)}
    for name, (base, depth) in plans.items():
        rows = [reranked(np.asarray(o), c, depth) for o, c in zip(out[base], cached)]
        if all(r is not None for r in rows):
            out[name] = rows
        else:
            log("not measurable:", name, sum(r is None for r in rows), "queries lack reranker scores")
    return out


def analyse_one(name: str, setting: str) -> dict:
    """Per-query metrics of every variant for one set and setting; cached in metrics.json."""
    path = RUNS / name / setting / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    doc_ids, _, qs, _, rels = load(name)
    views = load_views(name, setting) if (RUNS / name / setting / "dense.npz").exists() else {
        f"lexical:{k.split(':ids')[0]}": _unpack(z[k], z[k.replace(':ids', ':scores')])
        for z in [np.load(RUNS / name / setting / "lexical.npz")] for k in z.files if k.endswith(":ids")}
    cached = None
    if (RUNS / name / setting / "rerank.npz").exists():
        z = np.load(RUNS / name / setting / "rerank.npz")
        ids, sc = np.split(z["ids"], z["offsets"][1:-1]), np.split(z["scores"], z["offsets"][1:-1])
        cached = [dict(zip(i.tolist(), x.tolist())) for i, x in zip(ids, sc)]
    per = {v: [retrieval.metrics([doc_ids[i] for i in o], r) for o, r in zip(orders, rels)]
           for v, orders in variants(views, cached).items()}
    depth_recall = {}
    if "qwen3" in views:
        pos = {d: i for i, d in enumerate(doc_ids)}
        relevant = [{pos[d] for d in r} for r in rels]
        fused = [fuse(views["lexical:current"][k], views["qwen3"][k])[0] for k in range(len(qs))]
        depth_recall = {f"@{d}": float(np.mean([len(set(f[:d].tolist()) & r) / len(r) for f, r in zip(fused, relevant)]))
                        for d in (10, 30, 50, 100)}
    result = {"set": name, "setting": setting, "language": SETS[name][0], "queries": qs,
              "per_query": {v: {m: [x[m] for x in rows] for m in rows[0]} for v, rows in per.items()},
              "empty_bm25": {v: float(np.mean([len(views[f"lexical:{v}"][k][1]) == 0 or
                                               float(np.max(views[f"lexical:{v}"][k][1])) <= 0 for k in range(len(qs))]))
                             for v in LEXICAL},
              "fusion_recall_by_depth": depth_recall}
    path.write_text(json.dumps(result))
    return result


GROUPS = ["en", "vi", "vi-no-diacritics", "zh", "ja", "ko", "hi"]
# Candidate ranking cores and how many models each runs (a simpler core wins within 1 point).
CORES = {"qwen3": 1, "bge-m3-hybrid": 1, "qwen3 + rerank": 2, "bge-m3-hybrid + rerank": 2,
         **{f"fusion {v} rho={r}": 2 for v in LEXICAL for r in (0.3, 0.5, 0.7)}, **{f"rrf {v}": 2 for v in LEXICAL},
         **{f"fusion {v} rho=0.5 guarded": 2 for v in LEXICAL},
         "zetokrag (as used)": 3, "zetokrag, rerank top 50": 3, f"zetokrag with bm25 {CANDIDATE}": 3}
BASELINE = "zetokrag (as used)"


def group_of(r: dict) -> str:
    return "vi-no-diacritics" if r["setting"] == "no-diacritics" else r["language"]


def mean6(g: dict[str, float]) -> float:
    """The six languages alike; Vietnamese is the mean of its two ways of typing."""
    vi = [g[x] for x in ("vi", "vi-no-diacritics") if x in g]
    parts = [g[x] for x in ("en", "zh", "ja", "ko", "hi") if x in g] + ([float(np.mean(vi))] if vi else [])
    return float(np.mean(parts)) if parts else float("nan")


def gate1_check(results: list[dict]) -> list[str]:
    """ZetokRAG as used on the gate-1 sets must give gate 1's own per-query nDCG@10 (same code, data, models)."""
    lines = []
    for r in results:
        if SETS[r["set"]][1] != "gate1" or BASELINE not in r["per_query"]:
            continue
        found = sorted(ROOT.glob(f"runs/kaggle/*/runs/retrieval/{r['set']}/{r['setting']}/metrics.json"))
        if not found:
            lines.append(f"| {r['set']} | {r['setting']} | gate-1 metrics not found | | |")
            continue
        old = json.loads(found[0].read_text())
        if old["queries"] != r["queries"]:
            lines.append(f"| {r['set']} | {r['setting']} | different queries | | |")
            continue
        a = np.array([x["ndcg@10"] for x in old["per_query"]["zetokrag"]])
        b = np.array(r["per_query"][BASELINE]["ndcg@10"])
        lines.append(f"| {r['set']} | {r['setting']} | {100 * a.mean():.2f} | {100 * b.mean():.2f} | "
                     f"{int((np.abs(a - b) < 1e-9).sum())}/{len(a)} |")
    return lines


def analyze(names, n_boot: int = 10_000, out_path: Path = None) -> str:
    out_path = out_path or Path(os.environ.get("PERESEARCH_RESULTS", ROOT / "results")) / "multilingual.md"
    rng = np.random.default_rng(0)
    results = [analyse_one(n, st) for n in names for st in settings(n) if (RUNS / n / st / "lexical.npz").exists()]
    full = [r for r in results if BASELINE in r["per_query"]]
    set_mean = lambda r, v, m="ndcg@10": 100 * float(np.mean(r["per_query"][v][m]))

    def by_group(rs, v, m="ndcg@10"):
        g: dict[str, list] = {}
        for r in rs:
            if v in r["per_query"]:
                g.setdefault(group_of(r), []).append(set_mean(r, v, m))
        return {k: float(np.mean(x)) for k, x in g.items()}

    lines = ["# Choosing ZetokRAG's ranking core: six languages", "",
             f"{len(full)} set-settings scored in full, {len(results) - len(full)} with BM25 only (MLDR). nDCG@10 "
             "unless stated; a language is the mean of its sets; `mean 6` counts Vietnamese once (the mean of as "
             "typed and without diacritics). Sets, revisions and variants are defined in eval/multilingual.py.", ""]
    cores = [c for c in CORES if full and all(c in r["per_query"] for r in full)]
    table = {c: by_group(full, c) for c in cores}
    lines += ["## Ranking cores by language", "", "| core | models | " + " | ".join(GROUPS) + " | mean 6 |",
              "|---|---:|" + "---:|" * (len(GROUPS) + 1)]
    for c in sorted(cores, key=lambda c: -mean6(table[c])):
        lines.append(f"| {c} | {CORES[c]} | " + " | ".join(f"{table[c].get(g, float('nan')):.1f}" for g in GROUPS)
                     + f" | **{mean6(table[c]):.1f}** |")
    if cores:
        best = max(cores, key=lambda c: mean6(table[c]))
        base = table[BASELINE]
        allowed = [c for c in cores if all(table[c].get(g, 0) >= base.get(g, 0) - 1.0 for g in GROUPS if g in base)]
        pick = min((c for c in allowed if mean6(table[c]) >= mean6(table[best]) - 1.0),
                   key=lambda c: (CORES[c], -mean6(table[c])), default=None)
        lines += ["", "## Decision by the rule set beforehand", "",
                  "Best mean over the six languages; a core with fewer models within 1 point of the best wins; no "
                  f"language may fall more than 1 point below `{BASELINE}`.", "",
                  f"- best mean: `{best}` ({mean6(table[best]):.1f}); ZetokRAG as used: {mean6(base):.1f}",
                  f"- cores that keep every language within 1 point: {len(allowed)} of {len(cores)}",
                  f"- **pick: `{pick}`**" + (f" ({mean6(table[pick]):.1f}, {CORES[pick]} models)" if pick else
                                            " (none passes the rule)"), "",
                  f"{len(cores)} cores are compared on the same queries, so the best of them is favoured by chance: "
                  "the intervals below are per comparison, not corrected for the number of comparisons.", ""]
        lines += [f"## Each core minus `{BASELINE}` (nDCG@10 points, paired bootstrap 95% interval)", "",
                  "| core | " + " | ".join(GROUPS) + " |", "|---|" + "---|" * len(GROUPS)]
        for c in sorted(cores, key=lambda c: -mean6(table[c])):
            if c == BASELINE:
                continue
            cells = []
            for g in GROUPS:
                sel = [r for r in full if group_of(r) == g]
                if not sel:
                    cells.append("")
                    continue
                diffs = [np.array(r["per_query"][c]["ndcg@10"]) - np.array(r["per_query"][BASELINE]["ndcg@10"])
                         for r in sel]
                point = np.mean([d.mean() for d in diffs])
                boot = np.mean([[d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)] for d in diffs],
                               axis=0)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                cells.append(f"{100 * point:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines += ["", "## Per set (nDCG@10)", "", "| set | setting | lang | queries | " + " | ".join(
            c for c in cores if CORES[c] != 2 or c.startswith("fusion current rho=0.5")) + " |",
                  "|---|---|---|---:|" + "---:|" * len([c for c in cores if CORES[c] != 2 or
                                                         c.startswith("fusion current rho=0.5")])]
        shown = [c for c in cores if CORES[c] != 2 or c.startswith("fusion current rho=0.5")]
        for r in full:
            lines.append(f"| {r['set']} | {r['setting']} | {r['language']} | {len(r['queries'])} | " +
                         " | ".join(f"{set_mean(r, c):.1f}" for c in shown) + " |")
    lex = {f"bm25 {v}": by_group(results, f"bm25 {v}") for v in LEXICAL}
    lines += ["", "## BM25 alone by language (MLDR included)", "", "| view | " + " | ".join(GROUPS) + " | mean 6 |",
              "|---|" + "---:|" * (len(GROUPS) + 1)]
    for v, g in lex.items():
        lines.append(f"| {v} | " + " | ".join(f"{g.get(x, float('nan')):.1f}" for x in GROUPS) + f" | {mean6(g):.1f} |")
    mldr = [r for r in results if r["set"].startswith("mldr")]
    if mldr:
        lines += ["", "## MLDR (long documents), BM25 only", "", "| set | " + " | ".join(LEXICAL) + " |",
                  "|---|" + "---:|" * len(LEXICAL)]
        for r in mldr:
            lines.append(f"| {r['set']} | " + " | ".join(f"{set_mean(r, f'bm25 {v}'):.1f}" for v in LEXICAL) + " |")
    lines += ["", "## Diagnostics", "", "Queries for which BM25 finds no document at all (share), and how much of the "
              "relevant set ZetokRAG's fusion holds at each depth before reranking (recall).", "",
              "| set | setting | empty BM25 current | empty BM25 " + CANDIDATE + " | recall@10 | @30 | @50 | @100 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        d = r.get("fusion_recall_by_depth") or {}
        lines.append(f"| {r['set']} | {r['setting']} | {100 * r['empty_bm25']['current']:.1f}% | "
                     f"{100 * r['empty_bm25'][CANDIDATE]:.1f}% | " +
                     " | ".join(f"{100 * d[k]:.1f}" if k in d else "" for k in ("@10", "@30", "@50", "@100")) + " |")
    check = gate1_check(full)
    if check:
        lines += ["", "## Cross-check with gate 1", "", "ZetokRAG as used, recomputed here, against gate 1's own "
                  "numbers (same code, data and models; embeddings reused).", "",
                  "| set | setting | gate 1 | here | queries identical |", "|---|---|---:|---:|---:|", *check]
    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text


# --- Kaggle helpers ---------------------------------------------------------------------------------------------

REUSED = {"scifact", "nfcorpus", "fiqa", "arguana", "scidocs",
          "zalo-legal-vn", "nano-hotpotqa-vn", "nano-msmarco-vn", "fiqa-vn", "scifact-vn"}   # gate-1 Qwen3 vectors


def cost(name: str) -> float:
    """GPU minutes of a set on one T4, from the laptop's measured rates (which match a T4's for the embedders, see
    CLAUDE.md) and the set's measured text lengths; only used to balance the two GPUs."""
    return COST.get(name, 10.0)


# GPU minutes per set (2026-09-30): Qwen3 at 6.9k tokens/s unless its vectors are reused, BGE-M3 at 18k, the
# reranker at 33k tokens/s on 65 documents per query (the union's mean size in the laptop test: 60-71), + 1.
COST: dict[str, float] = {
    "autorag-ko": 4.1, "xpqa-hi": 3.4, "nlp-journal-title-abs-ja": 7.3, "ko-strategyqa": 12.3, "wikipedia-hi": 28.9,
    "jagovfaqs-ja": 17.7, "miracl-en": 76.7, "scifact": 6.0, "nfcorpus": 6.0, "fiqa": 13.6, "arguana": 18.0,
    "scidocs": 14.1, "zalo-legal-vn": 38.6, "nano-hotpotqa-vn": 19.2, "nano-msmarco-vn": 16.7, "fiqa-vn": 17.8,
    "scifact-vn": 7.1, "miracl-zh": 31.4, "duretrieval-zh": 72.6, "cmedqa-zh": 69.4, "miracl-ja": 98.8,
    "miracl-ko": 28.7, "miracl-hi": 77.6}


def shards(names: list[str], n: int) -> list[list[str]]:
    """Largest first onto the least loaded GPU."""
    load_, out = [0.0] * n, [[] for _ in range(n)]
    for name in sorted(names, key=lambda x: -cost(x)):
        k = int(np.argmin(load_))
        out[k].append(name)
        load_[k] += cost(name)
    return out


def prefetch(names) -> None:
    """Download every set and model once, before the workers start (they then run offline)."""
    from huggingface_hub import hf_hub_download

    from peresearch.zetokrag.models import EMBEDDERS as SPECS, RERANKER, Embedder, Reranker

    for name in names:
        doc_ids, _, qs, _, _ = load(name)
        log(name, f"{len(doc_ids)} documents, {len(qs)} queries")
    retry(lambda: Embedder("qwen3-embedding-0.6b"))
    retry(lambda: BgeM3())
    retry(lambda: hf_hub_download(BGE[0], "sparse_linear.pt", revision=BGE[1]))
    retry(Reranker)
    log("models ready:", SPECS["qwen3-embedding-0.6b"]["revision"][:7], BGE[1][:7], RERANKER["revision"][:7])


def reuse_check(names, n: int = 64) -> dict:
    """Gate 1's Qwen3 document vectors are reused only if this code re-embeds a sample of each set to the same
    vectors (cosine ≥ 0.9999 for every sampled document: fp16 on the same GPU type, batching may move the last
    bits); a set that fails has its file removed, so its vectors are computed afresh."""
    from peresearch.zetokrag.models import Embedder, free_gpu

    e, out = Embedder("qwen3-embedding-0.6b"), {}
    for name in names:
        path = RUNS / name / "docs-qwen3-embedding-0.6b.npy"
        if not path.exists():
            continue
        _, texts, _, _, _ = load(name)
        old = np.load(path, mmap_mode="r")
        idx = np.linspace(0, len(texts) - 1, n).astype(int)
        ok = old.shape == (len(texts), old.shape[1])
        cos, same = [], 0
        if ok:
            half = e.documents([texts[i] for i in idx])
            new, ref = half.astype(np.float32), np.asarray(old[idx], dtype=np.float32)
            cos = (new * ref).sum(1) / (np.linalg.norm(new, axis=1) * np.linalg.norm(ref, axis=1))
            same = int((half == old[idx]).all(axis=1).sum())
            ok = bool(cos.min() >= 0.9999)
        out[name] = {"shape_ok": old.shape[0] == len(texts), "min_cosine": float(min(cos)) if len(cos) else None,
                     "bit_identical": f"{same}/{n}", "reused": ok}
        log("reuse", name, out[name])
        if not ok:
            path.unlink()
    del e
    free_gpu()
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / "reuse_check.json"                  # a resumed run adds to the record, never erases it
    record = {**(json.loads(path.read_text()) if path.exists() else {}), **out}
    path.write_text(json.dumps(record, indent=1))
    return out


def overlap_scores(qtexts, doc_texts_per_query, corpus_texts) -> dict:
    """How many of a question's words appear in its relevant documents. Words found in more than 10% of the
    corpus's documents are left out (they would count as overlap anywhere); each question counts once."""
    from peresearch.zetokrag import core

    df: dict[str, int] = {}
    for t in corpus_texts:
        for w in set(core.tokenize(t, folded=False)):
            df[w] = df.get(w, 0) + 1
    common = {w for w, n in df.items() if n > 0.1 * len(corpus_texts)}
    shares = []
    for q, docs in zip(qtexts, doc_texts_per_query):
        words = set(core.tokenize(q, folded=False)) - common
        if words:
            seen = set().union(*(set(core.tokenize(d, folded=False)) for d in docs)) if docs else set()
            shares.append(len(words & seen) / len(words))
    return {"questions": len(shares), "mean_share": float(np.mean(shares)),
            "share_all_found": float(np.mean([s == 1.0 for s in shares]))}


def overlap(names) -> dict:
    """Word overlap between questions and their relevant documents: the private set (read in place, only the
    numbers are kept) against public sets whose questions were written independently of the documents."""
    from peresearch import guard

    out = {}
    home = guard.home() / "eval"
    if (home / "personal.jsonl").exists():
        items = [json.loads(x) for x in open(home / "personal.jsonl", encoding="utf-8")]
        chunks = {(c["path"], c["unit"], c["start"], c["end"]): c["text"]
                  for c in map(json.loads, open(home / "chunks.jsonl", encoding="utf-8"))}
        ans = [it for it in items if it["answerable"]]
        docs = [[chunks.get((r["path"], r["unit"], r["start"], r["end"]), "") for r in it["relevant"]] for it in ans]
        out["private-85"] = overlap_scores([it["question"] for it in ans], docs, list(chunks.values()))
        log("private-85", out["private-85"])
    for name in names:
        doc_ids, texts, qs, qtexts, rels = load(name)
        pos = {d: i for i, d in enumerate(doc_ids)}
        docs = [[texts[pos[d]] for d in r] for r in rels]
        out[name] = overlap_scores(qtexts, docs, texts)
        log(name, out[name])
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "overlap.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] in (["lexical"], ["scores"], ["analyze"]):
        names = [a for a in args[1:] if a in SETS] or list(SETS)
        if args[0] == "lexical":
            failed = run_lexical(names)
        elif args[0] == "scores":
            failed = run_scores([n for n in names if not n.startswith("mldr")])
        else:
            failed = []
            print(analyze(names))
        if failed:
            sys.exit(f"failed: {' '.join(failed)}")
    elif args[:1] == ["prefetch"]:
        prefetch([a for a in args[1:] if a in SETS])
    elif args[:1] == ["reuse-check"]:
        print(json.dumps(reuse_check([a for a in args[1:] if a in SETS]), indent=1))
    elif args == ["speed"]:
        print(json.dumps(speed(), indent=1))
    elif args[:1] == ["overlap"]:
        print(json.dumps(overlap([a for a in args[1:] if a in SETS]), indent=1))
    elif args[:1] == ["stats"]:
        print(json.dumps(stats([a for a in args[1:] if a in SETS] or list(SETS)), indent=1))
    else:
        sys.exit(__doc__)
