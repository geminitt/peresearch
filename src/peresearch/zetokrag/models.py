"""The local encoder models, pinned to exact Hugging Face revisions (a moved revision is a different model).

ZetokRAG embeds with Qwen3-Embedding-0.6B, chosen over BGE-M3 by a rule fixed before the benchmark
(results/retrieval.md), and reranks with bge-reranker-v2-m3. The other embedders are here so the benchmark can
compare against them.

Input length: an embedder reads up to its own limit (Qwen3-Embedding 32k tokens), so a chunk is embedded whole.
The benchmarks cut every model at 512 tokens (BENCHMARK_MAX_TOKENS) to compare them alike; that setting had been
left in the product, where 53.5% of the owner's chunks were longer and only their start was embedded. The
reranker is used at 512 tokens, the setting of its model card (its training length is not stated): a longer
text is scored window by window and keeps its best window.
"""

import os

import numpy as np

EMBEDDERS = {
    "bge-m3": {"repo": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181", "max_tokens": 8192},
    "multilingual-e5-large": {"repo": "intfloat/multilingual-e5-large",
                              "revision": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
                              "query_prefix": "query: ", "doc_prefix": "passage: ", "max_tokens": 512},
    "qwen3-embedding-0.6b": {"repo": "Qwen/Qwen3-Embedding-0.6B",
                             "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "query_prompt_name": "query",
                             "max_tokens": 32768},
}
RERANKER = {"repo": "BAAI/bge-reranker-v2-m3", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"}
BENCHMARK_MAX_TOKENS = 512   # every model alike in eval/retrieval.py, speed.py, multilingual.py
RERANKER_MAX_TOKENS = 512    # a (query, text) pair, special tokens included
SPECIAL_TOKENS = 4           # <s> query </s></s> text </s> for an XLM-R cross-encoder
WINDOW_OVERLAP = 64          # tokens shared by consecutive windows, so a sentence cut by one is whole in the next
RETOKENIZE_SLACK = 2         # a window decoded then encoded again can gain a token (measured: 513 of 512)
QUERY_MAX_TOKENS = 256       # a longer query is cut for the reranker (its start kept), so a window always has room:
                             # 512 - 4 - 256 = 252 tokens of text, far over the overlap (a query of 444+ tokens once left
                             # 64, the overlap itself: the windows never moved on). BM25 and the embedder read it whole.
MIN_FREE_GIB = 2.0   # less free GPU memory than this (e.g. a local vLLM server holds it): the CPU is used instead


def _device() -> str:
    """cuda when a GPU has room for the model, else cpu; PERESEARCH_DEVICE=cpu|cuda forces a choice."""
    if os.environ.get("PERESEARCH_DEVICE") in ("cpu", "cuda"):     # a forced choice needs no PyTorch (`setup` in CI)
        return os.environ["PERESEARCH_DEVICE"]
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free, _ = torch.cuda.mem_get_info()
    return "cuda" if free / 2**30 >= MIN_FREE_GIB else "cpu"


def signature(name: str, max_tokens: int | None = None) -> str:
    """What makes two sets of vectors comparable: the model, its revision and how much of a text it reads."""
    spec = EMBEDDERS[name]
    return f"{name}@{spec['revision'][:12]}/{max_tokens or spec['max_tokens']}"


def windows(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """[start, end) spans of at most `size` that cover range(n), consecutive spans sharing `overlap`."""
    assert 0 <= overlap < size, f"windows of {size} tokens sharing {overlap} would never move on"
    if n <= size:
        return [(0, n)]
    spans, start = [], 0
    while True:
        spans.append((start, min(start + size, n)))
        if start + size >= n:
            return spans
        start += size - overlap


def _load(cls, repo: str, revision: str, **kw):
    """`cls` (SentenceTransformer, CrossEncoder) from this machine when the pinned revision is here, else from the
    Hub at that revision. Loading by name checked the revision's files online at every start (27 requests to
    huggingface.co each run, end to end, 2026-10-05): what is here is used without asking, and a copy that does not
    load (a download stopped midway) is fetched again."""
    try:
        from huggingface_hub import snapshot_download

        here = snapshot_download(repo, revision=revision, local_files_only=True)
    except Exception:                       # not on this machine (or no huggingface_hub): the Hub
        here = None
    if here:
        try:
            return cls(here, **kw)
        except Exception:
            pass
    return cls(repo, revision=revision, **kw)


class Embedder:
    """Unit-length dense vectors; queries and documents get each model's own prefixes. `max_tokens` (default:
    the model's own limit) is where a longer text is cut."""

    def __init__(self, name: str = "bge-m3", batch_size: int = 32, max_tokens: int | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.name, self.spec, self.batch_size = name, EMBEDDERS[name], batch_size
        self.max_tokens = max_tokens or self.spec["max_tokens"]
        self.signature = signature(name, self.max_tokens)
        device = _device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = _load(SentenceTransformer, self.spec["repo"], self.spec["revision"], device=device,
                           model_kwargs={"torch_dtype": dtype})
        self.model.max_seq_length = self.max_tokens

    def _encode(self, texts: list[str], **kw) -> np.ndarray:
        out = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False, **kw).astype(np.float16)
        assert len(out) == len(texts), f"{len(out)} vectors for {len(texts)} texts"
        assert not texts or bool(np.all(np.abs(np.linalg.norm(out.astype(np.float32), axis=1) - 1) < 1e-2)), \
            "vectors are not unit length: cosine scores would be wrong"
        return out

    def documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.spec.get("doc_prefix", "") + t for t in texts])

    def queries(self, texts: list[str]) -> np.ndarray:
        if "query_prompt_name" in self.spec:
            return self._encode(texts, prompt_name=self.spec["query_prompt_name"])
        return self._encode([self.spec.get("query_prefix", "") + t for t in texts])


class Reranker:
    """Cross-encoder relevance in [0, 1] for (query, text) pairs. With `windowed` (the default), a text longer
    than one pair holds is scored in overlapping windows and gets its best window's score; without it, the text
    is cut, as the benchmarks did."""

    def __init__(self, batch_size: int = 32, windowed: bool = True):
        import torch
        from sentence_transformers import CrossEncoder

        device = _device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.batch_size, self.windowed = batch_size, windowed
        self.model = _load(CrossEncoder, RERANKER["repo"], RERANKER["revision"], max_length=RERANKER_MAX_TOKENS,
                           device=device, model_kwargs={"torch_dtype": dtype})

    def _pairs(self, query: str, texts: list[str]) -> tuple[list[tuple[str, str]], list[int]]:
        """The (query, window) pairs to score and, for each, the index of the text it comes from."""
        tok = self.model.tokenizer
        query_ids = tok.encode(query, add_special_tokens=False, verbose=False)
        if len(query_ids) > QUERY_MAX_TOKENS:
            query = tok.decode(query_ids[:QUERY_MAX_TOKENS])
            query_ids = tok.encode(query, add_special_tokens=False, verbose=False)   # as the model will read it
        room = RERANKER_MAX_TOKENS - SPECIAL_TOKENS - len(query_ids) - RETOKENIZE_SLACK
        assert room > WINDOW_OVERLAP, f"{room} tokens of text a window, overlap {WINDOW_OVERLAP}"
        pairs, owner = [], []
        for i, text in enumerate(texts):
            ids = tok.encode(text, add_special_tokens=False, verbose=False)   # longer than 512 on purpose
            spans = windows(len(ids), room, WINDOW_OVERLAP)
            for a, b in spans:
                pairs.append((query, text if len(spans) == 1 else tok.decode(ids[a:b])))
                owner.append(i)
        return pairs, owner

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        if not self.windowed:
            out = self.model.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)
            return np.asarray(out, dtype=np.float32)
        pairs, owner = self._pairs(query, texts)
        out = np.asarray(self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False),
                         dtype=np.float32)
        best = np.full(len(texts), -np.inf, dtype=np.float32)
        np.maximum.at(best, np.asarray(owner), out)
        assert bool(np.all(np.isfinite(best))), "a text got no window to score"
        return best


def free_gpu() -> None:
    """Return cached GPU memory after the caller has dropped its model (6 GB holds one embedder plus the
    reranker at most)."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_memory() -> str:
    """Device memory in use by everything on the GPU, and the part PyTorch holds (the rest is CUDA contexts and
    other libraries: a JAX that grabbed the GPU showed up here as 11 GiB)."""
    import torch

    if not torch.cuda.is_available():
        return "no GPU"
    free, total = torch.cuda.mem_get_info()
    return (f"GPU used {(total - free) / 2**30:.2f} of {total / 2**30:.2f} GiB, PyTorch "
            f"{torch.cuda.memory_allocated() / 2**30:.2f} allocated / {torch.cuda.memory_reserved() / 2**30:.2f} reserved")
