"""The local encoder models, pinned to exact Hugging Face revisions (a moved revision is a different model).

ZetokRAG uses Qwen3-Embedding-0.6B (dense) and bge-reranker-v2-m3, chosen on the benchmark; the other
embedders are here so the benchmark can compare against them. Texts are cut at 512 tokens for every model alike.
"""

import numpy as np

EMBEDDERS = {
    "bge-m3": {"repo": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181"},
    "multilingual-e5-large": {"repo": "intfloat/multilingual-e5-large",
                              "revision": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
                              "query_prefix": "query: ", "doc_prefix": "passage: "},
    "qwen3-embedding-0.6b": {"repo": "Qwen/Qwen3-Embedding-0.6B",
                             "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "query_prompt_name": "query"},
}
RERANKER = {"repo": "BAAI/bge-reranker-v2-m3", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"}
MAX_TOKENS = 512


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    """Unit-length dense vectors; queries and documents get each model's own prefixes."""

    def __init__(self, name: str = "bge-m3", batch_size: int = 32):
        import torch
        from sentence_transformers import SentenceTransformer

        self.name, self.spec, self.batch_size = name, EMBEDDERS[name], batch_size
        dtype = torch.float16 if _device() == "cuda" else torch.float32
        self.model = SentenceTransformer(self.spec["repo"], revision=self.spec["revision"], device=_device(),
                                         model_kwargs={"torch_dtype": dtype})
        self.model.max_seq_length = MAX_TOKENS

    def _encode(self, texts: list[str], **kw) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False, **kw).astype(np.float16)

    def documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.spec.get("doc_prefix", "") + t for t in texts])

    def queries(self, texts: list[str]) -> np.ndarray:
        if "query_prompt_name" in self.spec:
            return self._encode(texts, prompt_name=self.spec["query_prompt_name"])
        return self._encode([self.spec.get("query_prefix", "") + t for t in texts])


class Reranker:
    """Cross-encoder relevance in [0, 1] for (query, text) pairs."""

    def __init__(self, batch_size: int = 32):
        import torch
        from sentence_transformers import CrossEncoder

        dtype = torch.float16 if _device() == "cuda" else torch.float32
        self.batch_size = batch_size
        self.model = CrossEncoder(RERANKER["repo"], revision=RERANKER["revision"], max_length=MAX_TOKENS,
                                  device=_device(), model_kwargs={"torch_dtype": dtype})

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        out = self.model.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)


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
