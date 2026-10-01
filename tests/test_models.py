"""The encoder models' input lengths, with stand-ins for the Hugging Face classes (no download, no GPU)."""
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

pytest.importorskip("torch")
sentence_transformers = pytest.importorskip("sentence_transformers")

from peresearch.zetokrag import models  # noqa: E402


class FakeSentenceTransformer:
    """Records how it is called; returns float32 rows of the text lengths."""

    def __init__(self, *args, device=None, **kwargs):
        self.max_seq_length, self.calls, self.device = None, [], device

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        import numpy as np
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


class WordTokenizer:
    """One token per whitespace-separated word."""

    def __init__(self):
        self.vocab, self.words = {}, []

    def encode(self, text, add_special_tokens=False, **kwargs):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.words)
                self.words.append(w)
            ids.append(self.vocab[w])
        return ids

    def decode(self, ids):
        return " ".join(self.words[i] for i in ids)


class FakeCrossEncoder:
    """Scores 1 when the text holds "needle"; fails on a pair longer than the model reads, as truncation would
    silently drop the end of it."""

    def __init__(self, *args, max_length=None, device=None, **kwargs):
        self.max_length, self.tokenizer, self.seen, self.device = max_length, WordTokenizer(), [], device

    def predict(self, pairs, **kwargs):
        out = []
        for q, t in pairs:
            assert len(q.split()) + len(t.split()) + models.SPECIAL_TOKENS <= self.max_length, "pair too long"
            self.seen.append(t)
            out.append(1.0 if "needle" in t.split() else 0.0)
        return out


def test_the_embedder_reads_whole_chunks_unless_told_otherwise(monkeypatch):
    """The 512-token cut was a benchmark setting left in the product: Qwen3-Embedding reads 32k tokens, and 53.5%
    of the owner's chunks were longer than 512, so only their start was embedded."""
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    e = models.Embedder("qwen3-embedding-0.6b")
    assert e.model.max_seq_length == 32768 and e.signature.endswith("/32768")
    bench = models.Embedder("qwen3-embedding-0.6b", max_tokens=models.BENCHMARK_MAX_TOKENS)
    assert bench.model.max_seq_length == 512 and bench.signature != e.signature
    assert models.Embedder("multilingual-e5-large").model.max_seq_length == 512     # that model's own limit


def test_the_reranker_reads_a_long_text_in_windows(monkeypatch):
    """bge-reranker-v2-m3 is used at 512 tokens (its card's setting); a longer chunk is scored window by window
    and keeps its best window, instead of being cut after its first 512 tokens."""
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    r = models.Reranker()
    long_text = " ".join(f"w{i}" for i in range(1200)) + " needle"
    scores = r.scores("where is the needle", [long_text, "a needle here", "nothing"])
    assert scores.tolist() == [1.0, 1.0, 0.0]
    covered = set()
    for t in r.model.seen:
        covered.update(t.split())
    assert {f"w{i}" for i in range(1200)} <= covered                     # every word of the long text was read


@given(st.integers(0, 3000), st.integers(2, 600).flatmap(lambda size: st.tuples(st.just(size), st.integers(0, size - 1))))
@example(10, (4, 1))
@example(0, (4, 1))
def test_windows_cover_every_token_within_the_size_and_overlap_as_set(n, size_overlap):
    size, overlap = size_overlap
    spans = models.windows(n, size, overlap)
    assert spans[0][0] == 0 and spans[-1][1] == n
    assert all(0 < b - a <= size or n == 0 for a, b in spans)
    assert all(b1 - a2 == overlap for (_, b1), (a2, _) in zip(spans, spans[1:]))     # consecutive windows share it
    assert len(spans) == 1 or n > size


def test_the_models_use_the_cpu_when_the_gpu_is_busy(monkeypatch):
    import torch

    monkeypatch.delenv("PERESEARCH_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1 * 2**30, 6 * 2**30))   # a vLLM server holds 5 GB
    assert models._device() == "cpu"
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (5 * 2**30, 6 * 2**30))
    assert models._device() == "cuda"
    monkeypatch.setenv("PERESEARCH_DEVICE", "cpu")
    assert models._device() == "cpu"


def test_queries_and_documents_get_each_models_own_prompt(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(models, "_device", lambda: "cpu")                       # e.g. the GPU is busy
    assert models.Embedder("bge-m3").model.device == "cpu" and models.Reranker().model.device == "cpu"
    qwen, e5 = models.Embedder("qwen3-embedding-0.6b", batch_size=7), models.Embedder("multilingual-e5-large")
    d = qwen.documents(["a note"])
    assert d.dtype.name == "float16" and qwen.model.calls[-1] == (["a note"], {
        "batch_size": 7, "normalize_embeddings": True, "convert_to_numpy": True, "show_progress_bar": False})
    qwen.queries(["a question"])
    assert qwen.model.calls[-1][0] == ["a question"] and qwen.model.calls[-1][1]["prompt_name"] == "query"
    e5.documents(["a note"])
    e5.queries(["a question"])
    assert [c[0] for c in e5.model.calls] == [["passage: a note"], ["query: a question"]]


def test_without_windows_the_reranker_scores_the_texts_as_given(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    r = models.Reranker(windowed=False)
    assert r.model.max_length == models.RERANKER_MAX_TOKENS
    assert r.scores("q", ["a needle", "none"]).tolist() == [1.0, 0.0] and r.model.seen == ["a needle", "none"]
    assert models.Reranker().scores("q", []).tolist() == []
    short = models.Reranker()
    short.scores("q", ["a  needle   spaced"])
    assert short.model.seen == ["a  needle   spaced"]                       # a text that fits is passed untouched
