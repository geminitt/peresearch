"""The encoder models' input lengths, with stand-ins for the Hugging Face classes (no download, no GPU)."""
import pytest

pytest.importorskip("torch")
st = pytest.importorskip("sentence_transformers")

from peresearch.zetokrag import models  # noqa: E402


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.max_seq_length = None


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

    def __init__(self, *args, max_length=None, **kwargs):
        self.max_length, self.tokenizer, self.seen = max_length, WordTokenizer(), []

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
    monkeypatch.setattr(st, "SentenceTransformer", FakeSentenceTransformer)
    e = models.Embedder("qwen3-embedding-0.6b")
    assert e.model.max_seq_length == 32768 and e.signature.endswith("/32768")
    bench = models.Embedder("qwen3-embedding-0.6b", max_tokens=models.BENCHMARK_MAX_TOKENS)
    assert bench.model.max_seq_length == 512 and bench.signature != e.signature
    assert models.Embedder("multilingual-e5-large").model.max_seq_length == 512     # that model's own limit


def test_the_reranker_reads_a_long_text_in_windows(monkeypatch):
    """bge-reranker-v2-m3 is used at 512 tokens (its card's setting); a longer chunk is scored window by window
    and keeps its best window, instead of being cut after its first 512 tokens."""
    monkeypatch.setattr(st, "CrossEncoder", FakeCrossEncoder)
    r = models.Reranker()
    long_text = " ".join(f"w{i}" for i in range(1200)) + " needle"
    scores = r.scores("where is the needle", [long_text, "a needle here", "nothing"])
    assert scores.tolist() == [1.0, 1.0, 0.0]
    covered = set()
    for t in r.model.seen:
        covered.update(t.split())
    assert {f"w{i}" for i in range(1200)} <= covered                     # every word of the long text was read


def test_windows_cover_the_text_with_an_overlap():
    assert models.windows(10, 4, 1) == [(0, 4), (3, 7), (6, 10)]
    assert models.windows(3, 4, 1) == [(0, 3)]
    assert models.windows(0, 4, 1) == [(0, 0)]
