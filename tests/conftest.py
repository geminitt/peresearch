"""Small stand-ins for the GPU models, so every test runs on CPU (as in CI)."""
import hashlib

import numpy as np
import pytest

from peresearch.zetokrag.core import tokenize


class FakeEmbedder:
    """Hashed bag of (diacritic-folded) words: similar wording gives similar vectors."""

    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in tokenize(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).astype(np.float16)

    def documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float16)

    def queries(self, texts):
        return self.documents(texts)


class FakeReranker:
    """Share of query words found in the text, in [0, 1]."""

    def scores(self, query, texts):
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(t))) / max(len(q), 1) for t in texts], dtype=np.float32)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("PERESEARCH_HOME", str(h))
    return h
