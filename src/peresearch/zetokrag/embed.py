"""The user's embedding and rerank endpoints: no model runs inside peresearch (owner, 2026-10-07).

Embeddings go through POST <url>/embeddings {model, input} — OpenAI's shape, which Ollama, llama.cpp, vLLM, LM Studio and
the providers serve —, a rerank through POST <url>/rerank {model, query, documents} — Cohere's and Jina's shape, which
vLLM, llama.cpp, Text Embeddings Inference, Jina and Cohere serve. Neither is set by default: the search is then
lexical (BM25, and grep beside it); M1 (runs/gold/m1-auxiliary-models-2026-10-07.json) measured what each adds.

Every text leaves redacted (guard), one audit line a request. Vectors are made unit length here, as an endpoint may
not; a reranker's scores outside [0, 1] (raw logits) go through the logistic function, which keeps their order.
"""

import time
from urllib.parse import urlparse

import numpy as np

from peresearch import guard, settings

# What each family's model card asks to put before a query or a document (an endpoint applies no prompt of its own:
# Ollama's /v1/embeddings, measured in M1); a model not listed is sent the text as it is.
QWEN3_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
PREFIXES = [("qwen3-embedding", QWEN3_QUERY, ""), ("embeddinggemma", "task: search result | query: ",
                                                    "title: none | text: "), ("e5", "query: ", "passage: ")]
WAIT_CAP = 60.0          # a server asking for a longer wait (Retry-After) is not waited for: said instead
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
TOO_LONG_STATUS = {400, 413, 422, 500}   # what a server answers a text longer than its model takes (Ollama: 500)
MIN_TEXT = 64            # a text refused even this short is not halved further: the refusal is said


class EndpointError(RuntimeError):
    """An endpoint that failed or answered wrongly, said with where it is; `status` the HTTP status, when there was
    one."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _unit_interval(x: np.ndarray) -> np.ndarray:
    """Scores in [0, 1]: as they are when all already lie there, else every one through the logistic function."""
    x = np.asarray(x, dtype=np.float64)
    if np.all((x >= 0) & (x <= 1)):
        return x
    return np.where(x >= 0, 1 / (1 + np.exp(-np.clip(x, 0, 700))), np.exp(np.clip(x, -700, 0)) /
                    (1 + np.exp(np.clip(x, -700, 0))))


class _Endpoint:
    channel = ""

    def __init__(self, url: str, model: str | None, key: str | None = None, client=None, retries: int = 3,
                 wait: float = 2.0):
        import httpx

        self.url, self.model = url.rstrip("/"), model or None
        self.client = client or httpx.Client(base_url=self.url, timeout=300.0,
                                             headers={"Authorization": f"Bearer {key}"} if key else {})
        if client is not None and key:
            self.client.headers["Authorization"] = f"Bearer {key}"
        self.retries, self.wait = retries, wait
        self.local = urlparse(self.url).hostname in ("localhost", "127.0.0.1", "::1")

    def _post(self, path: str, body: dict, redacted: int, chars: int) -> dict:
        """The answer's JSON; a busy or failing server tried `retries` times (its Retry-After kept up to WAIT_CAP)."""
        import httpx

        guard.audit("outbound", channel=self.channel, chars=chars, redacted=redacted)
        last, status = "", None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.client.post(path, json=body)
            except httpx.HTTPError as e:
                last, wait, status = f"{type(e).__name__}", self.wait, None
            else:
                if r.status_code < 400:
                    try:
                        return r.json()
                    except ValueError:
                        raise EndpointError(f"{self.url}{path}: the answer is not JSON") from None
                if r.status_code not in RETRY_STATUS:
                    raise EndpointError(f"{self.url}{path}: {r.status_code} {r.text[:300]}", r.status_code)
                last, wait, status = str(r.status_code), self.wait, r.status_code
                asked = r.headers.get("retry-after", "")
                if asked.replace(".", "", 1).isdigit():
                    wait = float(asked)
            if attempt == self.retries or wait > WAIT_CAP:
                break
            time.sleep(wait)
        raise EndpointError(f"{self.url}{path}: {last} after {attempt} tries", status)

    def _clean(self, texts: list[str]) -> tuple[list[str], int]:
        out, n = [], 0
        for t in texts:
            clean, k = guard.redact(t)
            assert not guard.find_secrets(clean), f"{self.channel}: a secret is still there after redaction"
            out.append(clean)
            n += k
        return out, n


class Embedder(_Endpoint):
    """Unit-length vectors (float16) from the user's /embeddings endpoint, `batch` texts a request."""
    channel = "embeddings"

    def __init__(self, url: str, model: str | None, key: str | None = None, client=None, batch: int = 32,
                 retries: int = 3, wait: float = 2.0):
        super().__init__(url, model, key, client, retries, wait)
        self.batch = batch
        self.halved = 0                                # texts embedded by their head: the model took no more
        name = (model or "").lower()
        self.query_prefix, self.doc_prefix = next(((q, d) for family, q, d in PREFIXES if family in name), ("", ""))
        self.signature = f"{model or "the server's model"} at {self.url}"

    def documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.doc_prefix + t for t in texts])

    def queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.query_prefix + t for t in texts])

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float16)
        parts = []
        for i in range(0, len(texts), self.batch):
            sent, redacted = self._clean(texts[i:i + self.batch])
            parts.append(np.asarray(self._vectors(sent, redacted), dtype=np.float64))
        if len({p.shape[1] for p in parts}) != 1:
            raise EndpointError(f"{self.url}/embeddings: vectors of different lengths")
        vecs = np.concatenate(parts)
        norms = np.linalg.norm(vecs, axis=1)
        if not np.all(np.isfinite(vecs)) or np.any(norms == 0):
            raise EndpointError(f"{self.url}/embeddings: a zero or non-finite vector")
        return (vecs / norms[:, None]).astype(np.float16)


    def _vectors(self, sent: list[str], redacted: int = 0) -> list[list[float]]:
        """The vectors of redacted texts, in order. Refused (a status a too long text gets), the batch is split to find
        the text, and a text refused alone is embedded by its head, halved until taken (counted in `halved`)."""
        try:
            got = self._post("/embeddings", {"input": sent, **({"model": self.model} if self.model else {})},
                             redacted, sum(map(len, sent)))
        except EndpointError as e:
            if e.status not in TOO_LONG_STATUS:
                raise
            if len(sent) > 1:
                mid = len(sent) // 2
                return self._vectors(sent[:mid]) + self._vectors(sent[mid:])
            if len(sent[0]) // 2 < MIN_TEXT:
                raise
            self.halved += 1
            return self._vectors([sent[0][:len(sent[0]) // 2]])
        data = got.get("data") if isinstance(got, dict) else None
        if not isinstance(data, list):
            raise EndpointError(f"{self.url}/embeddings: no data in the answer")
        if len(data) != len(sent):
            raise EndpointError(f"{self.url}/embeddings: {len(sent)} texts sent, {len(data)} vectors back")
        rows = [d.get("embedding") for d in sorted(data, key=lambda d: d.get("index", 0))]
        if len({len(r) if isinstance(r, list) else -1 for r in rows}) != 1 or not isinstance(rows[0], list):
            raise EndpointError(f"{self.url}/embeddings: vectors of different lengths")
        return rows


class Reranker(_Endpoint):
    """Relevance in [0, 1] of each text to a query, from the user's /rerank endpoint, in the order the texts came."""
    channel = "rerank"

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        (q, *docs), redacted = self._clean([query, *texts])
        body = {"query": q, "documents": docs, **({"model": self.model} if self.model else {})}
        got = self._post("/rerank", body, redacted, len(q) + sum(map(len, docs)))
        results = got.get("results") if isinstance(got, dict) else None
        if not isinstance(results, list):
            raise EndpointError(f"{self.url}/rerank: no results in the answer")
        by = {r.get("index"): r.get("relevance_score", r.get("score")) for r in results if isinstance(r, dict)}
        if len(by) != len(texts) or set(by) != set(range(len(texts))):
            raise EndpointError(f"{self.url}/rerank: {len(texts)} texts sent, {len(by)} scores back")
        raw = np.array([by[i] for i in range(len(texts))], dtype=np.float64)
        if not np.all(np.isfinite(raw)):
            raise EndpointError(f"{self.url}/rerank: a non-finite score")
        return _unit_interval(raw).astype(np.float32)


def from_settings() -> tuple["Embedder | None", "Reranker | None"]:
    """The endpoints the user set (PERESEARCH_EMBED_URL/_MODEL/_KEY, PERESEARCH_RERANK_URL/_MODEL/_KEY): each None
    when its URL is not set."""
    def one(cls, prefix):
        url = settings.get(f"{prefix}_URL")
        return cls(url, settings.get(f"{prefix}_MODEL") or None, settings.get(f"{prefix}_KEY") or None) if url else None
    return one(Embedder, "PERESEARCH_EMBED"), one(Reranker, "PERESEARCH_RERANK")



def check(embedder: "Embedder | None", reranker: "Reranker | None") -> list[str]:
    """One line for each endpoint, as `peresearch check` and /setup show it: what it is, that it answers (dimensions,
    time), and where the user's passages go when it is not on this machine; none set, what the search does without it
    and what that costs (the gold dev set through these endpoints: runs/serve/endpoints-2026-10-08.json)."""
    from urllib.parse import urlparse

    lines = []
    if embedder is None:
        lines.append("embeddings: none set — the search goes by words alone (BM25): a question put in other words or "
                     "another language finds less (gold dev set, nDCG@10: 0.336 by words, 0.546 with bge-m3 through "
                     "Ollama). Set an /embeddings endpoint in /setup (Ollama, llama.cpp, vLLM, LM Studio or a provider)")
    else:
        head = f"embeddings: {embedder.signature}"
        try:
            t0 = time.monotonic()
            dims = embedder.documents(["a test passage"]).shape[1]
            lines.append(f"{head} · {dims} dimensions · answered in {time.monotonic() - t0:.1f} s" + (
                "" if embedder.local else
                f" · your passages are sent to {urlparse(embedder.url).hostname} to be embedded"))
        except EndpointError as e:
            lines.append(f"{head} · FAILED: {e}")
    if reranker is None:
        lines.append("rerank: none set — results keep the search's order, without a relevance score (gold dev set, "
                     "nDCG@10: 0.546 → 0.648 with bge-reranker-v2-m3 through vLLM). Optional: a /rerank endpoint in "
                     "/setup (vLLM, llama.cpp, TEI, Jina, Cohere)")
    else:
        head = f"rerank: {reranker.model or "the server's model"} at {reranker.url}"
        try:
            t0 = time.monotonic()
            reranker.scores("a test question", ["a test passage"])
            lines.append(f"{head} · answered in {time.monotonic() - t0:.1f} s" + (
                "" if reranker.local else
                f" · your passages are sent to {urlparse(reranker.url).hostname} to be ranked"))
        except EndpointError as e:
            lines.append(f"{head} · FAILED: {e}")
    return lines
