"""Contract of the user's embedding and rerank endpoints (peresearch.zetokrag.embed, step 2 of 2026-10-08): no model
runs inside peresearch; texts leave redacted, one audit line a request; vectors come back unit length, in the order
asked, whatever order the server answers in; a reranker's scores come back in [0, 1], in the order asked; a server
that answers wrongly or fails is said by name, a busy one is waited for."""
import json
import math

import httpx
import numpy as np
import pytest
from hypothesis import given, settings as hsettings
from hypothesis import strategies as st

from peresearch import guard
from peresearch.zetokrag import embed
from tests.test_guard import CANARIES

SECRET = CANARIES["github-token"]


def server(handler):
    """An httpx client whose requests `handler(request) -> (status, json body, headers)` answers; the requests kept."""
    seen = []

    def answer(request):
        seen.append(request)
        status, body, headers = handler(request)          # NaN written as JSON allows it (a broken server's)
        return httpx.Response(status, content=json.dumps(body).encode(), headers={"content-type": "application/json",
                                                                                  **(headers or {})})
    client = httpx.Client(transport=httpx.MockTransport(answer), base_url="http://e.test/v1")
    client.seen = seen
    return client


def vectors_for(request, dim=4, scale=3.0, order=None):
    texts = json.loads(request.content)["input"]
    data = [{"index": i, "embedding": [scale * (1 + i)] + [float(len(t))] + [1.0] * (dim - 2)} for i, t in enumerate(texts)]
    return 200, {"data": data if order is None else [data[i] for i in order(len(data))]}, None


def test_documents_are_embedded_through_the_users_endpoint_redacted_and_in_order(home):
    client = server(lambda r: vectors_for(r, order=lambda n: list(reversed(range(n)))))   # answered back to front
    e = embed.Embedder("http://e.test/v1", "bge-m3", key="k-123", client=client, batch=2)
    texts = ["first", f"key {SECRET} here", "third"]
    out = e.documents(texts)
    assert out.dtype == np.float16 and out.shape == (3, 4)
    assert np.allclose(np.linalg.norm(out.astype(np.float32), axis=1), 1, atol=1e-2)            # unit length
    lengths = [len(guard.redact(t)[0]) for t in texts]
    want = np.array([[3.0 * (1 + i % 2), n, 1, 1] for i, n in enumerate(lengths)], dtype=np.float32)
    assert np.allclose(out.astype(np.float32), want / np.linalg.norm(want, axis=1, keepdims=True), atol=1e-2)
    bodies = [json.loads(r.content) for r in client.seen]
    assert [b["input"] for b in bodies] == [["first", guard.redact(texts[1])[0]], ["third"]]    # batches of 2
    assert all(b["model"] == "bge-m3" for b in bodies) and SECRET not in "".join(r.content.decode() for r in client.seen)
    assert all(r.url.path == "/v1/embeddings" and r.headers["authorization"] == "Bearer k-123" for r in client.seen)
    audits = [json.loads(x) for x in (home / "audit.jsonl").read_text().splitlines()]
    assert [(a["channel"], a["redacted"]) for a in audits] == [("embeddings", guard.redact(texts[1])[1]),
                                                               ("embeddings", 0)]                 # one a request
    assert e.documents([]).shape == (0, 0) and len(client.seen) == 2                            # nothing to send


@pytest.mark.parametrize("model, query, document", [
    ("qwen3-embedding:0.6b", embed.QWEN3_QUERY, ""),             # Qwen3-Embedding's card: an instruction on queries
    ("Qwen/Qwen3-Embedding-0.6B", embed.QWEN3_QUERY, ""),
    ("embeddinggemma", "task: search result | query: ", "title: none | text: "),   # EmbeddingGemma's card
    ("multilingual-e5-large", "query: ", "passage: "),           # E5's card: both marked
    ("bge-m3", "", ""),                                          # BGE-M3's card: none
    ("text-embedding-3-small", "", ""),                          # unknown: as given
])
def test_queries_and_documents_get_the_prefixes_the_model_card_asks_for(model, query, document):
    assert embed.QWEN3_QUERY == ("Instruct: Given a web search query, retrieve relevant passages that answer the "
                                 "query\nQuery:")                       # the card's own words, as M1 measured them
    client = server(vectors_for)
    e = embed.Embedder("http://e.test/v1", model, client=client)
    e.queries(["what is BPE"])
    e.documents(["BPE merges pairs"])
    assert [json.loads(r.content)["input"] for r in client.seen] == [[query + "what is BPE"], [document + "BPE merges pairs"]]


@pytest.mark.parametrize("answer, said", [
    (lambda r: (200, {"data": []}, None), "1 texts sent, 0 vectors back"),
    (lambda r: (200, {"data": [{"index": 0, "embedding": [1, 2]}, {"index": 1, "embedding": [1, 2, 3]}]}, None),
     "vectors of different lengths"),
    (lambda r: (200, {"data": [{"index": 0, "embedding": [0, 0, 0]}]}, None), "a zero or non-finite vector"),
    (lambda r: (200, {"data": [{"index": 0, "embedding": [float("nan"), 1, 1]}]}, None), "a zero or non-finite vector"),
    (lambda r: (200, {"error": "no"}, None), "no data in the answer"),
    (lambda r: (404, {"error": {"message": "model 'x' not found"}}, None), "404"),
])
def test_a_wrong_answer_is_said_with_the_endpoint_never_used(answer, said):
    e = embed.Embedder("http://e.test/v1", "x", client=server(answer), retries=1)
    with pytest.raises(embed.EndpointError) as err:
        e.documents(["t"] if said != "vectors of different lengths" else ["t", "u"])
    assert said in str(err.value) and "http://e.test/v1" in str(err.value)


def test_a_text_too_long_for_the_model_is_embedded_by_its_head_and_counted():
    """An endpoint whose model takes less than a chunk (a 512-token model; Ollama's context) refuses a request: the
    batch is split to find the text, and that text is embedded by its head, halved until the endpoint takes it (M1's
    way, runs/gold/m1-…); each halving counted. A text refused even at MIN_TEXT characters is said."""
    def picky(request):
        texts = json.loads(request.content)["input"]
        if any(len(t) > 100 for t in texts):
            return 400, {"error": {"message": "input length exceeds the context length"}}, None
        return vectors_for(request)
    client = server(picky)
    e = embed.Embedder("http://e.test/v1", "m", client=client)
    out = e.documents(["short", "x" * 350, "also short"])
    assert out.shape == (3, 4) and e.halved == 2                                 # 350 → 175 → 87
    lengths = [round(float(v[1]) / float(v[3])) for v in out.astype(np.float32)]  # each text's own vector, in order
    assert lengths == [5, 87, 10]
    sent = [json.loads(r.content)["input"] for r in client.seen]
    assert ["x" * 87] in sent and sent[-1] == ["also short"] or ["also short"] in sent
    never = embed.Embedder("http://e.test/v1", "m", client=server(lambda r: (400, {"error": "too long"}, None)))
    with pytest.raises(embed.EndpointError, match="400"):
        never.documents(["y" * 300])


def test_a_refusal_is_said_at_once_with_the_servers_words():
    """A 4xx other than 408/429 (a wrong model name, a bad key) is not tried again: one request, its answer said."""
    client = server(lambda r: (404, {"error": {"message": "model 'x' not found"}}, None))
    with pytest.raises(embed.EndpointError, match="404 .*model 'x' not found"):
        embed.Embedder("http://e.test/v1", "x", client=client, retries=3).documents(["t", "u", "v"])
    assert len(client.seen) == 1                                       # not split either: no text is too long


def test_a_busy_or_failing_server_is_waited_for_then_said(monkeypatch):
    slept = []
    monkeypatch.setattr(embed.time, "sleep", slept.append)
    replies = iter([(503, {}, None), (429, {}, {"retry-after": "7"}), (200, None, None)])

    def answer(r):
        status, body, headers = next(replies)
        return vectors_for(r) if status == 200 else (status, body, headers)
    e = embed.Embedder("http://e.test/v1", "m", client=server(answer), retries=3, wait=2.0)
    assert e.documents(["t"]).shape == (1, 4) and slept == [2.0, 7.0]           # its own wait, then the server's
    always = embed.Embedder("http://e.test/v1", "m", client=server(lambda r: (500, {"error": "down"}, None)), retries=2,
                            wait=1.0)
    with pytest.raises(embed.EndpointError, match="500 after 2 tries"):
        always.documents(["t"])
    far = embed.Embedder("http://e.test/v1", "m", client=server(lambda r: (429, {}, {"retry-after": "600"})), retries=2,
                         wait=1.0)
    slept.clear()
    with pytest.raises(embed.EndpointError):
        far.documents(["t"])
    assert slept == []                                          # a wait past 60 s is not taken: said at once


def test_the_signature_names_the_model_and_where_it_runs():
    a = embed.Embedder("http://e.test/v1", "bge-m3", client=server(vectors_for))
    assert a.signature == "bge-m3 at http://e.test/v1"
    assert embed.Embedder("http://other/v1/", "bge-m3", client=server(vectors_for)).signature == "bge-m3 at http://other/v1"
    assert embed.Embedder("http://e.test/v1", "", client=server(vectors_for)).signature == "the server's model at http://e.test/v1"


def ranked(request, scores, order=None):
    body = json.loads(request.content)
    results = [{"index": i, "relevance_score": s} for i, s in enumerate(scores[:len(body["documents"])])]
    return 200, {"results": results if order is None else [results[i] for i in order]}, None


@given(st.lists(st.floats(-20, 20, allow_nan=False), min_size=1, max_size=8), st.booleans())
@hsettings(max_examples=60, deadline=None)
def test_rerank_scores_come_back_in_order_and_in_zero_to_one(scores, shuffled):
    """Cohere's and Jina's shape (vLLM, llama.cpp, TEI serve it): results sorted by score, each with its index. A score
    outside [0, 1] (a raw logit) puts every score of that answer through the logistic function: the order is kept."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i]) if shuffled else None
    client = server(lambda r: ranked(r, scores, order))
    r = embed.Reranker("http://e.test/v1", "bge-reranker-v2-m3", client=client)
    got = r.scores("q", [f"t{i}" for i in range(len(scores))])
    raw = np.array(scores, dtype=np.float64)
    want = raw if np.all((raw >= 0) & (raw <= 1)) else 1 / (1 + np.exp(-raw))
    assert got.dtype == np.float32 and np.allclose(got, want, atol=1e-6) and np.all((got >= 0) & (got <= 1))
    assert client.seen[0].url.path == "/v1/rerank" and json.loads(client.seen[0].content)["model"] == "bge-reranker-v2-m3"


def test_rerank_texts_are_redacted_and_a_missing_score_is_said(home):
    client = server(lambda r: ranked(r, [0.9, 0.1]))
    r = embed.Reranker("http://e.test/v1", "m", client=client)
    r.scores(f"q {SECRET}", ["a", f"b {SECRET}"])
    body = json.loads(client.seen[0].content)
    assert SECRET not in json.dumps(body) and body["query"] == guard.redact(f"q {SECRET}")[0]
    short = embed.Reranker("http://e.test/v1", "m", client=server(lambda r: (200, {"results": [{"index": 0,
                                                                                               "relevance_score": 1}]}, None)))
    with pytest.raises(embed.EndpointError, match="2 texts sent, 1 scores back"):
        short.scores("q", ["a", "b"])
    assert r.scores("q", []).shape == (0,)


def test_the_endpoints_come_from_the_settings(home, monkeypatch):
    for name in ("PERESEARCH_EMBED_URL", "PERESEARCH_EMBED_MODEL", "PERESEARCH_EMBED_KEY", "PERESEARCH_RERANK_URL",
                 "PERESEARCH_RERANK_MODEL", "PERESEARCH_RERANK_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert embed.from_settings() == (None, None)                      # none by default: the search is lexical
    monkeypatch.setenv("PERESEARCH_EMBED_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("PERESEARCH_EMBED_MODEL", "bge-m3")
    monkeypatch.setenv("PERESEARCH_RERANK_URL", "http://localhost:8081/v1")
    e, r = embed.from_settings()
    assert e.signature == "bge-m3 at http://localhost:11434/v1" and r.model is None and e.local and r.local
    monkeypatch.setenv("PERESEARCH_EMBED_URL", "https://api.example.com/v1")
    assert not embed.from_settings()[0].local                          # chunks would leave this machine


def test_math_of_the_logistic_is_stable_at_the_ends():
    assert np.allclose(embed._unit_interval(np.array([-800.0, 0.0, 800.0])), [0.0, 0.5, 1.0])
    assert not math.isnan(float(embed._unit_interval(np.array([1e308]))[0]))


def test_the_endpoint_check_says_what_runs_where_and_what_is_missing(monkeypatch):
    """`peresearch check` and /setup: each endpoint answers or is said failing; one off this machine says the user's
    passages go to that host; none set says the search is by words alone, with what that costs (M1)."""
    e = embed.Embedder("https://api.example.com/v1", "bge-m3", client=server(vectors_for))
    r = embed.Reranker("http://localhost:8081/v1", None, client=server(lambda q: ranked(q, [0.9])))
    lines = embed.check(e, r)
    assert lines[0].startswith("embeddings: bge-m3 at https://api.example.com/v1 · 4 dimensions · ")
    assert lines[0].endswith("· your passages are sent to api.example.com to be embedded")
    assert lines[1].startswith("rerank: the server's model at http://localhost:8081/v1 · answered in ")
    assert embed.check(None, None) == [
        "embeddings: none set — the search goes by words alone (BM25): a question put in other words or another "
        "language finds less (gold dev set, nDCG@10: 0.336 by words, 0.546 with bge-m3 through Ollama). Set an "
        "/embeddings endpoint in /setup (Ollama, llama.cpp, vLLM, LM Studio or a provider)",
        "rerank: none set — results keep the search's order, without a relevance score (gold dev set, nDCG@10: "
        "0.546 → 0.648 with bge-reranker-v2-m3 through vLLM). Optional: a /rerank endpoint in /setup (vLLM, llama.cpp, "
        "TEI, Jina, Cohere)"]
    down = embed.Embedder("http://localhost:9/v1", "m", client=server(lambda q: (500, {}, None)), retries=1)
    assert embed.check(down, None)[0] == "embeddings: m at http://localhost:9/v1 · FAILED: http://localhost:9/v1/embeddings: 500 after 1 tries"
