"""Contract of peresearch's MCP client (peresearch/mcp.py, step 3 of 2026-10-08): a server started as a command
(stdio) or reached at a URL (Streamable HTTP) is initialized once, lists its tools and runs a call; whatever it sends
meanwhile (a notification, a ping of its own, logs on stderr) neither blocks nor confuses the answer; a server that
crashes, says nothing, or refuses is said by name; a closed client leaves no process behind."""
import json
import sys
import time
from pathlib import Path

import httpx
import pytest

from peresearch import mcp

FAKE = [sys.executable, str(Path(__file__).parent / "data" / "fake_mcp_server.py")]


def test_a_stdio_server_lists_its_tools_and_runs_a_call_answering_its_ping():
    with mcp.Server(FAKE) as s:
        tools = s.tools()
        assert [t["name"] for t in tools] == ["web_search", "fetch"]
        assert tools[0]["inputSchema"]["properties"]["query"] == {"type": "string"}
        text = s.call("web_search", {"query": "bpe", "max_results": 3})
        assert text == "\n".join(f"{i}. Result {i} for bpe\nhttps://example.org/{i}" for i in (1, 2, 3))   # text only
        assert s.call("web_search", {"query": "two"}).count("https://") == 2                          # again, alive
        assert s.info["serverInfo"]["name"] == "fake"
    with mcp.Server(FAKE + ["stale"]) as s:                          # an answer to another request is not this one
        assert s.call("web_search", {"query": "q"}).startswith("1. Result 1 for q")
    s = mcp.Server(FAKE)
    s.tools()
    s.close()
    assert s.proc.returncode == 0                                    # its stdin closed: it ended on its own


@pytest.mark.parametrize("mode, said", [
    ("crash-on-call", "exited (3)"),
    ("silent", "did not answer in 1 s"),
    ("error-tool", "web_search: quota exceeded"),
])
def test_a_server_that_fails_is_said_by_name(mode, said):
    with mcp.Server(FAKE + [mode], timeout=1.0) as s:
        with pytest.raises(mcp.MCPError) as e:
            s.call("web_search", {"query": "x"})
    assert said in str(e.value) and "fake_mcp_server.py" in str(e.value)


def test_a_server_writing_much_to_stderr_does_not_block():
    """stderr is read as it comes: a pipe left unread fills (64 KiB on Linux) and stops the server."""
    with mcp.Server(FAKE + ["noisy"], timeout=10.0) as s:
        for _ in range(8):                                    # 8 × 18,000 characters of logs
            assert s.call("web_search", {"query": "x"}).startswith("1. Result 1")


def test_a_closed_server_leaves_no_process_and_a_missing_command_is_said():
    s = mcp.Server(FAKE)
    s.tools()
    pid = s.proc.pid
    s.close()
    assert s.proc.poll() is not None
    with pytest.raises(ProcessLookupError):
        import os
        os.kill(pid, 0)
    with pytest.raises(mcp.MCPError, match="cannot start no-such-mcp-server"):
        mcp.Server(["no-such-mcp-server"]).tools()
    with pytest.raises(mcp.MCPError, match="no command"):
        mcp.Server([])


def http_server(handler):
    """An httpx client whose POSTs `handler(body, headers) -> (status, headers, content)` answers; requests kept."""
    seen = []

    def answer(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, body, dict(request.headers)))
        status, headers, content = handler(body, request.headers)
        return httpx.Response(status, headers=headers, content=content)
    client = httpx.Client(transport=httpx.MockTransport(answer))
    client.seen = seen
    return client


def streamable(body, headers):
    """A Streamable HTTP server: a session from initialize, a notification answered 202, a call answered by an SSE
    stream that carries a notification first."""
    if body["method"] == "initialize":
        return 200, {"content-type": "application/json", "mcp-session-id": "s-42"}, json.dumps(
            {"jsonrpc": "2.0", "id": body["id"], "result": {"protocolVersion": body["params"]["protocolVersion"],
                                                            "capabilities": {}, "serverInfo": {"name": "remote"}}})
    if "id" not in body:
        return 202, {}, b""
    if body["method"] == "tools/list":
        return 200, {"content-type": "application/json"}, json.dumps(
            {"jsonrpc": "2.0", "id": body["id"], "result": {"tools": [{"name": "search", "inputSchema": {}}]}})
    events = [{"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}},
              {"jsonrpc": "2.0", "id": body["id"], "result": {"content": [{"type": "text", "text": "a\nhttps://x.org"}]}}]
    sse = "".join(f"event: message\ndata: {json.dumps(e)}\n\n" for e in events)
    return 200, {"content-type": "text/event-stream"}, sse.encode()


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_an_event_stream_is_read_whatever_its_line_ends(newline):
    """SSE lines may end in LF, CRLF or CR: duckduckgo-mcp-server (FastMCP) answers with CRLF, which a reader of LF
    alone took for one event and lost the answer (2026-10-08)."""
    def handler(body, headers):
        status, h, content = streamable(body, headers)
        if body.get("method") == "tools/call":
            content = content.decode().replace("\n", newline).encode()
        return status, h, content
    with mcp.Server("https://mcp.example.org/mcp", client=http_server(handler)) as s:
        assert s.call("search", {"query": "q"}) == "a\nhttps://x.org"


def test_tools_are_listed_across_their_pages():
    def paged(body, headers):
        if body["method"] != "tools/list":
            return streamable(body, headers)
        page = {None: ([{"name": "one"}], "c2"), "c2": ([{"name": "two"}], None)}[(body.get("params") or {}).get("cursor")]
        return 200, {"content-type": "application/json"}, json.dumps(
            {"jsonrpc": "2.0", "id": body["id"], "result": {"tools": page[0], **({"nextCursor": page[1]} if page[1] else {})}})
    with mcp.Server("https://mcp.example.org/mcp", client=http_server(paged)) as s:
        assert [t["name"] for t in s.tools()] == ["one", "two"]


def test_a_streamable_http_server_keeps_its_session_and_answers_by_json_or_sse():
    client = http_server(streamable)
    with mcp.Server("https://mcp.example.org/mcp", client=client) as s:
        assert [t["name"] for t in s.tools()] == ["search"]
        assert s.call("search", {"query": "q"}) == "a\nhttps://x.org"
    methods = [(m, b and b.get("method")) for m, b, _ in client.seen]
    assert methods[:4] == [("POST", "initialize"), ("POST", "notifications/initialized"), ("POST", "tools/list"),
                           ("POST", "tools/call")]
    first, later = client.seen[0][2], client.seen[2][2]
    assert "application/json" in first["accept"] and "text/event-stream" in first["accept"]
    assert "mcp-session-id" not in first and later["mcp-session-id"] == "s-42"
    assert later["mcp-protocol-version"] == mcp.PROTOCOL


@pytest.mark.parametrize("status, content, said", [
    (401, b'{"error": "missing api key"}', "401"),
    (200, json.dumps({"jsonrpc": "2.0", "id": 2, "error": {"code": -32602, "message": "bad query"}}).encode(),
     "bad query"),
])
def test_a_remote_refusal_is_said(status, content, said):
    def handler(body, headers):
        if body["method"] in ("initialize", "notifications/initialized"):
            return streamable(body, headers)
        return status, {"content-type": "application/json"}, content
    with mcp.Server("https://mcp.example.org/mcp", client=http_server(handler)) as s:
        with pytest.raises(mcp.MCPError) as e:
            s.tools()
    assert said in str(e.value) and "https://mcp.example.org/mcp" in str(e.value)


def test_a_url_with_a_key_in_it_is_never_said_whole():
    """Providers put the key in the URL (Tavily's ?tavilyApiKey=…): errors name the host and path, never the query."""
    def handler(body, headers):
        return 500, {}, b"boom"
    s = mcp.Server("https://mcp.tavily.com/mcp/?tavilyApiKey=tvly-SECRET123", client=http_server(handler))
    with pytest.raises(mcp.MCPError) as e:
        s.tools()
    assert "tvly-SECRET123" not in str(e.value) and "mcp.tavily.com/mcp/" in str(e.value)
    assert "SECRET" not in s.name


def test_waiting_is_bounded_even_when_a_server_trickles():
    start = time.monotonic()
    with mcp.Server(FAKE + ["silent"], timeout=0.5) as s:
        with pytest.raises(mcp.MCPError):
            s.call("web_search", {"query": "x"})
    assert time.monotonic() - start < 5


def test_a_started_server_is_stopped_when_peresearch_exits(monkeypatch):
    """An agent the interface replaces closes its server (Chat._retire_agent); one left running at exit is stopped
    then (atexit), never left behind."""
    registered = []
    monkeypatch.setattr(mcp.atexit, "register", lambda f: registered.append(f))
    s = mcp.Server(FAKE)
    s.tools()
    assert registered == [s.close]
    s.close()
