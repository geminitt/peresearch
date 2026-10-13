"""A small MCP server over stdio for tests/test_mcp.py: newline-delimited JSON-RPC on stdin/stdout, logs on stderr.
It names its tools, answers a search with the URLs a real search server returns as text, sends a notification and a
ping of its own before answering a call (a client must answer the ping), and can misbehave on purpose:

    python fake_mcp_server.py [ok | crash-on-call | silent | error-tool | noisy | stale]
"""
import json
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "ok"
TOOLS = [{"name": "web_search", "description": "Search the web.",
          "inputSchema": {"type": "object", "properties": {"query": {"type": "string"},
                                                           "max_results": {"type": "integer"}},
                          "required": ["query"]}},
         {"name": "fetch", "description": "Fetch a page.",
          "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}}}]


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def read():
    line = sys.stdin.readline()
    return json.loads(line) if line else None


while True:
    msg = read()
    if msg is None:
        break
    if MODE == "noisy":
        sys.stderr.write("log line " * 2000 + "\n")            # more than a pipe holds, if nobody read it
        sys.stderr.flush()
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:                                           # a notification: nothing to answer
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": msg["params"]["protocolVersion"],
                                                      "capabilities": {"tools": {}},
                                                      "serverInfo": {"name": "fake", "version": "1"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        if MODE == "crash-on-call":
            sys.exit(3)
        if MODE == "silent":
            continue
        if MODE == "stale":                                   # a late answer to some earlier request first
            send({"jsonrpc": "2.0", "id": 999, "result": {"content": [{"type": "text", "text": "WRONG"}]}})
        send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "searching"}})
        send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})      # the server asks; the client must answer
        answer = read()
        assert answer == {"jsonrpc": "2.0", "id": "srv-1", "result": {}}, answer
        args = msg["params"]["arguments"]
        if MODE == "error-tool":
            send({"jsonrpc": "2.0", "id": mid, "result": {"isError": True,
                                                          "content": [{"type": "text", "text": "quota exceeded"}]}})
            continue
        n = args.get("max_results", 2)
        text = "\n".join(f"{i + 1}. Result {i + 1} for {args['query']}\nhttps://example.org/{i + 1}" for i in range(n))
        send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text},
                                                                   {"type": "image", "data": "", "mimeType": "x"}]}})
    else:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}})
