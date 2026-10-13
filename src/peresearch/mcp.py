"""A minimal client of the Model Context Protocol (revision 2025-06-18), for the servers a user adds (web search).

A server is a command (stdio: newline-delimited JSON-RPC 2.0 on its stdin and stdout, logs on its stderr) or a URL
(Streamable HTTP: each message POSTed, answered by JSON or by a stream of server-sent events, the session kept by the
Mcp-Session-Id header). The client initializes once, lists tools and calls them; what a server sends meanwhile — a
notification, a ping of its own, logs — is answered or set aside, never mistaken for the answer. Written here rather
than taken from the official SDK, which brings a server stack and asyncio into this synchronous client.
"""

import atexit
import collections
import json
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from urllib.parse import urlsplit

from peresearch import guard

PROTOCOL = "2025-06-18"
CLIENT = {"name": "peresearch", "version": "0.1"}
_CREDENTIAL_ENV = re.compile(r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?|AUTH)(?:_|$)", re.I)


def service_environment(explicit: dict | None) -> dict:
    """Keep runtime settings, but credentials belong only to their configured service.

    Explicit server variables win, including keys the server actually requires.
    This is environment hygiene, not an OS sandbox for a same-user executable.
    """
    inherited = {name: value for name, value in os.environ.items()
                 if not name.startswith("PERESEARCH_") and not _CREDENTIAL_ENV.search(name)}
    return {**inherited, **(explicit or {})}


class MCPError(RuntimeError):
    """A server that could not be started, failed, refused or said nothing, said with its name."""


class Server:
    """One MCP server: `target` a URL (http or https), a command line, or the command as a list. Used as a context
    manager, or closed with close(); started on first use."""

    def __init__(self, target, env: dict | None = None, client=None, timeout: float = 60.0):
        if isinstance(target, str) and target.startswith(("http://", "https://")):
            self.url, self.command = target, None
            parts = urlsplit(target)
            self.name = f"{parts.scheme}://{parts.netloc}{parts.path}"     # never its query: a key may be there
        else:
            command = shlex.split(target) if isinstance(target, str) else list(target)
            if not command:
                raise MCPError("no command for the MCP server")
            self.url, self.command = None, command
            self.name = guard.redact(" ".join(command))[0]                  # a key given as an argument: redacted
        self.env, self.client, self.timeout = env, client, timeout
        self.proc, self.info, self._started, self._id, self._session = None, None, False, 0, None
        self._opened, self._own_client = False, False
        self._lock = threading.RLock()                     # _request starts the server, which sends initialize
        self._stderr: collections.deque = collections.deque(maxlen=20)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- the protocol ---

    def tools(self) -> list[dict]:
        """Every tool the server offers: {name, description, inputSchema}, across its pages."""
        out, cursor = [], None
        while True:
            got = self._request("tools/list", {"cursor": cursor} if cursor else {})
            out += got.get("tools") or []
            cursor = got.get("nextCursor")
            if not cursor:
                return out

    def call(self, name: str, arguments: dict) -> str:
        """The text a tool returns (its text blocks, joined); a tool that says it failed raises MCPError."""
        got = self._request("tools/call", {"name": name, "arguments": arguments})
        text = "\n".join(c.get("text", "") for c in got.get("content") or [] if isinstance(c, dict)
                         and c.get("type") == "text")
        if got.get("isError"):
            raise MCPError(f"{self.name}: {name}: {text or 'the tool failed'}")
        return text

    def _start(self) -> None:
        if self._started:
            return
        try:
            if not self._opened:
                if self.command:
                    try:
                        self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                     stderr=subprocess.PIPE, env=service_environment(self.env),
                                                     start_new_session=True)
                    except OSError as e:
                        raise MCPError(f"cannot start {self.command[0]}: {e.strerror or e}") from None
                    self._lines: queue.Queue = queue.Queue()
                    self._stderr = collections.deque(maxlen=20)
                    atexit.register(self.close)                 # never left running after peresearch ends
                    threading.Thread(target=self._read_out, args=(self.proc, self._lines), daemon=True).start()
                    threading.Thread(target=self._read_err, args=(self.proc, self._stderr), daemon=True).start()
                elif self.client is None:
                    import httpx
                    self.client = httpx.Client(timeout=self.timeout, follow_redirects=True)
                    self._own_client = True
                self._opened = True
            self.info = self._request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                                     "clientInfo": CLIENT})
            self._notify("notifications/initialized")
            self._started = True
        except BaseException:
            self.close()
            raise

    def _read_out(self, proc, lines) -> None:
        for raw in iter(proc.stdout.readline, b""):
            lines.put(raw.decode("utf-8", "replace").rstrip("\n"))
        lines.put(None)                                       # the end: this transport closed its stdout

    def _read_err(self, proc, stderr) -> None:
        for raw in iter(proc.stderr.readline, b""):
            stderr.append(raw.decode("utf-8", "replace").rstrip("\n"))

    def _request(self, method: str, params: dict) -> dict:
        with self._lock:
            if method != "initialize":
                self._start()
            self._id += 1
            message = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
            return self._http(message) if self.url else self._stdio(message)

    def _notify(self, method: str) -> None:
        message = {"jsonrpc": "2.0", "method": method}
        if self.url:
            self._post(message)
        else:
            self._write(message)

    # --- stdio ---

    def _write(self, message: dict) -> None:
        try:
            self.proc.stdin.write((json.dumps(message) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise self._gone() from None

    def _gone(self) -> MCPError:
        code = self.proc.wait(5) if self.proc.poll() is None else self.proc.returncode
        last = next((x for x in reversed(self._stderr) if x.strip()), "")
        return MCPError(f"{self.name}: exited ({code})" + (f": {last[:200]}" if last else ""))

    def _stdio(self, message: dict) -> dict:
        self._write(message)
        deadline = time.monotonic() + self.timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise MCPError(f"{self.name}: did not answer in {self.timeout:g} s")
            try:
                line = self._lines.get(timeout=min(left, 0.2))
            except queue.Empty:
                if self.proc.poll() is not None and self._lines.empty():
                    raise self._gone() from None
                continue
            if line is None:
                raise self._gone()
            try:
                got = json.loads(line)
            except ValueError:                                # not JSON-RPC (a print on stdout): set aside
                continue
            answer = self._handle(got, message["id"], self._write)
            if answer is not None:
                return answer

    def _handle(self, got, want, reply) -> dict | None:
        """The result when `got` answers request `want`; a server's own request answered (ping) or refused; a
        notification set aside (None)."""
        if not isinstance(got, dict):
            return None
        if "method" in got:
            if "id" in got:
                reply({"jsonrpc": "2.0", "id": got["id"], "result": {}} if got["method"] == "ping" else
                      {"jsonrpc": "2.0", "id": got["id"], "error": {"code": -32601, "message": "not supported"}})
            return None
        if got.get("id") != want:
            return None
        if "error" in got:
            err = got["error"] if isinstance(got["error"], dict) else {"message": str(got["error"])}
            raise MCPError(f"{self.name}: {err.get('message', 'error')}")
        return got.get("result") or {}

    # --- Streamable HTTP ---

    def _post(self, message: dict):
        import httpx
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        if self._session:
            headers["Mcp-Session-Id"] = self._session
        if self.info:
            headers["MCP-Protocol-Version"] = self.info.get("protocolVersion", PROTOCOL)
        try:
            r = self.client.post(self.url, content=json.dumps(message), headers=headers)
        except httpx.HTTPError as e:
            raise MCPError(f"{self.name}: {type(e).__name__}") from None
        if r.status_code >= 400:
            raise MCPError(f"{self.name}: {r.status_code} {guard.redact(r.text[:200])[0]}")
        if r.headers.get("mcp-session-id"):
            self._session = r.headers["mcp-session-id"]
        return r

    def _http(self, message: dict) -> dict:
        r = self._post(message)
        kind = r.headers.get("content-type", "")
        if "text/event-stream" in kind:
            messages = []
            text = r.text.replace("\r\n", "\n").replace("\r", "\n")       # SSE lines end in LF, CRLF or CR
            for event in text.split("\n\n"):
                data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
                if data:
                    try:
                        messages.append(json.loads(data))
                    except ValueError:
                        continue
        else:
            try:
                body = r.json()
            except ValueError:
                raise MCPError(f"{self.name}: the answer is not JSON") from None
            messages = body if isinstance(body, list) else [body]
        for got in messages:
            answer = self._handle(got, message["id"], self._post)
            if answer is not None:
                return answer
        raise MCPError(f"{self.name}: no answer to {message['method']}")

    def close(self) -> None:
        """Stop a server this client started (its process group), or end the session of a remote one."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, 15)
                    self.proc.wait(5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    os.killpg(self.proc.pid, 9)
                    self.proc.wait(5)
        if self.url and self._session and self.client is not None:
            try:
                self.client.delete(self.url, headers={"Mcp-Session-Id": self._session,
                    "MCP-Protocol-Version": (self.info or {}).get("protocolVersion", PROTOCOL)})
            except Exception:                                 # ending a session is a courtesy: never an error
                pass
        if getattr(self, "_own_client", False):
            self.client.close()
            self.client, self._own_client = None, False
        self._opened, self._started, self.info, self._session = False, False, None, None
