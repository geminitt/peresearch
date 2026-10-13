"""Private model catalogs and metadata discovery, without paid provider requests."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace as NS

import pytest

from peresearch import connections, settings


def test_catalog_is_private_and_selection_changes_only_its_role(home):
    settings.save({"PERESEARCH_LLM_URL": "http://chat.test/v1", "PERESEARCH_LLM_MODEL": "chat-id",
                   "PERESEARCH_LLM_KEY": "synthetic-chat-key"})
    entry = connections.add("embedding", "Embedding service", "http://embed.test/v1/", "synthetic-embed-key", "embed-id")
    assert "synthetic-embed-key" not in repr(entry)
    assert entry.url == "http://embed.test/v1"
    assert (home / "models.json").stat().st_mode & 0o777 == 0o600
    assert home.stat().st_mode & 0o777 == 0o700
    # the catalog holds what was saved; the chat endpoint in use stays in settings.env, its key not copied here
    assert [e.role for e in connections.load()] == ["embedding"]
    assert "synthetic-chat-key" not in (home / "models.json").read_text()
    assert connections.selection(entry, "another-id") == {
        "PERESEARCH_EMBED_URL": entry.url, "PERESEARCH_EMBED_MODEL": "another-id",
        "PERESEARCH_EMBED_KEY": entry.key}
    assert settings.get("PERESEARCH_LLM_MODEL") == "chat-id"


def test_discovery_uses_injected_client_even_when_falsy_and_does_not_close_it(home, monkeypatch):
    import openai

    class Client:
        models = NS(list=lambda: NS(data=[NS(id="z"), NS(id="a"), NS(id="z"), NS(id="")]))
        def __bool__(self):
            return False
        def close(self):
            raise AssertionError("injected client is not owned")

    monkeypatch.setattr(openai, "OpenAI", lambda **kw: (_ for _ in ()).throw(AssertionError("unexpected client")))
    assert connections.discover(connections.Connection("x", "chat", "Local", "http://local.test/v1"), Client()) == ["a", "z"]


def test_metadata_discovery_uses_only_get_models(home):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.command, self.path, self.headers.get("Authorization")))
            body = json.dumps({"object": "list", "data": [{"id": "model-id", "object": "model", "created": 0,
                                                          "owned_by": "fixture"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        entry = connections.Connection("x", "chat", "Fixture", f"http://127.0.0.1:{server.server_port}/v1", "synthetic-key")
        assert connections.discover(entry) == ["model-id"]
        assert requests == [("GET", "/v1/models", "Bearer synthetic-key")]
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_changed_identity_cannot_accept_discovery_and_invalid_catalog_is_preserved(home):
    entry = connections.add("chat", "Old", "http://old.test/v1", "old-key")
    changed = connections.load()
    changed[0].key = "new-key"
    connections.save(changed)
    before = (home / "models.json").read_bytes()
    with pytest.raises(ValueError, match="changed"):
        connections.update_models(entry.id, ["stale-id"], entry)
    assert (home / "models.json").read_bytes() == before
    (home / "models.json").write_text('{"schema": 999, "connections": []}')
    with pytest.raises(ValueError, match="invalid"):
        connections.add("chat", "New", "http://new.test/v1", "key")
    assert (home / "models.json").read_text() == '{"schema": 999, "connections": []}'


@pytest.mark.parametrize("url", ["file:///tmp/model", "https://key:secret@example.test/v1", "not-a-url"])
def test_invalid_endpoints_do_not_create_catalog(home, url):
    with pytest.raises(ValueError):
        connections.add("chat", "Invalid", url, "")
    assert not (home / "models.json").exists()
