"""Routes and differences declared beforehand (owner decision 2026-10-14, model gateway step 2): the provider registry
says each provider's protocol and `Compatibility`, the model profile what a model needs; a server's errors are read
only for what cannot be known beforehand.

Red before the change (2026-10-15): an OpenAI model that takes tools only through the Responses API first sent a
Chat Completions request to be refused; Ollama was sent the chat template's switch it ignores; a cloud endpoint was
asked Ollama's /api/ps."""

from types import SimpleNamespace as NS

import pytest

from peresearch import profiles, providers
from peresearch.llm import LLM
from tests.helpers import request, streamed
from tests.test_llm_responses import TOOLS, USER, client, events


@pytest.fixture(autouse=True)
def plain(monkeypatch):
    for name in ("PERESEARCH_LLM_TOOLS", "PERESEARCH_LLM_CONTEXT", "PERESEARCH_LLM_API", "PERESEARCH_LLM_THINKING",
                 "PERESEARCH_LLM_PROVIDER", "PERESEARCH_LLM_PROTOCOL"):
        monkeypatch.delenv(name, raising=False)


def test_every_provider_declares_its_protocol_and_compatibility():
    for p in providers.PROVIDERS:
        assert p.protocol in ("chat", "anthropic", "gemini") and isinstance(p.compatibility, providers.Compatibility)
    assert providers.get("google").protocol == "gemini"
    assert providers.get("ollama").compatibility.thinking_switch == "reasoning_effort"
    assert not any(p.compatibility.ollama_probe for p in providers.PROVIDERS if p.kind == "cloud")
    assert all(p.compatibility.ollama_probe for p in providers.PROVIDERS if p.kind == "local")


def test_a_model_that_takes_tools_only_through_responses_goes_there_on_its_first_request(home):
    assert profiles.profile("gpt-6.1-sol").tools == "responses"          # OpenAI's model page, read 2026-10-12
    chat, sent = [], []

    def responses(**kw):
        sent.append(kw)
        return iter(events(text="ok"))
    llm = LLM(client=client(lambda **kw: chat.append(kw), responses), model="gpt-6.1-sol", retries=1, provider="openai")
    assert llm.chat(request(USER, TOOLS)).text == "ok"
    assert chat == [] and len(sent) == 1 and sent[0]["tools"][0]["name"] == "search_notes"
    assert llm.route == "responses"


def test_ollama_is_switched_through_its_own_field_only(home):
    sent = []
    fake = NS(chat=NS(completions=NS(create=lambda **kw: sent.append(kw) or iter(streamed(content="ok")))),
              models=NS(list=lambda: NS(data=[])))
    llm = LLM(client=fake, model="qwen3:0.6b", retries=1, provider="ollama", thinking=False)
    llm.chat(request(USER))
    assert sent[0]["extra_body"] == {"reasoning_effort": "none"}         # no chat_template_kwargs: Ollama ignores it


@pytest.mark.parametrize("provider,asked", [("openai", False), ("anthropic", False), ("ollama", True), ("", True)])
def test_only_a_server_that_may_be_ollama_is_asked_its_loaded_context(home, provider, asked):
    gets = []
    http = NS(get=lambda url, timeout: gets.append(url) or NS(status_code=404), post=lambda *a, **k: None)
    fake = NS(_client=http, base_url="https://endpoint.test/v1", models=NS(list=lambda: NS(data=[])))
    llm = LLM(client=fake, model="some-model", retries=1, provider=provider or None)
    assert llm.context_length() is None or llm.context_length() > 0
    assert (gets == ["https://endpoint.test/api/ps"]) is asked


def test_compatibility_starts_empty_and_an_unregistered_provider_may_be_ollama():
    assert providers.Compatibility() == providers.Compatibility(thinking_switch="", ollama_probe=False)
    assert providers.compatibility("openai") == providers.Compatibility() and \
        providers.compatibility("nope") == providers.LOCAL and providers.compatibility("") == providers.LOCAL
    assert providers.protocol("x") == "chat" and providers.protocol("x", "responses") == "responses"


@pytest.mark.parametrize("protocol,route,module", [("gemini", "gemini", "google"), ("anthropic", "anthropic", "anthropic"),
                                                    ("responses", "responses", "openai"), ("chat", "chat", "openai")])
def test_a_protocol_set_for_an_added_provider_chooses_the_route_and_the_sdk(home, monkeypatch, protocol, route, module):
    monkeypatch.setenv("PERESEARCH_LLM_PROTOCOL", protocol)
    llm = LLM(url="https://proxy.example/v1", key="synthetic-key", model="m")
    assert llm.route == route and type(llm.client).__module__.split(".")[0] == module and llm.provider == ""


def test_the_openai_client_reaches_the_url_with_the_key_and_the_responses_adapter_uses_it_too():
    from peresearch.protocols import anthropic, chat, responses
    for made in (chat.client("https://api.example/v1", "synthetic-key"), responses.client("https://api.example/v1", "synthetic-key")):
        assert str(made.base_url).rstrip("/") == "https://api.example/v1" and made.api_key == "synthetic-key"
    assert chat.client("http://localhost:1/v1", "").api_key == "none"
    made = anthropic.client("https://claude.example/v1/", "synthetic-key")
    assert str(made.base_url).rstrip("/") == "https://claude.example" and made.api_key == "synthetic-key"


# --- connections: a provider the user added -----------------------------------------------------------------------

def test_a_protocol_is_checked_and_named_in_full():
    from peresearch import connections
    assert connections.check_protocol(" Gemini ") == "gemini"
    with pytest.raises(ValueError, match=r"^The protocol must be one of chat, responses, anthropic or gemini\.$"):
        connections.check_protocol("smoke")


def test_a_provider_added_with_a_protocol_is_not_mistaken_for_a_known_one(home):
    from peresearch import connections
    entry = connections.add("chat", "Mine", "https://api.openai.com/v1", "k", protocol="anthropic")
    assert (entry.provider, entry.protocol) == ("", "anthropic")
    chosen = connections.selection(entry, "m")
    assert chosen["PERESEARCH_LLM_PROVIDER"] == "" and chosen["PERESEARCH_LLM_PROTOCOL"] == "anthropic"
    listed = connections.add("chat", "Listed", "https://api.openai.com/v1", "k2")
    chosen = connections.selection(listed, "m")
    assert (listed.provider, chosen["PERESEARCH_LLM_PROVIDER"], chosen["PERESEARCH_LLM_PROTOCOL"]) == ("openai", "openai", "")
    again = connections.connect("chat", "Mine", "https://api.openai.com/v1", "k", protocol="anthropic")
    assert again.id == entry.id and connections.connect("chat", "Mine", "https://api.openai.com/v1", "k").id != entry.id


def test_models_are_listed_by_the_sdk_of_the_protocol_and_the_client_is_closed(home, monkeypatch):
    from peresearch import connections
    import anthropic as anthropic_sdk
    import openai as openai_sdk
    from peresearch.protocols import gemini
    made, closed = [], []

    class Client:
        def __init__(self, kind, *a, **kw):
            made.append((kind, a, kw))
            self.models = NS(list=lambda: NS(data=[NS(id="b"), NS(id="a"), NS(id=None)]))

        def close(self):
            closed.append(made[-1][0])
    monkeypatch.setattr(anthropic_sdk, "Anthropic", lambda **kw: Client("anthropic", **kw))
    monkeypatch.setattr(openai_sdk, "OpenAI", lambda **kw: Client("openai", **kw))
    monkeypatch.setattr(gemini, "client", lambda url, key: (made.append(("gemini", (url, key), {})), NS(
        models=NS(list=lambda: [NS(name="models/g")])))[1])
    claude = connections.add("chat", "C", "https://api.anthropic.com", "key-c")
    google = connections.add("chat", "G", "https://generativelanguage.googleapis.com/v1beta/openai/", "key-g")
    local = connections.add("chat", "L", "http://localhost:11434/v1", "")
    embed = connections.add("embedding", "E", "https://generativelanguage.googleapis.com/v1beta/openai/", "key-e")
    assert connections.discover(claude) == ["a", "b"] and made[-1][2]["api_key"] == "key-c" and closed == ["anthropic"]
    assert connections.discover(google) == ["g"] and made[-1] == ("gemini", ("https://generativelanguage.googleapis.com/v1beta/openai", "key-g"), {})
    assert connections.discover(local) == ["a", "b"] and made[-1][2]["api_key"] == "none" and closed[-1] == "openai"
    assert connections.discover(embed) == ["a", "b"] and made[-1][0] == "openai"      # embeddings: always /v1/models
    before = len(made)
    fake = NS(models=NS(list=lambda: NS(data=[NS(id="z")])))
    assert connections.discover(claude, client=fake) == ["z"] and len(made) == before   # a client given is used, not built


def test_a_saved_connection_with_an_unknown_protocol_is_refused(home):
    from peresearch import connections
    connections.add("chat", "Mine", "https://proxy.example/v1", "k", protocol="chat")
    path = home / "models.json"
    path.write_text(path.read_text().replace('"chat",\n      "name"', '"chat",\n      "name"').replace('"protocol": "chat"', '"protocol": "pigeon"'))
    with pytest.raises(ValueError, match="catalog is invalid"):
        connections.load()


def test_a_connection_that_names_chat_is_listed_over_chat_whatever_its_address_looks_like(home, monkeypatch):
    """Added with its own protocol, a connection is not taken for the provider its address resembles."""
    from peresearch import connections
    import openai as openai_sdk
    used = []
    monkeypatch.setattr(openai_sdk, "OpenAI", lambda **kw: used.append("openai") or NS(
        models=NS(list=lambda: NS(data=[NS(id="m")])), close=lambda: None))
    entry = connections.add("chat", "Mine", "https://api.anthropic.com", "k", protocol="chat")
    assert connections.discover(entry) == ["m"] and used == ["openai"]
