"""`/model` → a: connect a model in three steps, as OpenCode does — choose where the model runs, give only what that
place needs (a cloud's API key, a local server's address), then choose the model from the list the endpoint returns.

The owner, 2026-10-14: "nhiều trường cần nhập quá … bình thường chỉ cần nhập API, còn model thì tự chọn". Saving
sends nothing; the model list is requested only by the explicit Enter of the second step and sends no prompt."""

from peresearch import connections, settings, tui
from peresearch.windows import PRESETS
from tests.helpers import settle, texts, tui_run


async def submit(app, pilot, command):
    app.query_one("#ask", tui.Prompt).value = command
    await pilot.press("enter")
    await pilot.pause()


def choose(app, name):
    """Highlight the row of the connect list whose id is `name`."""
    listing = app.screen.query_one("#connect-list")
    ids = [listing.get_option_at_index(i).id for i in range(listing.option_count)]
    listing.highlighted = ids.index(name)


def test_presets_name_verified_endpoints():
    urls = {name: url for name, url, _ in PRESETS}
    assert urls["OpenAI"] == "https://api.openai.com/v1"
    assert urls["OpenRouter"] == "https://openrouter.ai/api/v1"
    assert urls["Groq"] == "https://api.groq.com/openai/v1"
    assert urls["Ollama"] == "http://localhost:11434/v1"
    assert urls["LM Studio"] == "http://localhost:1234/v1"
    assert all(kind in ("cloud", "local") for _, _, kind in PRESETS)


def test_connect_reuses_an_existing_connection_instead_of_a_duplicate(home):
    first = connections.connect("chat", "OpenAI", "https://api.openai.com/v1", "synthetic-key")
    again = connections.connect("chat", "OpenAI", "https://api.openai.com/v1/", "synthetic-key")
    other = connections.connect("chat", "OpenAI", "https://api.openai.com/v1", "other-key")
    assert again.id == first.id and other.id != first.id
    assert len([e for e in connections.load() if e.role == "chat"]) == 2


def test_a_cloud_takes_only_its_key_then_the_listed_model_is_chosen(home):
    asked = []

    def listed(entry):
        asked.append((entry.url, entry.key))
        return ["gpt-6.1-mini", "gpt-6.1-sol", "text-embedding-4"]

    async def steps(app, pilot):
        app.agent = None
        await submit(app, pilot, "/model")
        app.screen.discover_models = listed
        await pilot.press("a")
        await pilot.pause()
        assert app.screen.id == "connect-window"
        choose(app, "preset:OpenAI")
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.id == "api-window"
        inputs = [w.id for w in app.screen.query("Input")]
        assert inputs == ["api-key"]                                  # a cloud: its key, nothing else
        assert asked == [] and connections.load() == []               # nothing sent, nothing saved yet
        app.screen.query_one("#api-key").value = "synthetic-key"
        await pilot.press("enter")                                    # connect and list: the explicit request
        await settle(app, pilot, lambda: app.screen.id == "pick-window")
        assert asked == [("https://api.openai.com/v1", "synthetic-key")]
        warning = app.screen.query_one("#window-detail")              # listed is not tested: said while choosing
        assert warning.display and "not yet tested" in str(warning.render())
        app.screen.query_one("#pick-filter").value = "sol"
        await pilot.pause()
        listing = app.screen.query_one("#pick-list")
        assert [listing.get_option_at_index(i).id for i in range(listing.option_count)] == ["gpt-6.1-sol"]
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.id == "model-window"                        # back where the flow began
        assert settings.get("PERESEARCH_LLM_URL") == "https://api.openai.com/v1"
        assert settings.get("PERESEARCH_LLM_MODEL") == "gpt-6.1-sol"
        assert settings.get("PERESEARCH_LLM_KEY") == "synthetic-key"
        shown = " ".join(str(w.render()) for w in app.screen.query("Static"))
        assert "synthetic-key" not in shown
        await pilot.press("escape")
        await pilot.pause()
        assert texts(app) == ""

    tui_run(lambda event: None, steps)


def test_a_local_server_takes_only_its_address_and_a_failed_list_allows_an_id(home):
    def refused(entry):
        raise ConnectionError("connection refused")

    async def steps(app, pilot):
        await submit(app, pilot, "/model")
        app.screen.discover_models = refused
        await pilot.press("a")
        await pilot.pause()
        choose(app, "preset:Ollama")
        await pilot.press("enter")
        await pilot.pause()
        inputs = {w.id: w.value for w in app.screen.query("Input")}
        assert inputs == {"api-url": "http://localhost:11434/v1"}     # editable address, no key
        await pilot.press("enter")
        await settle(app, pilot, lambda: "connection refused" in str(app.screen.query_one("#window-message").render()))
        assert app.screen.id == "api-window"
        await pilot.press("ctrl+t")                                   # type a known ID instead
        await pilot.pause()
        app.screen.query_one("Input").value = "qwen3:8b"
        await pilot.press("enter")
        await pilot.pause()
        assert settings.get("PERESEARCH_LLM_MODEL") == "qwen3:8b"
        assert settings.get("PERESEARCH_LLM_URL") == "http://localhost:11434/v1"
        assert app.screen.id == "model-window"

    tui_run(lambda event: None, steps)


def test_a_connected_endpoint_goes_straight_to_its_models(home):
    entry = connections.connect("embedding", "Ollama", "http://localhost:11434/v1", "")
    connections.update_models(entry.id, ["bge-m3", "nomic-embed-text"], entry)

    async def steps(app, pilot):
        await submit(app, pilot, "/model")
        await pilot.press("right")                                    # the embeddings role
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
        choose(app, "entry:" + entry.id)
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.id == "pick-window"
        await pilot.press("down", "enter")
        await pilot.pause()
        assert settings.get("PERESEARCH_EMBED_MODEL") == "nomic-embed-text"
        assert not settings.get("PERESEARCH_LLM_MODEL")              # another role is untouched

    tui_run(lambda event: None, steps)


def test_a_late_list_after_leaving_the_step_is_not_applied(home):
    import threading
    release = threading.Event()

    def slow(entry):
        release.wait(2)
        return ["late-model"]

    async def steps(app, pilot):
        await submit(app, pilot, "/model")
        app.screen.discover_models = slow
        await pilot.press("a")
        await pilot.pause()
        choose(app, "preset:OpenAI")
        await pilot.press("enter")
        await pilot.pause()
        app.screen.query_one("#api-key").value = "synthetic-key"
        await pilot.press("enter")
        await pilot.press("escape")                                   # left the step before the list came
        await pilot.pause()
        release.set()
        await settle(app, pilot, lambda: not any(t.is_alive() for t in app.background))
        await pilot.pause()
        assert app.screen.id == "connect-window"
        assert not settings.get("PERESEARCH_LLM_MODEL")

    try:
        tui_run(lambda event: None, steps)
    finally:
        release.set()


def test_connecting_the_endpoint_already_in_use_does_not_list_it_twice(home):
    """The endpoint in the active settings is shown as a connection even before it is saved; connecting the same
    endpoint and key adopts it instead of saving a second row (two "✔ in use" rows, found on 2026-10-14)."""
    settings.save({"PERESEARCH_LLM_URL": "http://127.0.0.1:8000/v1", "PERESEARCH_LLM_MODEL": "qwen3-8b"})
    entry = connections.connect("chat", "vLLM, llama.cpp or another server", "http://127.0.0.1:8000/v1", "")
    connections.update_models(entry.id, ["qwen3-8b"], entry)
    chat = [e for e in connections.load() if e.role == "chat"]
    assert len(chat) == 1 and chat[0].url == "http://127.0.0.1:8000/v1"


def test_the_catalog_holds_only_saved_connections(home):
    """One source of truth: models.json lists what was saved; the active settings are not mixed into it."""
    settings.save({"PERESEARCH_LLM_URL": "http://127.0.0.1:8000/v1", "PERESEARCH_LLM_MODEL": "qwen3-8b"})
    assert connections.load() == []


def test_opening_models_saves_the_endpoint_in_use_once_but_never_a_key_from_the_environment(home, monkeypatch):
    settings.save({"PERESEARCH_LLM_URL": "http://127.0.0.1:8000/v1", "PERESEARCH_LLM_MODEL": "qwen3-8b"})
    monkeypatch.setenv("PERESEARCH_EMBED_URL", "https://embed.test/v1")
    monkeypatch.setenv("PERESEARCH_EMBED_KEY", "synthetic-environment-key")

    async def steps(app, pilot):
        for _ in range(2):                                            # opened twice: still one row
            await submit(app, pilot, "/model")
            listing = app.screen.query_one("#models-chat")
            app.rows = listing.option_count
            await pilot.press("escape")
            await pilot.pause()

    app = tui_run(lambda event: None, steps)
    saved = connections.load()
    assert app.rows == 1 and [(e.role, e.url, e.models) for e in saved] == [
        ("chat", "http://127.0.0.1:8000/v1", ["qwen3-8b"])]
    assert "synthetic-environment-key" not in (home / "models.json").read_text()


def test_an_endpoint_from_the_environment_is_named_as_in_use(home, monkeypatch):
    monkeypatch.setenv("PERESEARCH_LLM_URL", "https://env-model.test/v1")
    monkeypatch.setenv("PERESEARCH_LLM_MODEL", "env-model")

    async def steps(app, pilot):
        await submit(app, pilot, "/model")
        app.said = str(app.screen.query_one("#window-message").render())
    app = tui_run(lambda event: None, steps)
    assert "env-model.test" in app.said and "environment" in app.said and "No API connected" not in app.said
