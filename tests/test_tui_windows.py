"""User controls live in grouped windows, with local shortcuts and a clean transcript."""

from textual.screen import ModalScreen
from textual.widgets import Input, Markdown, Static

from peresearch import settings, tui
from peresearch.workspace import Workspace
from peresearch.zetokrag.index import Index
from tests.conftest import FakeEmbedder
from tests.helpers import texts, tui_run, notes
from tests.helpers import settle
from peresearch import connections
import pytest


async def submit(app, pilot, command):
    app.query_one("#ask", tui.Prompt).value = command
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.parametrize("no_color", [True, False])
def test_window_text_keeps_terminal_defaults_in_color_and_no_color_modes(home, monkeypatch, no_color):
    from rich.color import ColorType
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    async def steps(app, pilot):
        await submit(app, pilot, "/model")
        found = []
        for strip in app.screen._compositor.render_strips():
            for segment in strip:
                if "Models and APIs" in segment.text:
                    found.append(segment)
                    assert segment.style.color.type == ColorType.DEFAULT
                    assert segment.style.bgcolor.type == ColorType.DEFAULT
        assert found
    tui_run(lambda event: None, steps)


def test_status_is_a_floating_window_and_never_enters_the_transcript(home):
    async def steps(app, pilot):
        await submit(app, pilot, "/status")
        assert isinstance(app.screen, ModalScreen)
        assert any("synthetic-status" in w.source for w in app.screen.query(Markdown))
        await pilot.press("escape")
        await pilot.pause()
        assert texts(app) == ""

    tui_run(lambda event: None, steps, status=lambda reason=None: "synthetic-status")


def test_folder_group_has_local_add_and_remove_controls(home, tmp_path):
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    folder = tmp_path / "documents"
    folder.mkdir()

    async def steps(app, pilot):
        await submit(app, pilot, "/folders")
        assert app.screen.id == "folders-window"
        await pilot.press("a")
        await pilot.pause()
        assert app.screen.id == "folder-path-window"
        app.screen.query_one(Input).value = str(folder)
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert ws.folders() == [folder.resolve()]
        assert app.screen.id == "folders-window"
        await settle(app, pilot, lambda: not app.indexing)                  # removing waits for the new folder's update
        await pilot.press("r")
        await pilot.pause()
        assert ws.folders() == []
        await pilot.press("escape")
        await pilot.pause()
        assert texts(app) == ""

    tui_run(lambda event: None, steps, workspace=ws)
    assert "/folders" in tui.COMMANDS
    assert not any(command in tui.COMMANDS for command in ("/add", "/remove", "/index"))


def test_model_group_opens_nested_connection_form_without_showing_saved_key(home):
    settings.save({"PERESEARCH_LLM_URL": "https://model.test/v1", "PERESEARCH_LLM_MODEL": "chosen-model",
                   "PERESEARCH_LLM_KEY": "synthetic-saved-key"})

    async def steps(app, pilot):
        await submit(app, pilot, "/model")
        assert app.screen.id == "model-window"
        shown = " ".join(str(w.render()) for w in app.screen.query(Static))
        assert "synthetic-saved-key" not in shown
        await pilot.press("a")                                         # where the model runs, then its key
        await pilot.pause()
        assert app.screen.id == "connect-window"
        listing = app.screen.query_one("#connect-list")
        listing.highlighted = [listing.get_option_at_index(i).id for i in range(listing.option_count)].index("other")
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.id == "api-window"
        key = app.screen.query_one("#api-key", Input)
        assert key.password and key.value == ""
        await pilot.press("escape", "escape")
        await pilot.pause()
        assert app.screen.id == "model-window"
        await pilot.press("escape")
        await pilot.pause()
        assert texts(app) == ""

    tui_run(lambda event: None, steps)


def test_window_selection_does_not_dispatch_a_row_id_as_a_chat_command(home):
    connections.add("chat", "Fixture", "http://model.test/v1", "", "model-id")
    async def steps(app, pilot):
        seen, command = [], app.command
        def recorded(text):
            seen.append(text)
            command(text)
        app.command = recorded
        await submit(app, pilot, "/model")
        await pilot.press("enter")
        await pilot.pause()
        assert seen == ["/model"]
        assert settings.get("PERESEARCH_LLM_MODEL") == "model-id"
    tui_run(lambda event: None, steps)


@pytest.mark.parametrize("command", ["/folders", "/model", "/settings", "/sessions", "/check", "/status", "/help"])
@pytest.mark.parametrize("no_color", [True, False])
def test_window_actions_fit_small_terminal_and_escape_restores_prompt(home, command, no_color, monkeypatch):
    from textual.widgets import Button
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)

    async def steps(app, pilot):
        await submit(app, pilot, command)
        window = app.screen
        box = window.query_one("#window")
        assert box.region.x == (80 - box.region.width) // 2
        assert box.region.y == (24 - box.region.height) // 2
        from rich.color import ColorType
        from peresearch import mascot
        mascot_colors = {tuple(color.triplet) for color in mascot.COLORS.values() if color.triplet is not None}
        for y, strip in enumerate(window._compositor.render_strips()):
            x = 0
            for segment in strip:
                if segment.text.strip() and segment.style:
                    for color in (segment.style.color, segment.style.bgcolor):
                        if color is not None and color.type == ColorType.TRUECOLOR:
                            # The owner's mascot is the one documented RGB exception, outside the floating panel.
                            assert app.bird.region.contains(x, y) and not box.region.contains(x, y)
                            assert tuple(color.triplet) in mascot_colors
                        else:
                            assert color is None or color.type in (ColorType.DEFAULT, ColorType.STANDARD)
                x += segment.cell_length
        for button in window.query(Button):
            assert button.display and button.region.width > 0
            assert 0 <= button.region.x and button.region.right <= 80
            assert 0 <= button.region.y and button.region.bottom <= 24
        await pilot.press("escape")
        await pilot.pause()
        assert app.focused is app.query_one("#ask", tui.Prompt)
        await pilot.press("a", "r", "i")
        assert app.screen.id != "folder-path-window"
        assert app.query_one("#ask", tui.Prompt).value == "ari"
        assert texts(app) == ""

    tui_run(lambda event: None, steps, size=(80, 24))


def test_api_save_is_local_and_role_selection_preserves_conversation_and_other_roles(home):
    """Connecting another endpoint for the embeddings: typing sends nothing; Enter requests only its model IDs
    (no service check, no prompt); the chosen ID becomes this role's, other roles and the conversation stay."""
    from types import SimpleNamespace as NS
    from textual.widgets import TabbedContent
    settings.save({"PERESEARCH_LLM_URL": "http://chat.test/v1", "PERESEARCH_LLM_MODEL": "chat-id",
                   "PERESEARCH_LLM_KEY": "chat-key"})
    checked, listed = [], []

    async def steps(app, pilot):
        app.agent = NS(conversation="current-conversation")
        await submit(app, pilot, "/model")
        parent = app.screen
        parent.discover_models = lambda entry: listed.append((entry.url, entry.key)) or ["embedding-id"]
        parent.query_one(TabbedContent).active = "role-embedding"
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
        listing = app.screen.query_one("#connect-list")
        listing.highlighted = [listing.get_option_at_index(i).id for i in range(listing.option_count)].index("other")
        await pilot.press("enter")
        await pilot.pause()
        api = app.screen
        hint = api.query_one("#window-hint", Static)
        assert hint.region.right <= 80 and hint.region.bottom <= 24      # its keys on screen at 80 x 24
        api.query_one("#api-url", Input).value = "http://embedding.test/v1"
        api.query_one("#api-key", Input).value = "embedding-key"
        await pilot.pause()
        assert listed == [] and not settings.get("PERESEARCH_EMBED_URL")   # typing sends and saves nothing
        await pilot.press("enter")
        await settle(app, pilot, lambda: app.screen.id == "pick-window")
        assert listed == [("http://embedding.test/v1", "embedding-key")]
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is parent and parent.role == "embedding"
        assert settings.get("PERESEARCH_EMBED_MODEL") == "embedding-id"
        assert settings.get("PERESEARCH_LLM_MODEL") == "chat-id"
        assert settings.get("PERESEARCH_LLM_KEY") == "chat-key"
        assert app.pending_resume == "current-conversation" and app.agent is None
        assert checked == []
        await pilot.press("escape")
        assert texts(app) == ""

    tui_run(lambda event: None, steps, size=(80, 24), model_check=lambda: checked.append("model"),
            endpoint_check=lambda: checked.append("endpoints"))


def test_settings_and_checks_keep_outputs_out_of_chat_and_do_not_overlap(home):
    checked = []
    settings.save({"PERESEARCH_SEARCH_MCP": "old-command", "JINA_API_KEY": "synthetic-kept-key"})

    async def steps(app, pilot):
        await submit(app, pilot, "/settings")
        window = app.screen
        window.query_one("#setting-pages", Input).value = "invalid"
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert settings.get("PERESEARCH_PAGES") == ""
        window.query_one("#setting-pages", Input).value = "4"
        window.query_one("#service-peresearch_search_mcp", Input).value = ""   # one list: no section to switch to
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert settings.get("PERESEARCH_PAGES") == "4"
        assert settings.get("PERESEARCH_SEARCH_MCP") == ""
        assert settings.get("JINA_API_KEY") == "synthetic-kept-key" and checked == []
        await pilot.press("escape")
        await submit(app, pilot, "/check")
        assert checked == []
        app.busy = True
        await pilot.press("r")
        assert checked == []
        app.busy = False
        await pilot.press("r")
        await settle(app, pilot, lambda: checked == ["model", "endpoints"] and not app._checking_connections)
        body = app.screen.query_one("#check-results", Markdown).source
        assert "fixture model" in body and "fixture endpoints" in body
        await pilot.press("escape")
        await pilot.pause()
        assert texts(app) == ""

    tui_run(lambda event: None, steps, model_check=lambda: checked.append("model") or (True, ["fixture model"]),
            endpoint_check=lambda: checked.append("endpoints") or ["fixture endpoints"])


def test_optional_model_can_be_disabled_inside_its_role(home):
    from textual.widgets import TabbedContent
    settings.save({"PERESEARCH_LLM_URL": "http://chat.test/v1", "PERESEARCH_LLM_MODEL": "chat-id",
                   "PERESEARCH_EMBED_URL": "http://embed.test/v1", "PERESEARCH_EMBED_MODEL": "embed-id",
                   "PERESEARCH_EMBED_KEY": "synthetic-key"})

    async def steps(app, pilot):
        await submit(app, pilot, "/model")
        app.screen.query_one(TabbedContent).active = "role-embedding"
        await pilot.pause()
        await pilot.press("o")
        await pilot.pause()
        assert settings.get("PERESEARCH_EMBED_URL") == ""
        assert settings.get("PERESEARCH_LLM_MODEL") == "chat-id"
        assert settings.get("PERESEARCH_EMBED_KEY") == "synthetic-key"
        await pilot.press("escape")
        assert texts(app) == ""

    tui_run(lambda event: None, steps)


def test_source_preview_is_floating_and_closing_releases_its_links(home):
    from peresearch.agent import Answer, Check
    from peresearch.tools import Source
    answer = Answer("A fact [N1].", {"N1": Source("N1", "file", "/n/a.md:L2", "a.md", "verbatim fact")}, Check(), 1)

    async def steps(app, pilot):
        app.show_source(answer.sources["N1"], answer)
        await pilot.pause()
        assert isinstance(app.screen, ModalScreen)
        page = app.screen.query_one(Markdown)
        assert "verbatim fact" in page.source and "open:N1" in page.source
        assert app.sources_of[page] is answer
        await pilot.press("escape")
        await pilot.pause()
        assert page not in app.sources_of
        assert texts(app) == ""

    tui_run(lambda event: None, steps)


def test_sessions_window_renames_exports_and_resumes_without_control_messages(home, notes, tmp_path):
    from textual.widgets import OptionList
    from tests.helpers import agent
    from peresearch.llm import Reply
    from peresearch.agent import history_file
    import json
    path = history_file("default")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"conversation": "saved", "time": 1000, "question": "Saved question", "answer": "Saved answer",
                               "sources": {}}) + "\n")
    destination = tmp_path / "export.md"

    async def steps(app, pilot):
        await submit(app, pilot, "/sessions")
        assert app.screen.id == "sessions-window"
        await pilot.press("enter")
        await pilot.pause()
        assert app.pending_resume == "saved" and "Saved answer" in texts(app)
        assert "Resumed the conversation" not in texts(app)
        await submit(app, pilot, "/sessions")
        await pilot.press("r")
        await pilot.pause()
        app.screen.query_one(Input).value = "Research session"
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert "Research session" in str(app.screen.query_one(OptionList).get_option_at_index(0).prompt)
        await pilot.press("e")
        await pilot.pause()
        app.screen.query_one(Input).value = str(destination)
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert "Saved question" in destination.read_text() and "Saved answer" in destination.read_text()
        assert "Saved to" not in texts(app)
        await pilot.press("n")
        await pilot.pause()
        assert app.pending_resume is None and app.last is None and app.exchanges == []
        assert texts(app) == ""

    tui_run(lambda event: agent(home, notes, Reply("unused"))[0], steps)


def test_folder_window_refuses_application_storage_and_busy_mutations(home, tmp_path):
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    async def steps(app, pilot):
        await submit(app, pilot, "/folders")
        window = app.screen
        app.busy = True
        await pilot.press("a")
        assert app.screen is window
        app.busy = False
        await pilot.press("a")
        await pilot.pause()
        app.screen.query_one(Input).value = str(home)
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert ws.folders() == []
        message = str(window.query_one("#window-message", Static).render()).lower()
        assert "refused" in message and "peresearch's own data" in message
        await pilot.press("escape")
        assert texts(app) == ""
    tui_run(lambda event: None, steps, workspace=ws)


def test_model_selection_explains_an_environment_override_in_its_window(home, monkeypatch):
    monkeypatch.setenv("PERESEARCH_LLM_MODEL", "environment-id")
    connections.add("chat", "Fixture", "http://model.test/v1", "", "selected-id")
    async def steps(app, pilot):
        await submit(app, pilot, "/model")
        await pilot.press("enter")
        await pilot.pause()
        assert settings.saved()["PERESEARCH_LLM_MODEL"] == "selected-id"
        assert settings.get("PERESEARCH_LLM_MODEL") == "environment-id"
        message = str(app.screen.query_one("#window-message", Static).render())
        assert "PERESEARCH_LLM_MODEL" in message and "environment" in message.lower()
        assert texts(app) == ""
    tui_run(lambda event: None, steps)


@pytest.mark.parametrize("has_folders", [True, False])
def test_automatic_index_and_empty_startup_notice_do_not_enter_research_transcript(home, tmp_path, has_folders):
    settings.save({"PERESEARCH_LLM_URL": "http://fixture.test/v1"})
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    if has_folders:
        folder = tmp_path / "documents"
        folder.mkdir()
        (folder / "note.md").write_text("A small research document with checkable local evidence.")
        ws.add(str(folder))
    async def steps(app, pilot):
        await settle(app, pilot, lambda: not app.indexing)
        assert texts(app) == ""
        assert "Index updated" in app.flash if has_folders else "/folders" in app.flash
    tui_run(lambda event: None, steps, workspace=ws, index_on_start=True)


def test_source_open_notice_stays_in_its_window(home, monkeypatch):
    from peresearch.agent import Answer, Check
    from peresearch.tools import Source
    opened = []
    monkeypatch.setattr(tui, "open_outside", lambda target, line: opened.append((target, line)))
    answer = Answer("Fact [N1].", {"N1": Source("N1", "file", "/n/a.md:L2", "a.md", "Fact")}, Check(), 1)
    async def steps(app, pilot):
        app.system_opened = []
        app.open_url = lambda url, **kwargs: app.system_opened.append(url)
        app.show_source(answer.sources["N1"], answer)
        await pilot.pause()
        window = app.screen
        page = window.query_one(Markdown)
        page.post_message(Markdown.LinkClicked(page, "open:N1"))
        await pilot.pause()
        assert "opened /n/a.md" in str(window.query_one("#window-message", Static).render())
        assert texts(app) == "" and app.system_opened == []
    tui_run(lambda event: None, steps)
    assert opened == [("/n/a.md", 2)]


def test_folder_window_reports_a_broken_live_config_and_recovers_without_losing_scope(home, tmp_path):
    import json
    ws = Workspace(Index(home, embedder=FakeEmbedder()))
    folder = tmp_path / "documents"
    folder.mkdir()
    ws.add(str(folder))
    path = home / "config.json"
    saved = path.read_bytes()
    async def steps(app, pilot):
        await submit(app, pilot, "/folders")
        window = app.screen
        path.write_text("{broken")
        await pilot.press("r")
        await pilot.pause()
        assert app.is_running and app.screen is window
        assert "not valid JSON" in str(window.query_one("#window-message", Static).render())
        assert path.read_text() == "{broken"
        path.write_bytes(saved)
        window.refresh_folders()
        assert ws.folders() == [folder.resolve()]
        await pilot.press("r")
        await settle(app, pilot, lambda: not app.indexing)
        assert ws.folders() == [] and texts(app) == ""
    tui_run(lambda event: None, steps, workspace=ws)


def test_model_window_keeps_its_untested_ids_warning_while_the_list_has_the_focus(home):
    """The warning that a listed ID is not tested for inference or tool calling stays on screen: the detail line
    of a window without fields is not cleared when its list takes the focus (found on the redesigned window)."""
    connections.add("chat", "Fixture", "http://model.test/v1", "", "model-id")

    async def steps(app, pilot):
        await submit(app, pilot, "/model")
        await pilot.pause()
        detail = app.screen.query_one("#window-detail", Static)
        app.warning = (detail.display, str(detail.render()))
    app = tui_run(lambda event: None, steps, size=(80, 24))
    shown, text = app.warning
    assert shown and "not been tested for inference or tool calling" in text
