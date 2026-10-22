"""Grouped TUI controls. A window owns its shortcuts and keeps control output out of the transcript.

The look follows Claude Code and OpenCode: a rounded panel sized to its content, a bold title, lists with a `❯`
pointer, one-line fields (label, then value), section names as plain words with the current one in the accent
color, a dim line of key hints at the bottom and no buttons: every action has its key."""

from copy import deepcopy
import io
import os

from rich.console import Console
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Markdown, Static, TabbedContent, TabPane
from textual.widgets.option_list import Option

from peresearch import connections, guard, providers, settings
from peresearch.widgets import Field, SelectList, hints, home_path, row

SHORT = {"PERESEARCH_SEARCH_MCP": "Web search server", "PERESEARCH_SEARCH_MCP_KEY": "Web search key",
         "PERESEARCH_SEARCH_TOOL": "Web search tool", "OPENALEX_API_KEY": "OpenAlex key",
         "SEMANTIC_SCHOLAR_API_KEY": "Semantic Scholar key", "JINA_API_KEY": "Jina Reader key"}


def display_text(text: str) -> str:
    """Hide configured credentials as well as recognized secret patterns in control output."""
    keys = {settings.get(name) for name, _, hidden in settings.FIELDS if hidden}
    keys.add(settings.get("JINA_API_KEY"))
    try:
        keys.update(entry.key for entry in connections.load())
    except (ValueError, OSError):
        pass
    for key in sorted((key for key in keys if key), key=len, reverse=True):
        text = text.replace(key, "[REDACTED]")
    return guard.sanitize(guard.redact(text)[0])


def short_label(name: str, label: str) -> str:
    return SHORT.get(name) or label.split(" (")[0].split(",")[0]


class Window(ModalScreen):
    # Shared selectors must match the screen itself, not descendants of each concrete screen type.
    SCOPED_CSS = False
    CSS = """
    /* An opaque backdrop: the conversation and a window below this one do not show through at its sides. */
    .control-window { align: center middle; background: ansi_default; }
    #window { width: 100%; max-width: 96; height: auto; max-height: 90%; border: round ansi_blue;
              padding: 0 1; background: ansi_default; }
    #window-title { height: 1; text-style: bold; margin-bottom: 1; }
    #window-body { height: auto; max-height: 70vh; scrollbar-size-vertical: 1; }
    .control-window.-medium #window { max-width: 72; }
    #window-message { height: auto; max-height: 3; color: ansi_yellow; }
    #window-detail { height: auto; max-height: 2; text-style: dim; margin-top: 1; }
    #window-hint { height: auto; text-style: dim; margin-top: 1; }
    #window .section { height: 1; margin-top: 1; text-style: bold; }
    #window .section.-first { margin-top: 0; }
    #window Markdown > *:last-child { margin-bottom: 0; }
    #window .columns { height: auto; }
    #window .empty { height: auto; text-style: dim; }
    #window Markdown { margin: 0; padding: 0; }
    #window TabbedContent { height: auto; }
    #window ContentSwitcher { height: auto; }
    #window TabPane { height: auto; padding: 0; }
    #window Tabs { height: 1; margin-bottom: 1; }
    #window Tabs:focus .-active { background: ansi_default; }
    #window Tab { height: 1; padding: 0 2 0 0; color: ansi_default; text-style: dim; }
    #window Tab.-active { color: ansi_blue; text-style: bold; }
    #window Tab:hover { text-style: bold; }
    #window Underline { display: none; }
    """
    BINDINGS = [Binding("escape", "close", show=False)]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_class("control-window")

    def title_bar(self, title: str):
        return Static(title, id="window-title", markup=False)

    def footer(self, *pairs, detail: bool = False):
        """The message line (empty until something happens), the focused field's note, the key hints."""
        line = Static("", id="window-message", markup=False)
        line.display = False
        yield line
        if detail:
            note = Static("", id="window-detail", classes="field-notes", markup=False)
            note.display = False                       # until a field with a note has the focus
            yield note
        yield Static(hints(*pairs), id="window-hint", markup=False)

    def message(self, text: str) -> None:
        if self.is_attached:
            line = self.query_one("#window-message", Static)
            line.update(display_text(text))
            line.display = bool(text)                  # no empty line kept for a message that is not there

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        field = event.widget.parent
        detail = self.query("#window-detail.field-notes")
        if not detail:                                 # a window's own detail text is never replaced
            return
        line = detail.first(Static)
        shown = isinstance(field, Field) and field.note_text and not field.narrow   # a narrow one: on its own line
        line.update(display_text(field.note_text) if shown else "")
        line.display = bool(shown)

    def action_close(self) -> None:
        self.dismiss(None)

    def action_section(self, step: int) -> None:
        """← / →: the previous or next section of a window with sections."""
        found = self.query(TabbedContent)
        if not found:
            return
        tabs = found.first(TabbedContent)
        panes = [pane.id for pane in tabs.query(TabPane)]
        if panes and tabs.active in panes:
            tabs.active = panes[(panes.index(tabs.active) + step) % len(panes)]

    def apply_settings(self, changed, values) -> None:
        self.app.run_control(lambda: changed(values), self.message)
        overrides = [name for name in values if os.environ.get(name)]
        if overrides:
            self.message("Saved locally. The environment overrides " + ", ".join(overrides) +
                         "; the effective configuration still uses those environment values.")


class Columns(Static):
    """Aligned columns (a Rich table: /help's commands and keys); `source` is their plain text, as Markdown's."""

    def __init__(self, table):
        super().__init__(table, classes="columns")
        console = Console(width=200, record=True, file=io.StringIO())
        console.print(table)
        self.source = console.export_text()


class InfoWindow(Window):
    BINDINGS = Window.BINDINGS + [Binding("left", "section(-1)", show=False), Binding("right", "section(1)", show=False)]

    def __init__(self, title: str, text: str, tabs: dict[str, str] | None = None):
        super().__init__()
        self.title_text, self.text, self.tabs = title, text, tabs

    def compose(self) -> ComposeResult:
        with Vertical(id="window"):
            yield self.title_bar(self.title_text)
            with VerticalScroll(id="window-body"):
                if self.tabs:
                    with TabbedContent():
                        for title, text in self.tabs.items():
                            with TabPane(title):
                                yield (Markdown(display_text(text), open_links=False) if isinstance(text, str)
                                       else Columns(text))
                else:
                    yield Markdown(display_text(self.text), id="window-content", open_links=False)
            yield from self.footer(*([("←→", "section")] if self.tabs else []), ("↑↓", "scroll"), ("esc", "close"))


class TextEntry(Window):
    BINDINGS = [Binding("escape", "close", show=False), Binding("ctrl+s", "save", show=False)]

    def __init__(self, title: str, label: str, value: str = "", id: str | None = None):
        super().__init__(id=id)
        self.title_text, self.label, self.value = title, label, value

    def compose(self) -> ComposeResult:
        with Vertical(id="window"):
            yield self.title_bar(self.title_text)
            with Vertical(id="window-body"):
                yield Field(self.label, self.value, id="entry-value")
            yield from self.footer(("enter", "save"), ("esc", "cancel"))

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.action_save()

    def action_save(self) -> None:
        self.dismiss(self.query_one(Input).value.strip())


class SourceWindow(InfoWindow):
    def __init__(self, text, answer):
        super().__init__("Source", text)
        self.answer = answer

    def on_mount(self) -> None:
        self.page = self.query_one(Markdown)
        self.app.sources_of[self.page] = self.answer

    def on_unmount(self) -> None:
        self.app.sources_of.pop(self.page, None)


class FolderWindow(Window):
    BINDINGS = Window.BINDINGS + [Binding("a", "add", show=False), Binding("r", "remove", show=False),
                                 Binding("i", "index", show=False)]

    def __init__(self, workspace):
        super().__init__(id="folders-window")
        self.workspace, self.paths = workspace, []

    def compose(self) -> ComposeResult:
        with Vertical(id="window"):
            yield self.title_bar("Folders")
            with Vertical(id="window-body"):
                yield Static("peresearch reads only these folders and keeps an index of them.", classes="empty",
                             markup=False)
                yield SelectList(id="folder-list")
            yield from self.footer(("a", "add"), ("r", "remove selected"), ("i", "update index"), ("esc", "close"))

    def on_mount(self) -> None:
        self.refresh_folders()
        self.query_one(SelectList).focus()

    def refresh_folders(self) -> None:
        try:
            self.paths = self.workspace.folders() if self.workspace else []
            listing = self.query_one(SelectList)
            listing.clear_options()
            listing.add_options([Option(row(guard.sanitize(home_path(str(path)))), id=str(i))
                                 for i, path in enumerate(self.paths)])
            listing.display = bool(self.paths)
            if self.paths:
                listing.highlighted = 0
            elif not str(self.query_one("#window-message", Static).render()):
                self.message("No folders yet. Press a to add one.")
        except guard.ConfigError as error:
            self.message(str(error))

    def _ready(self) -> bool:
        if self.workspace is None:
            self.message("Folders are unavailable in this session.")
            return False
        if self.app.busy or self.app.indexing or self.app._checking_connections:
            self.message("Wait for the current operation to finish before changing folders.")
            return False
        return True

    def action_add(self) -> None:
        if self._ready():
            self.app.push_screen(TextEntry("Add folder", "Folder path", id="folder-path-window"), self._add)

    def _add(self, path) -> None:
        if path and self._ready():
            self.app.run_control(lambda: self.app.action_command("/add " + path), self.message)
            self.refresh_folders()

    def action_remove(self) -> None:
        selected = self.query_one(SelectList).highlighted
        if self._ready() and selected is not None and selected < len(self.paths):
            self.app.run_control(lambda: self.app.action_command("/remove " + str(self.paths[selected])),
                                 self.message)
            self.refresh_folders()

    def action_index(self) -> None:
        if self._ready():
            self.app.run_control(lambda: self.app.action_command("/index"), self.message)

    def on_option_list_option_selected(self, event: SelectList.OptionSelected) -> None:
        event.stop()


class ConnectWindow(Window):
    """Step 1 of connecting a model: where it runs — a connection already saved for this role, a cloud, a server on
    this machine, or a provider the user adds (its name, protocol, endpoint and key)."""

    def __init__(self, role: str, discover, chosen):
        super().__init__(id="connect-window")
        self.add_class("-medium")                    # a short step: OpenCode's medium width
        self.role, self.discover_models, self.chosen = role, discover, chosen

    def compose(self) -> ComposeResult:
        with Vertical(id="window"):
            yield self.title_bar("Connect a model · " + connections.ROLES[self.role][0])
            with Vertical(id="window-body"):
                yield SelectList(id="connect-list")
            yield from self.footer(("↑↓", "choose"), ("enter", "next"), ("esc", "back"))

    def on_mount(self) -> None:
        listing = self.query_one(SelectList)
        try:
            saved = [entry for entry in connections.load() if entry.role == self.role]
        except (ValueError, OSError) as error:
            saved = []
            self.message(str(error))
        rows = []
        if saved:
            rows.append(Option(Text("Connected", style="bold"), disabled=True))
            for entry in saved:
                host = display_text(connections.host(entry.url))
                name = display_text(entry.name)
                rows.append(Option(row(Text.assemble(name, ("" if name == host else "  " + host, "dim"),
                                                     (f"  {len(entry.models)} model IDs" if entry.models else "",
                                                      "dim"))), id="entry:" + entry.id))
        offered = [p for p in providers.PROVIDERS if self.role in p.roles]      # where this role can run
        width = max((len(p.name) for p in offered if p.kind == "local"), default=0)
        for heading, kind in (("Cloud", "cloud"), ("On this machine", "local")):
            rows.append(Option(Text(heading, style="bold"), disabled=True))
            for p in offered:
                if p.kind == kind:                     # a local server's address in a column of its own
                    where = "" if kind == "cloud" else "  " + p.url.split("//")[1].removesuffix("/v1")
                    rows.append(Option(row(Text.assemble(p.name.ljust(width) if where else p.name, (where, "dim"))),
                                       id="preset:" + p.id))
        rows.append(Option(row("Add a provider…"), id="other"))
        listing.add_options(rows)
        listing.highlighted = next(i for i, option in enumerate(rows) if not option.disabled)
        listing.focus()

    def on_option_list_option_selected(self, event: SelectList.OptionSelected) -> None:
        event.stop()
        choice = event.option.id
        if choice.startswith("entry:"):
            entry = next((e for e in connections.load() if e.id == choice[6:]), None)
            if entry is not None:                      # its models listed again; the saved list if that fails
                self.app.push_screen(CredentialWindow(self.role, entry.name, entry.url, "saved", self.discover_models,
                                                      self.chosen, saved=entry))
        elif choice.startswith("preset:"):
            p = providers.get(choice[7:])
            self.app.push_screen(CredentialWindow(self.role, p.name, p.url, p.kind, self.discover_models, self.chosen,
                                                  provider=p.id))
        else:
            self.app.push_screen(CredentialWindow(self.role, "", "", "other", self.discover_models, self.chosen))


class CredentialWindow(Window):
    """Step 2: only what the place needs — a cloud's API key, a local server's address, or for a provider the user
    adds its name, protocol (chat models only: embeddings and rerankers speak one), endpoint and key. Enter saves the
    connection and requests its model IDs (no prompt is sent); a list that cannot be had leaves a known ID to enter."""

    BINDINGS = Window.BINDINGS + [Binding("ctrl+s", "connect", show=False), Binding("ctrl+t", "enter_id", show=False, priority=True)]

    def __init__(self, role: str, name: str, url: str, kind: str, discover, chosen, saved=None, provider: str = ""):
        super().__init__(id="api-window")
        self.provider = provider
        self.add_class("-medium")                    # a short step: OpenCode's medium width
        self.role, self.place, self.url, self.kind = role, name, url, kind
        self.discover_models, self.chosen = discover, chosen
        self.entry, self._generation = saved, 0

    def compose(self) -> ComposeResult:
        title = self.place or "Add a provider"
        with Vertical(id="window"):
            yield self.title_bar(title + " · " + connections.ROLES[self.role][0])
            with Vertical(id="window-body"):
                if self.kind == "other":
                    yield Field("Name", id="api-name", placeholder="how the list shows it")
                    if self.role == "chat":
                        yield Field("Protocol", id="api-protocol", placeholder="chat",
                                    note="What it speaks: chat (OpenAI Chat Completions and compatible servers), "
                                         "responses (OpenAI), anthropic or gemini.")
                if self.kind in ("local", "other"):
                    yield Field("Endpoint", self.url, id="api-url", placeholder="http://…/v1",
                                note="Where the server answers." if self.kind == "other" else
                                     "Where the server answers: an OpenAI-compatible endpoint ending in /v1.")
                if self.kind in ("cloud", "other"):
                    yield Field("API key", id="api-key", password=True,
                                placeholder="required" if self.kind == "cloud" else "empty: no authentication",
                                note="Kept in a private file and never shown again. API billing is separate from "
                                     "chat subscriptions.")
                if self.kind == "saved":
                    yield Static("Saved connection: " + display_text(connections.host(self.url)), markup=False)
            yield from self.footer(("enter", "connect and list models"), ("ctrl+t", "type a model ID"),
                                   ("esc", "back"), detail=True)

    def on_mount(self) -> None:
        inputs = self.query(Input)
        (inputs.last() if inputs else self).focus()
        if self.kind == "saved":                       # chosen in step 1: that was the request to list
            self.action_connect()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.action_connect()

    def _value(self, name: str) -> str:
        found = self.query("#api-" + name)
        return found.first(Input).value.strip() if found else ""

    def _save(self):
        url = self._value("url") if self.kind in ("local", "other") else self.url
        key = self._value("key")
        if self.kind == "cloud" and not key:
            raise ValueError("Enter the API key of " + self.place + ".")
        if self.kind == "saved":
            return next(e for e in connections.load() if e.id == self.entry.id)
        if self.kind == "other":
            protocol = (self._value("protocol") or "chat") if self.role == "chat" else ""
            return connections.connect(self.role, self._value("name"), url, key, protocol=protocol)
        return connections.connect(self.role, self.place, url, key, provider=self.provider)

    def action_connect(self) -> None:
        try:
            self.entry = self._save()
        except (ValueError, OSError, StopIteration) as error:
            self.message(str(error) or "The connection is gone; choose it again.")
            return
        entry, app = deepcopy(self.entry), self.app
        self._generation += 1
        generation = self._generation
        self.message("Listing model IDs from " + connections.host(entry.url) + "…")

        def run():
            try:
                models = self.discover_models(entry)
                app._ui(self._listed, entry, models, generation)
            except Exception as error:
                app._ui(self._failed, entry, str(error), generation)
        app._in_background(run)

    def _current(self, generation) -> bool:
        return self.is_attached and self.app.screen is self and generation == self._generation

    def _listed(self, entry, models, generation) -> None:
        if not self._current(generation):
            return
        try:
            connections.update_models(entry.id, models, entry)
        except (ValueError, OSError) as error:
            self.message(str(error))
            return
        if not models:
            self.message("The endpoint lists no model. Press ctrl+t to type a model ID.")
            return
        self.message("")
        self.app.push_screen(PickWindow(entry, models, self.chosen))

    def _failed(self, entry, error, generation) -> None:
        if self._current(generation):
            error = display_text(error.replace(entry.key, "[REDACTED]") if entry.key else error)
            if self.kind == "saved" and entry.models:  # the IDs listed before stay usable
                self.app.push_screen(PickWindow(entry, entry.models, self.chosen,
                                                note="Could not list the models again: " + error +
                                                     ". These are the IDs listed before."))
                return
            self.message("Could not list the models: " + error + ". Check the address and key, or press ctrl+t to "
                         "type a model ID.")

    def action_close(self) -> None:
        self._generation += 1                        # a list still on its way is not applied
        super().action_close()

    def action_enter_id(self) -> None:
        try:
            self.entry = self.entry or self._save()
        except (ValueError, OSError, StopIteration) as error:
            self.message(str(error) or "The connection is gone; choose it again.")
            return
        entry = deepcopy(self.entry)
        self.app.push_screen(TextEntry("Model ID · " + display_text(entry.name), "Model ID", id="model-id-window"),
                             lambda model: self._entered(entry, model))

    def _entered(self, entry, model) -> None:
        if model:
            try:
                connections.update_models(entry.id, list(dict.fromkeys(entry.models + [model])), entry)
            except (ValueError, OSError) as error:
                self.message(str(error))
                return
            self.chosen(entry, model)


def capability(model: str, role: str) -> Text:
    """What the pinned models.dev snapshot (profiles.py) says of a chat model: tool calls and context, or that it
    does not know it. A note only: a listed model is never hidden, and the endpoint, not the table, decides."""
    if role != "chat":
        return Text("")
    from peresearch import profiles
    try:
        found = profiles.lookup(model)
    except (OSError, ValueError):
        found = None
    if found is None:
        return Text("not in the catalog", style="dim")
    _, tool_call, _, _, context, _ = found[1]
    size = f" · {context // 1000:,}k context" if context else ""
    return Text.assemble(("tools" if tool_call else "no tool calls", "dim" if tool_call else "yellow"), (size, "dim"))


class PickWindow(Window):
    """Step 3: the model, chosen from the IDs the endpoint listed; typing filters them."""

    BINDINGS = Window.BINDINGS + [Binding("up", "move(-1)", show=False), Binding("down", "move(1)", show=False)]

    def __init__(self, entry, models: list[str], chosen, note: str = ""):
        super().__init__(id="pick-window")
        self.add_class("-medium")                    # a short step: OpenCode's medium width
        self.entry, self.models, self.chosen, self.note = entry, list(models), chosen, note

    def compose(self) -> ComposeResult:
        with Vertical(id="window"):
            yield self.title_bar(display_text(self.entry.name) + " · choose a model")
            with Vertical(id="window-body"):
                yield Field("Filter", id="pick-filter", placeholder=f"type to filter {len(self.models)} model IDs")
                yield SelectList(id="pick-list")
            yield Static("A listed ID is not yet tested for answering or tool calling; /check tests the chosen one.",
                         id="window-detail", markup=False)
            yield from self.footer(("type", "filter"), ("↑↓", "choose"), ("enter", "use"), ("esc", "back"))

    def on_mount(self) -> None:
        self.show("")
        self.query_one("#pick-filter", Input).focus()

    def show(self, text: str) -> None:
        listing = self.query_one(SelectList)
        listing.clear_options()
        words = text.lower().split()
        shown = [m for m in self.models if all(word in m.lower() for word in words)]
        width = min(max((len(m) for m in shown), default=0), 40)
        listing.add_options([Option(row(Text.assemble(display_text(m).ljust(width), "  ",
                                                      capability(m, self.entry.role))), id=m) for m in shown])
        listing.highlighted = 0 if shown else None
        self.message(self.note if shown else "No listed ID matches.")

    def on_input_changed(self, event: Input.Changed) -> None:
        self.show(event.value)

    def action_move(self, step: int) -> None:
        listing = self.query_one(SelectList)
        (listing.action_cursor_down if step > 0 else listing.action_cursor_up)()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        listing = self.query_one(SelectList)
        if listing.highlighted is not None:
            self.chosen(self.entry, listing.get_option_at_index(listing.highlighted).id)

    def on_option_list_option_selected(self, event: SelectList.OptionSelected) -> None:
        event.stop()
        self.chosen(self.entry, event.option.id)


class ModelWindow(Window):
    BINDINGS = Window.BINDINGS + [Binding("a", "add", show=False), Binding("o", "off", show=False),
                                 Binding("left", "section(-1)", show=False), Binding("right", "section(1)", show=False)]

    def __init__(self, changed, discover=None):
        super().__init__(id="model-window")
        self.changed, self.discover_models = changed, discover or connections.discover
        self.choices, self._generation = {}, 0

    def compose(self) -> ComposeResult:
        with Vertical(id="window"):
            yield self.title_bar("Models and APIs")
            with Vertical(id="window-body"):
                with TabbedContent(initial="role-chat", id="model-roles"):
                    for role, (label, _) in connections.ROLES.items():
                        with TabPane(label, id="role-" + role):
                            yield SelectList(id="models-" + role)
            yield Static("Model IDs listed here have not been tested for inference or tool calling; /check tests "
                         "the chosen ones.", id="window-detail", markup=False)
            yield from self.footer(("←→", "role"), ("enter", "use"), ("a", "connect a model"),
                                   ("o", "turn off"), ("esc", "close"))

    @property
    def role(self) -> str:
        return self.query_one("#model-roles", TabbedContent).active.removeprefix("role-")

    def on_mount(self) -> None:
        try:
            connections.adopt_saved_selection()
        except (ValueError, OSError) as error:
            self.message(str(error))
        self.refresh_models()
        self.query_one("#models-chat", SelectList).focus()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self._generation += 1
        if self.is_mounted:
            self.query_one("#models-" + self.role, SelectList).focus()
            self.message(self.role_note())

    def refresh_models(self, prefer: str | None = None) -> None:
        try:
            entries = connections.load()
        except (ValueError, OSError) as error:
            self.message(str(error))
            return
        self.choices = {}
        for role, (_, prefix) in connections.ROLES.items():
            pairs = [(entry, model) for entry in entries if entry.role == role for model in entry.models or [""]]
            self.choices[role] = pairs
            listing = self.query_one("#models-" + role, SelectList)
            listing.clear_options()
            for i, (entry, model) in enumerate(pairs):
                active = settings.get(prefix + "_URL") == entry.url and settings.get(prefix + "_MODEL") == model \
                    and settings.get(prefix + "_KEY") == entry.key
                host = display_text(connections.host(entry.url))
                where = display_text(entry.name) + ("" if entry.name == host else " · " + host)   # a name, then where
                title = Text.assemble((display_text(model) or "no model ID yet: a, then this connection"),
                                      "  ", (where, "dim"), ("  ✔ in use" if active else "", "green"))
                listing.add_option(Option(row(title), id=str(i)))
            listing.display = bool(pairs)               # no empty row where nothing is connected
            if pairs:
                listing.highlighted = next((i for i, (entry, _) in enumerate(pairs) if entry.id == prefer), 0)
        note = self.role_note()
        if note:                                       # a message already shown (an override, an error) stays
            self.message(note)

    def role_note(self) -> str:
        """What the message line says of this role: an endpoint in use that no saved row shows (it comes from the
        environment, which is never saved), or that nothing is connected yet."""
        prefix = connections.ROLES[self.role][1]
        url = settings.get(prefix + "_URL")
        shown = any(entry.url == url for entry, _ in self.choices.get(self.role, []))
        if url and not shown:
            model = settings.get(prefix + "_MODEL")
            return ("In use from the environment: " + connections.host(url) + (" · " + model if model else "") +
                    ". Press a to connect another.")
        if self.choices.get(self.role):
            return ""
        if self.role == "embedding":                   # the search without it: words only (REFERENCE.md, measured)
            return ("No embeddings: your files are searched by their words only, not by meaning. Press a to connect "
                    "an embedding server, e.g. Ollama or LM Studio on this machine.")
        if self.role == "rerank":
            return "No reranker (optional): results keep the search's own order. Press a to connect one."
        return "No model connected yet. Press a to connect one."

    def selected(self):
        pairs = self.choices.get(self.role, [])
        i = self.query_one("#models-" + self.role, SelectList).highlighted
        return pairs[i] if i is not None and i < len(pairs) else None

    def on_option_list_option_selected(self, event: SelectList.OptionSelected) -> None:
        event.stop()
        selected = self.selected()
        if not selected:
            return
        entry, model = selected
        if not model:
            self.message("List model IDs with d or enter an ID with m before selecting it.")
            return
        if self.app.busy or self.app.indexing or self.app._checking_connections:
            self.message("Wait for the current operation to finish before changing models.")
            return
        try:
            self.apply_settings(self.changed, connections.selection(entry, model))
        except (ValueError, OSError) as error:
            self.message(str(error))
        self.refresh_models(entry.id)

    def action_add(self) -> None:
        self.app.push_screen(ConnectWindow(self.role, lambda entry: self.discover_models(entry), self.choose))

    def choose(self, entry, model: str) -> None:
        """The end of connecting: the chosen model becomes this role's, and the windows of the steps close."""
        while self.app.screen is not self and len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        if self.app.busy or self.app.indexing or self.app._checking_connections:
            self.message("Wait for the current operation to finish before changing models.")
            return
        try:
            self.apply_settings(self.changed, connections.selection(entry, model))
        except (ValueError, OSError) as error:
            self.message(str(error))
        self.refresh_models(entry.id)

    def action_off(self) -> None:
        if self.role == "chat":
            self.message("A chat model is required to answer questions.")
        elif self.app.busy or self.app.indexing or self.app._checking_connections:
            self.message("Wait for the current operation to finish before changing models.")
        else:
            prefix = connections.ROLES[self.role][1]
            try:
                self.apply_settings(self.changed, {prefix + "_URL": "", prefix + "_MODEL": ""})
                self.refresh_models()
            except (ValueError, OSError) as error:
                self.message(str(error))


class SettingsWindow(Window):
    """Research limits and search services in one list, as Claude Code's /config reads: each a line with its name,
    its value and, dimmed, what it does."""

    BINDINGS = Window.BINDINGS + [Binding("ctrl+s", "save", show=False)]

    def __init__(self, changed):
        super().__init__(id="settings-window")
        self.changed, self.current = changed, settings.saved()
        prefixes = tuple(prefix for _, prefix in connections.ROLES.values())
        self.services = [field for field in settings.FIELDS if not field[0].startswith(prefixes)]
        self.services += [("JINA_API_KEY", "Jina Reader key (optional; renders pages a plain read cannot)", True)]

    def compose(self) -> ComposeResult:
        with Vertical(id="window"):
            yield self.title_bar("Settings")
            with VerticalScroll(id="window-body"):
                yield Static("Research limits", classes="section -first", markup=False)
                for key, name, default, description, kind in settings.CONFIG:
                    yield Field(key, self.current.get(name, ""), id="setting-" + key, placeholder=default,
                                narrow=True, note=description)
                yield Static("Search services", classes="section", markup=False)
                for name, label, hidden in self.services:
                    state = ("set: empty keeps it" if self.current.get(name) else "not set") if hidden else "not set"
                    yield Field(short_label(name, label), "" if hidden else self.current.get(name, ""),
                                id="service-" + name.lower(), password=hidden, placeholder=state,
                                note=label[:1].upper() + label[1:])
            yield from self.footer(("tab", "next"), ("enter", "save"), ("empty", "default"), ("esc", "close"),
                                   detail=True)

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.action_save()

    def action_save(self) -> None:
        if self.app.busy or self.app.indexing or self.app._checking_connections:
            self.message("Wait for the current operation to finish before changing settings.")
            return
        changes = {}
        try:
            for key, name, _, _, _ in settings.CONFIG:
                value = self.query_one("#setting-" + key, Input).value.strip()
                value = settings.check(key, value)[1] if value else ""
                if value != self.current.get(name, ""):
                    changes[name] = value
            for name, _, hidden in self.services:
                value = self.query_one("#service-" + name.lower(), Input).value.strip()
                if (value or not hidden) and value != self.current.get(name, ""):
                    changes[name] = value
            self.apply_settings(self.changed, changes)
            self.current = settings.saved()
        except (ValueError, OSError) as error:
            self.message(str(error))


class SessionWindow(Window):
    BINDINGS = Window.BINDINGS + [Binding("n", "new", show=False), Binding("r", "rename", show=False),
                                 Binding("e", "export", show=False)]

    def __init__(self):
        super().__init__(id="sessions-window")
        self.past = []

    def compose(self) -> ComposeResult:
        with Vertical(id="window"):
            yield self.title_bar("Conversations")
            with Vertical(id="window-body"):
                yield SelectList(id="session-list")
            yield from self.footer(("enter", "resume"), ("n", "new"), ("r", "rename current"),
                                   ("e", "export current"), ("esc", "close"))

    def on_mount(self) -> None:
        self.refresh_sessions()
        self.query_one(SelectList).focus()

    def refresh_sessions(self) -> None:
        from peresearch.agent import conversations
        from peresearch.tui import conversation_rows

        self.past = conversations(self.app.project)
        listing = self.query_one(SelectList)
        listing.clear_options()
        width = (listing.size.width or max(8, min(96, self.app.size.width) - 4)) - 2
        listing.add_options([Option(row(text), id=item.id) for item, text in
                             zip(self.past, conversation_rows(self.past, width))])
        listing.display = bool(self.past)
        if self.past:
            listing.highlighted = 0
        else:
            self.message("No saved conversations yet. n starts a new one.")

    def _ready(self) -> bool:
        if self.app.busy or self.app._checking_connections:
            self.message("Finish or interrupt the current question or connection checks first.")
            return False
        return True

    def on_option_list_option_selected(self, event: SelectList.OptionSelected) -> None:
        event.stop()
        if self._ready():
            self.app.run_control(lambda: self.app.resume(event.option.id), self.message)
            self.action_close()

    def action_new(self) -> None:
        if self._ready():
            self.app.run_control(lambda: self.app.command("/new"), self.message)
            self.action_close()

    def action_rename(self) -> None:
        if self._ready():
            self.app.push_screen(TextEntry("Rename conversation", "New name"), self._rename)

    def _rename(self, name) -> None:
        if name and self._ready():
            self.app.run_control(lambda: self.app.name_conversation(name), self.message)
            self.refresh_sessions()

    def action_export(self) -> None:
        self.app.push_screen(TextEntry("Export conversation", "File or folder"), self._export)

    def _export(self, path) -> None:
        if path is not None:
            self.app.run_control(lambda: self.app.export(path), self.message)


class ChecksWindow(Window):
    BINDINGS = Window.BINDINGS + [Binding("r", "run", show=False)]

    def __init__(self):
        super().__init__(id="checks-window")
        self.results = []

    def compose(self) -> ComposeResult:
        with Vertical(id="window"):
            yield self.title_bar("Connection checks")
            with VerticalScroll(id="window-body"):
                yield Markdown("Running sends test requests to the configured model and search services. Remote "
                               "usage may be charged. Your documents are not sent.", id="check-results",
                               open_links=False)
            yield from self.footer(("r", "run checks"), ("esc", "close"))

    def _result(self, text) -> None:
        if self.is_attached:
            self.results.append(display_text(text))
            self.query_one("#check-results", Markdown).update("\n\n".join(self.results))

    def action_run(self) -> None:
        if self.app.busy or self.app.indexing or self.app._checking_connections:
            self.message("Wait for the current operation to finish.")
        else:
            self.app.run_control(lambda: self.app.check_connections(True, self._result), self.message)
