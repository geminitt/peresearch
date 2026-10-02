"""Contract of the input box (peresearch.tui.Prompt), checked as a property against a reference model.

The box is driven with any sequence of typing, pastes of one or several lines (some carrying the invisible mark or a
copied marker), Backspace, ←/→, Home/End, word jumps, new lines and Enter. The reference holds the text as atoms —
characters and paste markers — so that, whatever happens: a paste of several lines shows as one marker; a marker is
one block (the cursor steps over it, never stops inside it, one Backspace removes all of it, typing at its edge lands
outside it); the invisible mark is never typed or pasted in; and Enter sends the text with every marker expanded to
its paste, then empties the box. The real terminal path (bytes from a terminal and an input method) is in
test_tui_input.py.
"""

import asyncio

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from textual import events
from textual.app import App

from peresearch.tui import MARK, Prompt


class Marker:
    def __init__(self, n, text):
        self.text = text
        self.shown = f"[Pasted text #{n} +{text.rstrip(chr(10)).count(chr(10)) + 1} lines]{MARK}"


class Model:
    """The reference: atoms (one character, or one Marker) and a cursor between atoms."""

    def __init__(self):
        self.atoms, self.cursor, self.pastes = [], 0, 0

    def shown(self):
        return "".join(a.shown if isinstance(a, Marker) else a for a in self.atoms)

    def insert(self, items):
        self.atoms[self.cursor:self.cursor] = items
        self.cursor += len(items)

    def location(self):
        """(row, column) of the cursor in the shown text."""
        before = "".join(a.shown if isinstance(a, Marker) else a for a in self.atoms[:self.cursor])
        return before.count("\n"), len(before.rsplit("\n", 1)[-1])

    def at(self, location):
        """The atom index at a (row, column) of the shown text; None if it falls inside a marker."""
        row, col, r, c = *location, 0, 0
        for i, a in enumerate(self.atoms + [None]):
            if (r, c) == (row, col):
                return i
            if a is None:
                return None
            if isinstance(a, Marker):
                if r == row and c < col < c + len(a.shown):
                    return None
                c += len(a.shown)
            elif a == "\n":
                r, c = r + 1, 0
            else:
                c += 1
        return None

    def sent(self):
        text = "".join("\n" + a.text.rstrip("\n") + "\n" if isinstance(a, Marker) else a for a in self.atoms)
        return text.replace(MARK, "").strip()


typed = st.text(alphabet="ab xyz09ư机", min_size=1, max_size=4)
pasted = st.one_of(st.text(alphabet="pq rX\n\r", min_size=1, max_size=12),
                   st.just("one\r\ntwo\r\n"), st.just("a\nb  \n"), st.just("old\rmac\rlines"), st.just(f"copied [Pasted text #1 +2 lines]{MARK} back"),
                   st.just(f"x{MARK}y"))
ops = st.lists(st.one_of(st.tuples(st.just("type"), typed), st.tuples(st.just("paste"), pasted),
                         st.sampled_from([("key", "backspace"), ("key", "left"), ("key", "right"), ("key", "home"),
                                          ("key", "end"), ("key", "ctrl+left"), ("key", "ctrl+right"),
                                          ("key", "ctrl+j"), ("key", "enter")])), max_size=14)


class Box(App):
    def __init__(self):
        super().__init__()
        self.sent = []

    def compose(self):
        yield Prompt(id="ask")

    def menu_open(self):
        return False

    def action_up(self):
        pass

    def action_down(self):
        pass

    def on_prompt_submitted(self, event):
        self.sent.append(event.prompt.expand(event.value))


KEY_NAMES = {" ": "space"}


@given(ops)
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@example([("paste", "Traceback\n  File x\nKeyError: 'a'\n"), ("type", " tail"), ("key", "left"), ("key", "left"),
          ("key", "left"), ("key", "left"), ("key", "left"), ("key", "left"), ("key", "right"), ("type", "x"),
          ("key", "end"), ("key", "left"), ("key", "left"), ("key", "left"), ("key", "left"), ("key", "left"),
          ("key", "left"), ("key", "backspace"), ("key", "enter")])
@example([("type", "typed [Pasted text #1 +3 lines] by hand"), ("key", "enter")])     # typed is not a marker
@example([("paste", "a\nb\n"), ("paste", f"copied [Pasted text #1 +2 lines]{MARK} back")])   # a forged one is text
@example([("paste", "a\nb"), ("key", "left"), ("type", "x"), ("key", "enter")])     # typed at its start: before it
@example([("type", "s"), ("paste", "a\rb\rc"), ("type", "e"), ("key", "enter")])     # old Mac line ends
@example([("type", "s"), ("paste", "a\nb  \n"), ("type", "e"), ("key", "enter")])    # spaces ending a paste stay
@example([("type", "s"), ("paste", "a\nX\n"), ("type", "e"), ("key", "enter")])
def test_the_input_box_keeps_its_contract_whatever_is_typed_or_pasted(ops):
    asyncio.run(drive(ops))


async def drive(ops):
    app, model = Box(), Model()
    async with app.run_test(size=(100, 12)) as pilot:
        prompt = app.query_one(Prompt)
        await pilot.pause()
        for kind, value in ops:
            if kind == "type":
                await pilot.press(*[KEY_NAMES.get(c, c) for c in value])
                model.insert(list(value))
            elif kind == "paste":
                prompt.post_message(events.Paste(value))
                await pilot.pause()
                text = value.replace("\r\n", "\n").replace("\r", "\n")
                if text.rstrip("\n").count("\n"):
                    model.pastes += 1
                    model.insert([Marker(model.pastes, text)])
                else:
                    model.insert(list(text.replace(MARK, "")))
            else:
                await pilot.press(value)
                if value == "backspace" and model.cursor:
                    del model.atoms[model.cursor - 1]
                    model.cursor -= 1
                elif value == "left":
                    model.cursor = max(0, model.cursor - 1)
                elif value == "right":
                    model.cursor = min(len(model.atoms), model.cursor + 1)
                elif value == "ctrl+j":
                    model.insert(["\n"])
                elif value == "enter":
                    assert app.sent[-1] == model.sent()
                    model.atoms, model.cursor = [], 0
                elif value in ("home", "end", "ctrl+left", "ctrl+right"):  # not modelled: follow the box, if valid
                    at = model.at(prompt.cursor_location)
                    assert at is not None, ("the cursor stopped inside a marker", value, prompt.cursor_location)
                    model.cursor = at
            await pilot.pause()
            assert prompt.text == model.shown(), (kind, value)
            assert prompt.cursor_location == model.location(), (kind, value)
            assert prompt.text.count(MARK) == sum(isinstance(a, Marker) for a in model.atoms)


def test_keys_arriving_in_one_burst_keep_their_order():
    """A terminal read can deliver typed characters and keys bound to an action (Ctrl+U, arrows, Enter, Backspace)
    together; Textual inserted the characters at once and ran the actions later, so "abc ⌃U xyz" came out empty and
    an input method's "chao ⌫⌫ ào" as "chaoào". The same bursts through the real driver are in test_tui_input.py;
    here they are posted in process, where mutation runs can see them."""
    bursts = [["a", "b", "c", "ctrl+u", "x", "y", "z", "enter"], ["a", "b", "left", "c", "enter"],
              ["o", "n", "e", "enter", "t", "w", "o", "enter"],
              ["c", "h", "a", "o", "backspace", "backspace", "à", "o", "enter"]]

    async def run():
        app = Box()
        async with app.run_test() as pilot:
            prompt = app.query_one(Prompt)
            await pilot.pause()
            for burst in bursts:
                for k in burst:
                    prompt.post_message(events.Key(k, k if len(k) == 1 else None))
                await pilot.pause()
                await pilot.pause()
        return app.sent
    assert asyncio.run(run()) == ["xyz", "acb", "one", "two", "chào"]
