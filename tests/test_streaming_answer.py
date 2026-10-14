"""The answer shown while the model writes it (design revision 2026-10-14, step 3): the gateway hands each piece of
answer text to `on_text` as it streams; the agent turns them into "draft" events, at most every DRAFT_EVERY seconds;
the interface shows the draft under the conversation and replaces it with the checked answer, or with the tool line
when the model calls a tool. Tools described in the prompt are not drafted: their text holds the raw call syntax."""

import json
from types import SimpleNamespace as NS

import pytest

from peresearch import agent as agent_module, tui
from peresearch.llm import LLM
from tests.helpers import notes, settle, streamed, toolbox, tui_run
from tests.helpers import request, DictModel

USER = [{"role": "user", "content": "q"}]
TOOLS = [{"type": "function", "function": {"name": "search_notes", "description": "d",
                                           "parameters": {"type": "object", "properties": {}}}}]


@pytest.fixture(autouse=True)
def plain(monkeypatch):
    for name in ("PERESEARCH_LLM_TOOLS", "PERESEARCH_LLM_CONTEXT", "PERESEARCH_LLM_API"):
        monkeypatch.delenv(name, raising=False)


def test_the_gateway_hands_over_answer_text_as_it_streams(home):
    pieces = []
    llm = LLM(client=NS(chat=NS(completions=NS(create=lambda **kw: streamed(content="BPE merges pairs.", piece=5)))),
              model="m", retries=1)
    llm.on_text = pieces.append
    assert llm.chat(request(USER)).text == "BPE merges pairs." and "".join(pieces) == "BPE merges pairs." and len(pieces) == 4


def test_tools_in_the_prompt_are_not_handed_over(home, monkeypatch):
    monkeypatch.setenv("PERESEARCH_LLM_TOOLS", "prompt")
    pieces = []
    llm = LLM(client=NS(chat=NS(completions=NS(create=lambda **kw: streamed(
        content='<tool_call>{"name": "search_notes", "arguments": {}}</tool_call>')))), model="m", retries=1)
    llm.on_text = pieces.append
    assert [c.name for c in llm.chat(request(USER, TOOLS)).calls] == ["search_notes"] and pieces == []


def test_the_agent_drafts_at_most_every_interval(home, notes, monkeypatch):
    clock = iter([0.0, 0.05, 0.2, 0.25, 0.4] + [1.0] * 50)
    monkeypatch.setattr(agent_module, "_now", lambda: next(clock))
    events = []

    class Writer(DictModel):
        on_text = None

        def answer(self, messages, tools=None, thinking=None, stop=None):
            from peresearch.llm import Reply
            for piece in ("a", "b", "c", "d", "e"):
                self.on_text(piece)
            return Reply("abcde")

        def context_length(self):
            return None

        def reply_tokens(self):
            return None

    model = Writer()
    agent = agent_module.Agent(model, toolbox(home, notes), on_event=lambda k, d: events.append((k, d)))
    assert agent.ask("q").text == "abcde"
    drafts = [d for k, d in events if k == "draft"]
    assert drafts == ["a", "abc", "abcde"]                             # 0.0, 0.2, 0.4: none closer than the interval
    assert model.on_text is None                                       # handed back when the question ends


def test_the_interface_shows_the_draft_then_the_checked_answer(home, notes):
    async def steps(app, pilot):
        app.on_agent_event("draft", "Byte pair encoding merges")
        await pilot.pause()
        drafts = app.query("#log > .draft")
        app.seen_draft = [w.source for w in drafts]
        app.on_agent_event("tool", 'search_notes("bpe")')               # a call ends the draft
        await pilot.pause()
        app.after_tool = len(app.query("#log > .draft"))
        app.on_agent_event("draft", "Byte pair encoding merges the most frequent pair")
        app.on_agent_event("draft", "Byte pair encoding merges the most frequent pair of symbols.")
        await pilot.pause()
        app.updated = [w.source for w in app.query("#log > .draft")]
        from peresearch.agent import Answer, Check
        app.show_answer(Answer("Byte pair encoding merges the most frequent pair of symbols.", {}, Check(), 1))
        await pilot.pause()
        app.after_answer = len(app.query("#log > .draft"))
    app = tui_run(lambda e: None, steps)
    assert app.seen_draft == ["Byte pair encoding merges"] and app.after_tool == 0
    assert app.updated == ["Byte pair encoding merges the most frequent pair of symbols."] and app.after_answer == 0


# Found by mutation (2026-10-14): every route hands its text over, the threaded read included, and a fallback to tools
# in the prompt stops drafting.

def test_the_responses_and_anthropic_routes_hand_over_their_text(home, monkeypatch):
    from tests.test_llm_responses import client as responses_client, events
    from tests.test_llm_anthropic import Block, claude, reply
    monkeypatch.setenv("PERESEARCH_LLM_API", "responses")
    pieces = []
    llm = LLM(client=responses_client(lambda **kw: None, lambda **kw: iter(events(text="abcdefgh"))), model="m",
              retries=1)
    llm.on_text = pieces.append
    llm.chat(request(USER))
    assert "".join(pieces) == "abcdefgh" and len(pieces) == 2
    monkeypatch.delenv("PERESEARCH_LLM_API")
    native = []
    claude_llm = LLM(client=claude([], [reply([Block(type="text", text="hi")])]), model="m", retries=1,
                     provider="anthropic")
    claude_llm.on_text = native.append
    claude_llm.chat(request(USER))
    assert native == ["hi"]


def test_the_threaded_read_hands_over_too(home):
    pieces = []
    llm = LLM(client=NS(chat=NS(completions=NS(create=lambda **kw: streamed(content="abcdef", piece=3)))), model="m",
              retries=1)
    llm.on_text = pieces.append
    llm.chat(request(USER), stop=lambda: False)                                 # the agent always passes a stop
    assert pieces == ["abc", "def"]


def test_falling_back_to_tools_in_the_prompt_stops_drafting(home):
    from tests.test_llm_responses import TOOLS_UNSUPPORTED
    pieces = []

    def refuse(**kw):
        if kw.get("tools"):
            raise TOOLS_UNSUPPORTED
        return streamed(content='<tool_call>{"name": "search_notes", "arguments": {}}</tool_call>')
    llm = LLM(client=NS(chat=NS(completions=NS(create=refuse))), model="m", retries=1)
    llm.on_text = pieces.append
    llm.chat(request(USER, TOOLS))
    assert pieces == []

    class APIError(Exception):
        pass
    unreadable, sent = APIError("the tool-call parser could not parse the output"), []

    def parser_fails(**kw):
        sent.append(kw)
        if len(sent) == 1:
            raise unreadable
        return streamed(content='<tool_call>{"name": "search_notes", "arguments": {}}</tool_call>')
    other, later = LLM(client=NS(chat=NS(completions=NS(create=parser_fails))), model="m", retries=1), []
    other.on_text = later.append
    other.chat(request(USER, TOOLS))
    assert len(sent) == 2 and later == []


def test_a_mounted_draft_is_redrawn_and_a_new_model_call_drops_it(home, notes):
    async def steps(app, pilot):
        app.on_agent_event("draft", "first")
        await pilot.pause(0.2)                                          # mounted now
        app.on_agent_event("draft", "first and second")
        await pilot.pause(0.2)
        app.redrawn = [w.source for w in app.query("#log > .draft")]
        app.on_agent_event("think", "model call 2")
        await pilot.pause()
        app.after_think = len(app.query("#log > .draft"))
    app = tui_run(lambda e: None, steps)
    assert app.redrawn == ["first and second"] and app.after_think == 0


def test_a_draft_goes_out_once_the_interval_has_passed(home, notes, monkeypatch):
    clock = iter([0.0, agent_module.DRAFT_EVERY, agent_module.DRAFT_EVERY * 1.5])
    monkeypatch.setattr(agent_module, "_now", lambda: next(clock))
    events = []
    drafter = agent_module.Agent(NS(), toolbox(home, notes), on_event=lambda k, d: events.append(d))._drafter()
    for piece in "abc":
        drafter(piece)
    assert events == ["a", "ab"]                                        # exactly the interval: sent
