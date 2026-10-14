"""What goes on the wire, pinned: a fixed two-question conversation (a tool round, then a question with the first
turn as history) through the production Agent and LLM, over clients shaped like the SDKs, for each protocol and tool
mode: Chat Completions with native tools (the call carrying a Gemini-style thought signature) and with tools in the
prompt, the Responses API, and the Anthropic Messages API (a signed thinking block before the call).

Written on 2026-10-14 before the gateway's internal message format replaced OpenAI's dicts (PLAN.md, model gateway
step 1), so that the change is proven to leave every request body byte-identical. The baseline is
tests/data/gateway/wire-baseline.json; WRITE_WIRE_BASELINE=1 (not a PERESEARCH_ name: conftest clears those)
rewrites it, which only a deliberate change of what is sent may do."""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from peresearch.agent import Agent
from peresearch.llm import LLM
from tests.helpers import notes, toolbox  # noqa: F401  (notes is a fixture)
from tests.test_llm_anthropic import Block, reply as claude_reply
from tests.test_llm_responses import events as responses_events

BASELINE = Path(__file__).parent / "data" / "gateway" / "wire-baseline.json"
QUESTIONS = ("What do my notes say about BPE?", "And SuperBPE?")
SIGNATURE = {"google": {"thought_signature": "c2lnbmF0dXJlLW9m"}}
CALL_ARGS = '{"query": "BPE merges"}'


def chunk(**delta):
    return NS(choices=[NS(delta=NS(**delta), finish_reason=None)], usage=None)


def chat_end(finish, usage=(120, 30)):
    return [NS(choices=[NS(delta=NS(), finish_reason=finish)], usage=None),
            NS(choices=[], usage=NS(prompt_tokens=usage[0], completion_tokens=usage[1]))]


def chat_text(text):
    return [chunk(content=text[i:i + 7]) for i in range(0, len(text), 7)] + chat_end("stop")


def chat_native_call():
    call = NS(index=0, id="call-1", type="function", function=NS(name="search_notes", arguments=CALL_ARGS),
              extra_content=SIGNATURE)
    return [chunk(reasoning_content="look in the notes"), chunk(tool_calls=[call])] + chat_end("tool_calls")


def chat_prompt_call():
    return chat_text('<tool_call>{"name": "search_notes", "arguments": ' + CALL_ARGS + '}</tool_call>')


ANSWERS = ("BPE merges frequent symbol pairs into new tokens [N1].",
           "SuperBPE adds a second stage whose merges may cross spaces [N1].")


def chat_client(sent, first_round):
    replies = [first_round(), chat_text(ANSWERS[0]), chat_text(ANSWERS[1])]

    def create(**kw):
        sent.append(kw)
        return iter(replies.pop(0))
    return NS(chat=NS(completions=NS(create=create)), models=NS(list=lambda: NS(data=[])),
              base_url="http://localhost:8000/v1")


def responses_client(sent):
    replies = [responses_events(calls=[("call_1", "search_notes", CALL_ARGS)], reasoning="look in the notes"),
               responses_events(text=ANSWERS[0]), responses_events(text=ANSWERS[1])]

    def create(**kw):
        sent.append(kw)
        return iter(replies.pop(0))
    return NS(chat=NS(completions=NS(create=lambda **kw: pytest.fail("the chat route was used"))),
              responses=NS(create=create), models=NS(list=lambda: NS(data=[])), base_url="https://api.openai.com/v1")


def anthropic_client(sent):
    think = Block(type="thinking", thinking="look in the notes", signature="sig-1")
    call = Block(type="tool_use", id="toolu_1", name="search_notes", input=json.loads(CALL_ARGS))
    replies = [claude_reply([think, call], stop="tool_use"),
               claude_reply([Block(type="text", text=ANSWERS[0])]), claude_reply([Block(type="text", text=ANSWERS[1])])]

    def stream(**kw):
        sent.append(kw)
        return replies.pop(0)
    return NS(messages=NS(stream=stream), models=NS(list=lambda: NS(data=[])), base_url="https://api.anthropic.com")


CASES = {
    "chat-native-tools": lambda sent: (chat_client(sent, chat_native_call), "m", "", {}),
    "chat-prompt-tools": lambda sent: (chat_client(sent, chat_prompt_call), "m", "", {"PERESEARCH_LLM_TOOLS": "prompt"}),
    "responses": lambda sent: (responses_client(sent), "m", "", {"PERESEARCH_LLM_API": "responses"}),
    "anthropic-messages": lambda sent: (anthropic_client(sent), "claude-opus-5-5", "anthropic", {}),
}


def conversation(case, home, notes, tmp_path, monkeypatch):
    """The request bodies the conversation sent, as JSON text, with the temporary folder and today's date masked."""
    for name in ("PERESEARCH_LLM_TOOLS", "PERESEARCH_LLM_API", "PERESEARCH_LLM_THINKING", "PERESEARCH_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PERESEARCH_LLM_CONTEXT", "32768")
    sent = []
    client, model, provider, env = CASES[case](sent)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    llm = LLM(client=client, model=model, retries=1, provider=provider or None)
    agent = Agent(llm, toolbox(home, notes))
    answers = [agent.ask(q) for q in QUESTIONS]
    assert [a.text for a in answers] == list(ANSWERS) and answers[0].calls == [("search_notes", json.loads(CALL_ARGS))]
    text = json.dumps(sent, ensure_ascii=False, indent=1, sort_keys=True)
    text = text.replace(str(tmp_path), "<tmp>")
    return re.sub(r"Today is \d{4}-\d{2}-\d{2}, \w+\.", "Today is <date>.", text)


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_protocol_sends_the_pinned_request_bodies(case, home, notes, tmp_path, monkeypatch):
    got = json.loads(conversation(case, home, notes, tmp_path, monkeypatch))
    pinned = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    if os.environ.get("WRITE_WIRE_BASELINE") == "1":
        pinned[case] = got
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(pinned, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    assert case in pinned, f"no pinned bodies for {case}: WRITE_WIRE_BASELINE=1 records them"
    assert len(got) == 3                                           # a tool round, its answer, the second question
    assert got == pinned[case]
