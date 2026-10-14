"""Gemini's OpenAI-compatible endpoint (Google AI Studio): a thinking model's tool call carries a thought signature in
`extra_content`, and the next request must send it back with that call, or the endpoint answers 400 "Function call is
missing a thought_signature" (reported by the owner, 2026-10-14, gemini-3.5-flash-lite). The shape follows Google's
endpoint as OpenCode handles it (packages/ai/src/protocols/openai-chat.ts, ExtraContent)."""

import json
from types import SimpleNamespace as NS

from peresearch.llm import LLM

SIGNATURE = {"google": {"thought_signature": "c2lnbmF0dXJlLW9m"}}
TOOLS = [{"type": "function", "function": {"name": "tree", "description": "folders",
                                           "parameters": {"type": "object", "properties": {}}}}]


def chunks_with_call(extra=SIGNATURE, index=0):
    call = NS(index=index, id="function-call-1", type="function", function=NS(name="tree", arguments="{}"))
    if extra is not None:
        call.extra_content = extra
    return [NS(choices=[NS(delta=NS(content=None, tool_calls=[call]), finish_reason=None)], usage=None),
            NS(choices=[NS(delta=NS(content=None), finish_reason="tool_calls")], usage=None)]


def answer_chunks(text="Two projects."):
    return [NS(choices=[NS(delta=NS(content=text), finish_reason=None)], usage=None),
            NS(choices=[NS(delta=NS(content=None), finish_reason="stop")], usage=None)]


def gemini(sent, first):
    def create(**kw):
        sent.append(json.loads(json.dumps(kw["messages"])))
        return iter(first if len(sent) == 1 else answer_chunks())
    return NS(chat=NS(completions=NS(create=create)), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def ask_twice(llm):
    question = [{"role": "user", "content": "how many projects?"}]
    reply = llm.chat(question, tools=TOOLS)
    call = reply.calls[0]
    followed = question + [{"role": "assistant", "content": reply.text, "tool_calls": [
                               {"id": call.id, "type": "function",
                                "function": {"name": call.name, "arguments": json.dumps(call.args)}}]},
                           {"role": "tool", "tool_call_id": call.id, "content": "no folders are declared"}]
    return llm.chat(followed, tools=TOOLS)


def test_a_thought_signature_goes_back_with_its_call(home, monkeypatch):
    monkeypatch.delenv("PERESEARCH_LLM_TOOLS", raising=False)
    sent = []
    llm = LLM(client=gemini(sent, chunks_with_call()), model="gemini-3.5-flash-lite", retries=1)
    assert ask_twice(llm).text == "Two projects."
    replayed = sent[1][1]["tool_calls"][0]
    assert replayed["id"] == "function-call-1" and replayed["extra_content"] == SIGNATURE


def test_a_call_without_a_signature_is_sent_back_as_before(home, monkeypatch):
    monkeypatch.delenv("PERESEARCH_LLM_TOOLS", raising=False)
    sent = []
    llm = LLM(client=gemini(sent, chunks_with_call(extra=None)), model="m", retries=1)
    ask_twice(llm)
    assert "extra_content" not in sent[1][1]["tool_calls"][0]


def test_a_call_without_an_index_keeps_its_signature(home, monkeypatch):
    """Google's tool_calls items may come without an index (reported on Google's developer forum)."""
    monkeypatch.delenv("PERESEARCH_LLM_TOOLS", raising=False)
    sent = []
    llm = LLM(client=gemini(sent, chunks_with_call(index=None)), model="gemini-3.5-flash-lite", retries=1)
    ask_twice(llm)
    assert sent[1][1]["tool_calls"][0]["extra_content"] == SIGNATURE
