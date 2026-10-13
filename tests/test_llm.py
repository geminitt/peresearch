"""Contracts of the model client (peresearch.llm).

- Tool calls a model leaves as text are recovered exactly and removed from the text (a property).
- A call's arguments are one JSON object, or {} and the reason, whatever the server sent: cut anywhere, malformed,
  JSON of another kind (a property against a reference); only the last call of a reply that may be cut (finish
  "length", or every token allowed used) is said to be cut.
- A streamed reply is put back together exactly, whatever the pieces; the stream is always closed by its reader;
  one without a finish reason is a cut, not an answer (a property).
- Everything sent is redacted; nothing else in a message changes.
- Retries follow one policy: which errors, how many tries, which pauses, what the caller is told, which optional
  fields each request carries and which stay dropped, over two questions (a property against a reference).
- Settings are read with their defaults; a bad value is refused with the setting's name.
- An interrupt frees the caller at once, in flight or between retries, and a reader left blocked never keeps the
  program open; real servers: a cut stream, nothing listening on this machine, a slow start.
"""

import datetime
import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from hypothesis import example, given, settings as hsettings
from hypothesis import strategies as st

from peresearch import llm as llm_module
from peresearch.llm import (LLM, Interrupted, StreamCut, ToolCall, _outbound, _read_stream,
                            parse_text_calls, strip_text_calls)
from tests.helpers import streamed
from tests.test_guard import CANARIES

USER = [{"role": "user", "content": "x"}]


def switch(on: bool) -> dict:
    """The reasoning switch as sent: the chat template's flag (vLLM, llama.cpp) and, to turn it off, Ollama's own
    field (it ignores the template's: checked on Ollama 0.32.15, qwen3:0.6b reasoned 1,119 characters with it)."""
    return {"chat_template_kwargs": {"enable_thinking": on}, **({} if on else {"reasoning_effort": "none"})}


def client_of(create):
    return NS(chat=NS(completions=NS(create=create)))


# --- tool calls written as text ------------------------------------------------------------------------------

names = st.from_regex(r"[a-z][a-z_]{0,10}", fullmatch=True)
values = st.one_of(st.integers(-5, 500), st.text(alphabet="abc xyz/.-", min_size=1, max_size=12).filter(
    lambda s: s.strip() == s and not s.lstrip("-").isdigit()), st.lists(st.integers(0, 9), max_size=3))
calls = st.lists(st.tuples(names, st.dictionaries(names, values, max_size=3)), min_size=1, max_size=3)
prose = st.text(alphabet="Hello, I will look this up. ", max_size=20)


def qwen(name, args):
    params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def hermes(name, args):
    return f"<tool_call>{json.dumps({'name': name, 'arguments': args})}</tool_call>"


def bare(name, args):
    """A call as bare JSON, the tags left out (Qwen3-0.6B through the prompt on Ollama, 2026-10-07)."""
    return json.dumps({"name": name, "arguments": args})


def llama(name, args):
    """Llama 3.x's JSON form (seen through Ollama, 2026-10-06/07)."""
    return json.dumps({"type": "function", "function": {"name": name, "parameters": args}})


@given(calls, prose, prose, st.sampled_from([qwen, hermes, bare, llama]))
@example([("web_search", {"query": "bpe history"})], "", "", qwen)
@example([("read", {"path": "a.md", "start": 3})], "Looking. ", "", hermes)
@example([("tree", {})], "", "\n\nI cannot call the tree tool as it requires specific parameters.", bare)
def test_tool_calls_left_as_text_are_recovered_and_removed(cs, before, after, write):
    """In tags (Qwen's XML, Hermes' JSON) always; as bare JSON only when its name is one of the tools offered."""
    text = before + "\n".join(write(n, a) for n, a in cs) + after
    names = {n for n, _ in cs}
    got = parse_text_calls(text, names)
    assert [(c.id, c.name, c.args) for c in got] == [(f"text-{i}", n, a) for i, (n, a) in enumerate(cs)]
    assert strip_text_calls(text, names) == (before + "\n" * (len(cs) - 1) + after).strip()
    if write in (bare, llama):                              # JSON naming no tool offered is the answer's own text
        assert parse_text_calls(text) == [] and parse_text_calls(text, {"other"}) == []
        assert strip_text_calls(text) == text.strip()


def test_qwen_calls_win_over_hermes_ones_and_broken_json_is_skipped():
    fenced = 'Here:\n```json\n{"name": "tree", "arguments": {}}\n```\nDone.'          # a fence around it goes too
    assert [c.name for c in parse_text_calls(fenced, {"tree"})] == ["tree"]
    assert strip_text_calls(fenced, {"tree"}) == "Here:\n\nDone."
    assert parse_text_calls('{"name": "tree", "arguments": {"x": 1}} and {"name": "tree"', {"tree"}) == \
        [ToolCall("text-0", "tree", {"x": 1})]                     # the unfinished second one left as text
    assert parse_text_calls('{"name": "tree", "arguments": [1]}', {"tree"}) == [] and \
        parse_text_calls('{"type":"function","name":"tree","parameters":{"}}', {"tree"}) == []    # malformed (seen)
    assert parse_text_calls(hermes("a", {}) + bare("b", {}), {"a", "b"}) == [ToolCall("text-0", "a", {})]
    two = 'Set {x} first; {"name": "tree", "arguments": {}} then {"name": "read", "arguments": {"path": "a"}} ok'
    assert [c.name for c in parse_text_calls(two, {"tree", "read"})] == ["tree", "read"]    # braces before, two calls
    assert strip_text_calls(two, {"tree", "read"}) == "Set {x} first;  then  ok"
    mixed = qwen("a", {"x": 1}) + hermes("b", {"y": 2})
    assert [c.name for c in parse_text_calls(mixed)] == ["a"]
    assert parse_text_calls('<tool_call>{"name": "b", "arguments": {}}</tool_call><tool_call>{broken</tool_call>') == \
        [ToolCall("text-0", "b", {})]
    assert parse_text_calls('<tool_call>["not a call"]</tool_call>') == []
    assert parse_text_calls('<tool_call>{"arguments": {}}</tool_call>') == [] and parse_text_calls(None) == []
    assert strip_text_calls(None) == "" and parse_text_calls(hermes("n", None))[0].args == {}
    assert [(c.args, c.error) for c in parse_text_calls('<tool_call>{"name": "tree"}</tool_call>')] == [({}, "")]
    text = ('<tool_call>{"name": "read", "arguments": "{\\"path\\": \\"a.md\\"}"}</tool_call> '
            '<tool_call>{"name": "tree", "arguments": [1]}</tool_call>')
    assert [(c.name, c.args, c.error) for c in parse_text_calls(text)] == [
        ("read", {"path": "a.md"}, ""), ("tree", {}, "the arguments are a list, not a JSON object")]


json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-9, 99) | st.text(alphabet='ab "{}[]:,\\', max_size=4),
    lambda inner: st.lists(inner, max_size=2) | st.dictionaries(st.text(alphabet="ab", max_size=2), inner, max_size=2),
    max_leaves=6)


@given(json_values, st.integers(0, 60), st.booleans(), st.booleans())
@example({"path": "a.md", "start": 3}, 18, True, True)        # cut inside a string, in a reply that may be cut
@example({"path": "a.md"}, 99, True, True)                    # whole, although every token was used
@example({"path": "a.md"}, 99, False, False)                  # a Hermes call's arguments, already decoded
@example(None, 99, False, False)                              # a Hermes call without arguments
@example([1, 2], 99, True, True)                              # valid JSON of another kind: not said to be cut
def test_tool_call_arguments_are_a_json_object_or_the_call_says_why_not(value, keep, cut, as_text):
    """Arguments cut anywhere or malformed, or JSON that is not an object, do not end the question (they raised out
    of chat, diagram mark 1): {} and the reason, for the agent to hand to the model instead of running the call."""
    raw = json.dumps(value, ensure_ascii=False)[:keep] if as_text else value
    args, why = llm_module._arguments(raw, cut)
    if raw is None or raw == "":
        want = ({}, "the reply was cut at max_tokens before the call's arguments" if cut else "")
    elif not isinstance(raw, str):
        want = (raw, "") if isinstance(raw, dict) else ({}, f"the arguments are a {type(raw).__name__}, not a JSON object")
    else:
        try:
            parsed = json.loads(raw)
        except ValueError as e:
            want = ({}, f"the arguments are not valid JSON: {e}" + (" (the reply was cut at max_tokens)" if cut else ""))
        else:
            want = (parsed, "") if isinstance(parsed, dict) else (
                {}, f"the arguments are a {type(parsed).__name__}, not a JSON object")
    assert (args, why) == want and isinstance(args, dict)


def test_a_broken_text_call_does_not_hide_the_next_one():
    assert parse_text_calls('<tool_call>{broken</tool_call><tool_call>{"name": "b", "arguments": {}}</tool_call>') == \
        [ToolCall("text-0", "b", {})]
    assert parse_text_calls('<tool_call>{"name": "a", "arguments": {x}}</tool_call>'      # braces, not JSON
                            '<tool_call>{"name": "b", "arguments": {}}</tool_call>') == [ToolCall("text-0", "b", {})]


# --- a streamed reply ------------------------------------------------------------------------------------------

def chunk(**delta):
    return NS(choices=[NS(delta=NS(**delta), finish_reason=None)], usage=None)


pieces = st.lists(st.one_of(
    st.tuples(st.just("text"), st.text(max_size=6)),
    st.tuples(st.just("reasoning_content"), st.text(max_size=6)),
    st.tuples(st.just("reasoning"), st.text(max_size=6)),
    st.tuples(st.just("call"), st.tuples(st.integers(0, 2), st.sampled_from([None, "id-a", "id-b"]),
                                         st.text(alphabet="abc", max_size=3), st.text(alphabet='{}"1:', max_size=4)))),
    max_size=25)


class Stream:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, 0

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed += 1


@given(pieces, st.sampled_from(["stop", "tool_calls", "length", None]), st.booleans())
def test_a_streamed_reply_is_put_back_together_exactly(parts, finish, usage):
    chunks, text, thought, want = [], "", "", {}
    for kind, value in parts:
        if kind == "text":
            chunks.append(chunk(content=value))
            text += value
        elif kind in ("reasoning_content", "reasoning"):
            chunks.append(chunk(**{kind: value}))
            thought += value
        else:
            i, cid, name, args = value
            chunks.append(chunk(tool_calls=[NS(index=i, id=cid, function=NS(name=name, arguments=args))]))
            c = want.setdefault(i, {"id": None, "name": "", "args": ""})
            c["id"] = cid or c["id"]
            c["name"] += name
            c["args"] += args
    chunks.append(NS(choices=[NS(delta=NS(), finish_reason=finish)], usage=None))
    chunks.append(NS(choices=[], usage=NS(prompt_tokens=7, completion_tokens=3) if usage else None))
    stream, first = Stream(chunks), []
    if finish is None:
        with pytest.raises(StreamCut) as err:
            _read_stream(stream, None)
        assert str(err.value) == "the reply stream ended without a finish reason: the connection was cut mid-answer"
    else:
        r = _read_stream(stream, lambda: False, lambda: first.append(1))
        assert r == {"text": text, "reasoning": thought, "finish": finish,
                     "calls": [dict(c, id=c["id"] or f"call-{i}") for i, c in sorted(want.items())],
                     "prompt_tokens": 7 if usage else 0, "completion_tokens": 3 if usage else 0}
        assert first == [1]
    assert stream.closed == 1


def test_a_tool_call_piece_without_an_index_is_a_new_call_and_stop_interrupts_between_chunks():
    s = Stream([chunk(tool_calls=[NS(index=None, id=None, function=None)]),
                chunk(tool_calls=[NS(id="c", function=NS(name="read", arguments=None))]),
                NS(choices=[NS(delta=NS(), finish_reason="tool_calls")], usage=None)])
    assert _read_stream(s, None)["calls"] == [{"id": "call-0", "name": "", "args": ""}, {"id": "c", "name": "read", "args": ""}]
    s = Stream([chunk(content="a"), chunk(content="b")])
    seen = []
    with pytest.raises(Interrupted):
        _read_stream(s, lambda: seen.append(1) or len(seen) > 1)
    assert s.closed == 1 and seen == [1, 1]
    _read_stream(iter([NS(choices=[NS(delta=NS(), finish_reason="stop")], usage=None)]), None)   # no close(): fine


# --- what is sent ----------------------------------------------------------------------------------------------

def test_everything_sent_is_redacted_and_nothing_else_changes(home):
    key = CANARIES["openai-key"]
    m = {"role": "assistant", "content": f"key {key}", "name": "x",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "grep", "arguments": f'{{"pattern": "{key}"}}'}}]}
    out = _outbound(m)
    assert key not in json.dumps(out) and out["content"] == "key [REDACTED:openai-key]"
    assert out["tool_calls"][0]["function"] == {"name": "grep", "arguments": '{"pattern": "[REDACTED:openai-key]"}'}
    assert (out["role"], out["name"], out["tool_calls"][0]["id"]) == ("assistant", "x", "c1")
    assert m["content"] == f"key {key}"                                  # the caller's message is untouched
    assert _outbound({"role": "tool", "content": None}) == {"role": "tool", "content": None}
    sent = []
    LLM(client=client_of(lambda **kw: sent.append(kw) or streamed(content="ok")), model="m").chat(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": key}])
    assert sent[0]["messages"] == [{"role": "system", "content": "be brief"},
                                   {"role": "user", "content": "[REDACTED:openai-key]"}]
    log = [json.loads(x) for x in (home / "audit.jsonl").read_text().splitlines()]
    assert {e["channel"] for e in log if e["event"] == "outbound"} == {"model"}


@pytest.mark.parametrize("own, setting, call, want", [
    (None, "", None, None), (False, "", None, False), (None, "on", None, True), (None, "off", None, False),
    (True, "off", None, True), (None, "on", False, False), (None, "of", None, "refused"), (True, "ON", None, "refused")])
def test_the_request_carries_the_model_settings_and_the_reasoning_switch(home, monkeypatch, own, setting, call, want):
    monkeypatch.setenv("PERESEARCH_LLM_THINKING", setting)
    if want == "refused":                     # it was silently ignored: the server's default was used unawares
        with pytest.raises(ValueError, match=f"^PERESEARCH_LLM_THINKING must be on or off, .*got '{setting}'"):
            LLM(client=client_of(None), model="m", thinking=own)
        return
    sent = []
    llm = LLM(client=client_of(lambda **kw: sent.append(kw) or streamed(content="ok")), model="m", thinking=own,
              max_tokens=99, temperature=0.1, top_p=0.5)
    llm.chat(USER, tools=[{"type": "function"}], thinking=call)
    llm.chat(USER, tools=[])
    kw = sent[0]
    assert {k: kw[k] for k in ("model", "max_tokens", "temperature", "top_p", "tools", "stream", "stream_options")} == {
        "model": "m", "max_tokens": 99, "temperature": 0.1, "top_p": 0.5, "tools": [{"type": "function"}],
        "stream": True, "stream_options": {"include_usage": True}}
    assert kw.get("extra_body") == (None if want is None else switch(want))
    assert sent[1]["tools"] is None


@pytest.mark.parametrize("model, context, cap, budget", [
    ("m", "4096", None, 1024), ("m", "32768", None, 8192), ("m", "1024", None, 256), ("m", "16380", None, 4095),
    ("m", "", None, None),                       # nothing known: no max_tokens sent, the server's own
    ("gpt-4o", "", None, 16384),                 # the table's output limit (its context 128,000: a quarter is more)
    ("gpt-4o", "32768", None, 8192),             # the smaller of the two
    ("m", "32768", 99, 99), ("m", "", 99, 99)])  # a cap the caller sets
def test_the_reply_budget_comes_from_the_model(home, monkeypatch, model, context, cap, budget):
    """The tokens a reply may use: the smallest of what is known — the model's output limit (the profile), a quarter
    of its context, a cap the caller sets — and none sent when nothing is known: no fixed size (owner 2026-10-07).
    On Ollama's default context (4,096) a fixed 4,096 left the conversation no room (runs/e2e/e2e-silent-cut-2026-10-07).
    A reply that used the whole budget may be cut; with no budget only finish "length" says so."""
    monkeypatch.setenv("PERESEARCH_LLM_CONTEXT", context)
    sent = []
    used = budget or 50
    llm = LLM(client=client_of(lambda **kw: sent.append(kw) or streamed(tool_calls=[("read", '{"path": "a')], usage=NS(
        prompt_tokens=9, completion_tokens=used))), model=model, max_tokens=cap)
    reply = llm.chat(USER)
    field = "max_completion_tokens" if model == "gpt-4o" else "max_tokens"
    assert llm.reply_tokens() == budget and sent[0].get(field) == budget and (field in sent[0]) == bool(budget)
    assert not ({"max_tokens", "max_completion_tokens"} - {field}) & sent[0].keys()
    assert reply.calls[0].error.endswith("(the reply was cut at max_tokens)") == bool(budget)


def test_the_reply_carries_calls_reasoning_tokens_and_why_it_ended():
    xml = qwen("web_search", {"query": "bpe"})
    r = LLM(client=client_of(lambda **kw: streamed(content="see " + xml, reasoning="hmm",
                                                   usage=NS(prompt_tokens=5, completion_tokens=2), finish="length")),
            model="m").chat(USER)
    assert (r.text, r.reasoning, r.prompt_tokens, r.completion_tokens, r.finish) == ("see", "hmm", 5, 2, "length")
    assert [(c.name, c.args) for c in r.calls] == [("web_search", {"query": "bpe"})]
    structured = LLM(client=client_of(lambda **kw: streamed(content="keep " + xml, tool_calls=[("read", '{"path": "a"}')])),
                     model="m").chat(USER)
    assert structured.text == "keep " + xml and [c.name for c in structured.calls] == ["read"]
    assert structured.calls[0].id == "c0"
    empty_args = LLM(client=client_of(lambda **kw: streamed(tool_calls=[("tree", "")])), model="m").chat(USER)
    assert empty_args.calls[0].args == {}
    in_thought = LLM(client=client_of(lambda **kw: streamed(content="plain", reasoning="let me " + xml)), model="m").chat(USER)
    assert [c.name for c in in_thought.calls] == ["web_search"] and in_thought.text == "plain"
    # Only the last call of a reply that may be cut can be cut. vLLM reports a call cut at max_tokens as finish
    # "tool_calls" (measured with Qwen3.5-0.8B, runs/serve/client-real-check-2026-10-04.log): a reply that used every
    # token allowed may be cut, as may one that ended with finish "length".

    def errors(calls, used, allowed=32, finish="tool_calls"):
        reply = LLM(client=client_of(lambda **kw: streamed(tool_calls=calls, finish=finish, usage=NS(
            prompt_tokens=9, completion_tokens=used))), model="m", max_tokens=allowed).chat(USER)
        return [c.error for c in reply.calls]
    half, bad = '{"path": "notes/a', "the arguments are not valid JSON: Unterminated string starting at: line 1 column 10 (char 9)"
    before = "the reply was cut at max_tokens before the call's arguments"
    assert errors([("read", half), ("read", half)], 32) == [bad, bad + " (the reply was cut at max_tokens)"]
    assert errors([("tree", ""), ("read", "")], 32) == ["", before]
    assert errors([("read", '{"path": "a.md"}')], 32) == [""]                 # whole although every token was used
    assert errors([("tree", ""), ("read", half)], 31) == ["", bad]               # tokens left: not cut, not said
    assert errors([("tree", "")], 5, finish="length") == [before] and errors([("tree", "")], 1, allowed=1) == [before]
    assert errors([("tree", "[1]")], 32) == ["the arguments are a list, not a JSON object"]


words = st.text(alphabet="abc xyz.,", min_size=1, max_size=30).filter(lambda t: t.strip())


@given(words, st.text(alphabet="abc xyz.,", max_size=30), st.sampled_from(["field", "tags", "closing", "unclosed"]),
       st.sampled_from(["", " ", "\n"]))
@example("look it up", "BPE merges pairs.", "tags", "\n")
@example("look it up", "BPE merges pairs.", "closing", "")       # the template opened <think>: only the closing tag
@example("cut before the end", "", "unclosed", "")
@example("look it up", "Close it with </think> then.", "tags", "")      # split at the FIRST closing tag
@example("look it up", "Close it with </think> then.", "closing", "\n")
def test_reasoning_comes_back_as_reasoning_however_the_server_sends_it(thought, answer, form, gap):
    """A separate reasoning field, <think>…</think> at the start of the text, only a closing </think> (the chat
    template opened the tag), or a <think> never closed (cut): the reasoning is never shown as the answer (Aider's
    reasoning_tag). Text that merely mentions the tag is left alone."""
    content = {"field": answer, "tags": f"{gap}<think>{thought}</think>{gap}{answer}",
               "closing": f"{thought}</think>{gap}{answer}", "unclosed": f"{gap}<think>{thought}"}[form]
    reply = LLM(client=client_of(lambda **kw: streamed(content=content, reasoning=thought if form == "field" else "")),
                model="m").chat(USER)
    assert reply.reasoning.strip() == thought.strip()
    assert reply.text == ("" if form == "unclosed" else answer.strip())      # a text without calls is stripped
    mention = LLM(client=client_of(lambda **kw: streamed(content="Models write <think> tags.")), model="m").chat(USER)
    assert (mention.text, mention.reasoning) == ("Models write <think> tags.", "")
    joined = LLM(client=client_of(lambda **kw: streamed(content="<think>b</think>c", reasoning="a")), model="m").chat(USER)
    assert (joined.reasoning, joined.text) == ("a\nb", "c")         # a field and tags both: one after the other
    both = "Models write <think> and </think> tags."               # not at the start: a mention, left alone
    mention = LLM(client=client_of(lambda **kw: streamed(content=both)), model="m").chat(USER)
    assert (mention.text, mention.reasoning) == (both, "")
    call = LLM(client=client_of(lambda **kw: streamed(content="<think>first " + qwen("tree", {}) + "</think>")),
               model="m").chat(USER)
    assert [c.name for c in call.calls] == ["tree"] and call.text == ""     # a call inside the reasoning still runs


def test_the_client_is_made_from_the_settings_with_its_defaults(home, monkeypatch):
    import openai
    made = []
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: made.append(kw) or NS())
    monkeypatch.setenv("PERESEARCH_LLM_URL", "https://x.modal.run/v1")
    monkeypatch.delenv("PERESEARCH_LLM_KEY", raising=False)
    llm = LLM()
    assert made[-1] == {"base_url": "https://x.modal.run/v1", "api_key": "none", "timeout": 600.0, "max_retries": 0}
    monkeypatch.setenv("PERESEARCH_LLM_KEY", "wk-1")
    LLM()
    assert made[-1]["api_key"] == "wk-1" and LLM(key="own").client is not None and made[-1]["api_key"] == "own"
    assert (llm.max_tokens, llm.temperature, llm.top_p, llm.retries, llm.wait, llm.slow_start) == (None, None, None, 5, 10.0, 5.0)
    monkeypatch.delenv("PERESEARCH_LLM_URL")
    with pytest.raises(RuntimeError, match=r"^no model endpoint: set PERESEARCH_LLM_URL \(see peresearch/settings\.py\)$"):
        LLM()
    injected = LLM(client=client_of(None), model="m")
    assert injected.local is False and not hasattr(injected, "budget") and injected.on_wait("x") is None
    # the served model's context, in tokens: the setting, else what the server reports (vLLM's max_model_len, also
    # read where the OpenAI client keeps unknown fields), else unknown; asked of the server once
    asked = []

    def served(*models):
        def listing():
            asked.append(1)
            if models and isinstance(models[0], Exception):
                raise models[0]
            return NS(data=list(models))
        return NS(chat=client_of(None).chat, models=NS(list=listing))
    monkeypatch.delenv("PERESEARCH_LLM_CONTEXT", raising=False)
    llm = LLM(client=served(NS(id="other", max_model_len=1), NS(id="m", max_model_len=16384)), model="m")
    assert (llm.context_length(), llm.context_length(), len(asked)) == (16384, 16384, 1)
    assert LLM(client=served(NS(id="m", model_extra={"max_model_len": 8192})), model="m").context_length() == 8192
    assert LLM(client=served(NS(id="m")), model="m").context_length() is None
    assert LLM(client=served(error("APIConnectionError")), model="m").context_length() is None
    assert LLM(client=client_of(None), model="m").context_length() is None          # a client with no model list
    # llama.cpp says the context it runs with in `meta.n_ctx` (n_ctx_train is the model's own, larger); Ollama says
    # none in /v1/models, only for a LOADED model in its /api/ps — loaded first by an empty /api/generate. Without
    # them the model table's maximum was taken (131,072 for a server running 8,192: recorded 2026-10-06) and Ollama
    # cuts a longer conversation silently, its start (the instructions) first.
    meta = NS(id="m", model_extra={"meta": {"n_ctx": 8192, "n_ctx_train": 131072}})
    assert LLM(client=served(meta), model="m").context_length() == 8192

    def ollama(loaded, fail=None, base="http://127.0.0.1:11434/v1/", name="qwen3:0.6b", listed="qwen3:0.6b"):
        calls = []

        def answer(method, url, body=None, timeout=None):
            calls.append((method, url, body))
            assert timeout == (10 if url.endswith("/api/ps") else 600)     # a hung server never holds a question
            if isinstance(fail, str) and fail in url:
                raise httpx.ConnectError("refused")
            if fail == 404:
                return httpx.Response(404, json={"error": "not found"})
            if url.endswith("/api/ps"):
                return httpx.Response(200, json={"models": [{"name": "other", "context_length": 1}]
                                                 + ([{"name": listed, "model": listed, "context_length": 8192}]
                                                    if loaded else [])})
            loaded.append(1)
            return httpx.Response(200, json={"done": True})
        http = NS(get=lambda url, timeout=None: answer("GET", url, None, timeout),
                  post=lambda url, json=None, timeout=None: answer("POST", url, json, timeout))
        client = NS(chat=client_of(None).chat, models=NS(list=lambda: NS(data=[NS(id=listed)])),
                    base_url=httpx.URL(base), _client=http)
        return LLM(client=client, model=name), calls
    llm, calls = ollama([1])
    assert (llm.context_length(), llm.context_source()) == (8192, "server")
    assert calls == [("GET", "http://127.0.0.1:11434/api/ps", None)]
    llm, calls = ollama([])
    assert llm.context_length() == 8192 and [c[:2] for c in calls] == [
        ("GET", "http://127.0.0.1:11434/api/ps"), ("POST", "http://127.0.0.1:11434/api/generate"),
        ("GET", "http://127.0.0.1:11434/api/ps")] and calls[1][2] == {"model": "qwen3:0.6b"}
    for fail in ("/api/ps", "/api/generate"):                     # not Ollama, or it fails: unknown, nothing raised
        assert ollama([], fail)[0].context_length() is None
    llm, calls = ollama([], 404)                                  # another server: nothing is loaded on it
    assert llm.context_length() is None and [c[0] for c in calls] == ["GET"]
    # a name without a tag is Ollama's "<name>:latest" (checked on Ollama 0.32.15: "tiny" served and listed as
    # "tiny:latest" in /v1/models and /api/ps); another tag is another model
    assert ollama([1], name="tiny", listed="tiny:latest")[0].context_length() == 8192
    assert ollama([1], name="tiny", listed="tiny:q8")[0].context_length() is None
    assert ollama([1], name="org/tiny", listed="org/tiny:latest")[0].context_length() == 8192
    llm, calls = ollama([1], base="http://127.0.0.1:11434/")      # not an OpenAI-style …/v1 address: not asked
    assert (llm.context_length(), llm.context_source(), calls) == (None, "", [])
    monkeypatch.setenv("PERESEARCH_LLM_CONTEXT", "32768")
    llm = LLM(client=served(NS(id="m", max_model_len=16384)), model="m")
    assert (llm.context_length(), llm.context_source()) == (32768, "setting")
    monkeypatch.setenv("PERESEARCH_LLM_CONTEXT", "1024")                  # the smallest accepted
    assert LLM(client=served(), model="m").context_length() == 1024
    for bad in ("abc", "1023", "16k"):
        monkeypatch.setenv("PERESEARCH_LLM_CONTEXT", bad)
        with pytest.raises(ValueError, match=rf"^PERESEARCH_LLM_CONTEXT must be a whole number of tokens, at least 1024 "
                                             rf"\(got '{bad}'\)"):
            LLM(client=served(), model="m")


def test_the_request_follows_the_model_profile(home, monkeypatch):
    """The profile is found by the served name, else by vLLM's `root` for an alias (also a path in the Hugging Face
    cache, as vLLM reports it offline); sampling is sent only when the profile has it (Qwen's published values) or
    the caller gives it; the reasoning switch only to a family that has it, or to a model the table does not know (a
    server that refuses it is handled as before); the context falls back to the profile's."""
    for k in ("PERESEARCH_LLM_CONTEXT", "PERESEARCH_LLM_TOOLS", "PERESEARCH_LLM_TEMPERATURE", "PERESEARCH_LLM_TOP_P"):
        monkeypatch.delenv(k, raising=False)

    def run(model, *listed, **own):
        sent = []
        client = NS(chat=client_of(lambda **kw: sent.append(kw) or streamed(content="ok")).chat,
                    models=NS(list=lambda: NS(data=list(listed))))
        llm = LLM(client=client, model=model, **own)
        llm.chat(USER, tools=[{"type": "function", "function": {"name": "tree", "parameters": {}}}], thinking=False)
        return llm, sent[0]
    cache = "/__modal/volumes/vo-x/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d0"
    for listed in (NS(id="llm", root="Qwen/Qwen3.6-35B-A3B-FP8"), NS(id="llm", root=cache)):
        llm, kw = run("llm", listed)
        assert (llm.profile().family, kw["temperature"], kw["top_p"]) == ("qwen", 0.6, 0.95)
        assert kw["extra_body"] == switch(False) and kw["tools"]
        assert llm.context_length() == 262144                 # the server said none: the profile's
    llm, kw = run("llm", NS(id="llm", root="Qwen/Qwen3.6-35B-A3B-FP8", max_model_len=32768))
    assert llm.context_length() == 32768                      # the server's own limit wins over the model's
    llm, kw = run("my-own-model")                             # unknown: nothing forced, the switch tried
    assert "temperature" not in kw and "top_p" not in kw and "extra_body" in kw and llm.context_length() is None
    llm, kw = run("my-own-model", temperature=0.2, top_p=0.9)
    assert (kw["temperature"], kw["top_p"]) == (0.2, 0.9)
    llm, kw = run("gemma2:9b")                                # known, no switch, no tool calls of its own
    assert "extra_body" not in kw and "temperature" not in kw and kw["tools"] is None
    llm, kw = run("gemma3:27b")             # Ollama's name for the chat-tuned model: its row (gemma-3-27b-it) has them
    assert "extra_body" not in kw and kw["tools"] is not None
    assert "<tool_call>" in kw["messages"][0]["content"] if kw["messages"][0]["role"] == "system" else True


TOOLS = [{"type": "function", "function": {"name": n, "description": f"the {n} tool",
                                           "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}
         for n in ("tree", "search_notes", "read")]
rounds = st.lists(st.tuples(st.text(alphabet="ab .", max_size=8),
                            st.lists(st.tuples(st.sampled_from(["tree", "search_notes", "read"]),
                                               st.dictionaries(names, values, max_size=2),
                                               st.text(alphabet="result x\n", max_size=15)), min_size=1, max_size=3)),
                  max_size=3)


@given(rounds, st.booleans())
@hsettings(max_examples=150)
def test_tool_calls_through_the_prompt_keep_every_call_and_result(history, limit_note):
    """The conversation for a model whose calls go through the prompt: the tools and the call format in the system
    message; no tool role and no tool_calls left; user and assistant alternate; every earlier call reads back as the
    same call, every result sits in a tool_response with its tool's name."""
    from peresearch.llm import CALL_FORMAT, prompt_tools
    msgs, n, want_calls, want_results = [{"role": "system", "content": "rules"}, {"role": "user", "content": "q"}], 0, [], []
    for text, calls in history:
        ids = [f"c{n + i}" for i in range(len(calls))]
        n += len(calls)
        msgs.append({"role": "assistant", "content": text, "tool_calls": [
            {"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args, _) in zip(ids, calls)]})
        msgs += [{"role": "tool", "tool_call_id": i, "content": out} for i, (_, _, out) in zip(ids, calls)]
        want_calls.append([(name, args) for name, args, _ in calls])
        want_results += [(name, out) for name, _, out in calls]
    if limit_note:
        msgs.append({"role": "user", "content": "Limit reached: answer now."})
    out = prompt_tools(msgs, TOOLS)
    assert out[0]["role"] == "system" and out[0]["content"].startswith("rules" + CALL_FORMAT)
    assert all(f"- {t['function']['name']}: " in out[0]["content"] for t in TOOLS)
    assert all(set(m) == {"role", "content"} and m["role"] in ("system", "user", "assistant") for m in out)
    assert all(a["role"] != b["role"] for a, b in zip(out[1:], out[2:]))                  # they alternate
    said = [[(c.name, c.args) for c in parse_text_calls(m["content"])] for m in out if m["role"] == "assistant"]
    assert said == want_calls
    got = [(r.split('"')[1], r.split("\n", 1)[1].rsplit("\n</tool_response>", 1)[0])
           for m in out if m["role"] == "user" for r in m["content"].split("<tool_response name=")[1:]]
    assert got == want_results
    assert prompt_tools(msgs[:2], None) == msgs[:2]                 # no tools: the conversation is left as it is


def test_tools_through_the_prompt_read_exactly_so():
    """One conversation written out in full: each tool with its description and parameters (non-ASCII kept as it is:
    escaped Vietnamese costs tokens and reads badly), a call without arguments, a result whose call is not in the
    conversation, two system messages kept apart, plain turns left as they are, consecutive user turns joined."""
    from peresearch.llm import CALL_FORMAT, prompt_tools
    tools = [{"type": "function", "function": {"name": "đọc", "description": "Đọc một tệp",
                                               "parameters": {"type": "object", "properties": {"đường": {"type": "string"}}}}},
             {"type": "function", "function": {"name": "tree"}}]
    msgs = [{"role": "system", "content": "rules"}, {"role": "system", "content": "more"},
            {"role": "user", "content": "hỏi"}, {"role": "assistant", "content": "  thinking aloud  "},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function",
                                                               "function": {"name": "tree", "arguments": ""}}]},
            {"role": "tool", "tool_call_id": "a", "content": "listing"},
            {"role": "tool", "tool_call_id": "gone", "content": "older"}]
    listing = ('- đọc: Đọc một tệp\n  parameters: {"type":"object","properties":{"đường":{"type":"string"}}}\n'
               '- tree: \n  parameters: {}')
    assert prompt_tools(msgs, tools) == [
        {"role": "system", "content": "rules" + CALL_FORMAT + listing},
        {"role": "system", "content": "more" + CALL_FORMAT + listing},
        {"role": "user", "content": "hỏi"}, {"role": "assistant", "content": "  thinking aloud  "},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": '<tool_call>{"name": "tree", "arguments": {}}</tool_call>'},
        {"role": "user", "content": '<tool_response name="tree">\nlisting\n</tool_response>\n\n'
                                    '<tool_response name="tool">\nolder\n</tool_response>'}]
    said = prompt_tools([{"role": "assistant", "content": " Looking. ", "tool_calls": [
        {"id": "b", "type": "function", "function": {"name": "đọc", "arguments": '{"đường": "ghi chú.md"}'}}]}], tools)
    assert said == [{"role": "assistant", "content": 'Looking.\n<tool_call>{"name": "đọc", "arguments": '
                                                     '{"đường": "ghi chú.md"}}</tool_call>'}]


REFUSED = ("Error code: 400 - {'error': {'message': '\"auto\" tool choice requires --enable-auto-tool-choice and "
           "--tool-call-parser to be set'}}")                       # vLLM's, as served without a parser


def test_a_server_that_refuses_tools_gets_them_through_the_prompt_from_then_on(home, monkeypatch):
    for k in ("PERESEARCH_LLM_TOOLS", "PERESEARCH_LLM_CONTEXT"):
        monkeypatch.delenv(k, raising=False)
    sent = []

    def create(**kw):
        sent.append(kw)
        if kw.get("tools"):
            raise type("BadRequestError", (Exception,), {})(REFUSED)
        return streamed(content='Looking. <tool_call>{"name": "tree", "arguments": {"q": "x"}}</tool_call>')
    llm = LLM(client=client_of(create), model="m", retries=1)
    convo = [{"role": "system", "content": "rules"}, *USER]
    first = llm.chat(convo, tools=TOOLS)
    assert [(c.name, c.args) for c in first.calls] == [("tree", {"q": "x"})] and first.text == "Looking."
    assert [bool(k.get("tools")) for k in sent] == [True, False] and llm.tool_mode == "prompt"   # resent, not a try
    llm.chat(convo, tools=TOOLS)
    assert len(sent) == 3 and sent[2]["tools"] is None and CALL_FORMAT_IN(sent[2])         # remembered
    other = LLM(client=client_of(lambda **kw: (_ for _ in ()).throw(
        type("BadRequestError", (Exception,), {})(TOO_LONG))), model="m", retries=1)
    with pytest.raises(Exception, match="maximum context length"):
        other.chat(USER, tools=TOOLS)
    assert other.tool_mode is None                                    # a 400 about something else is not a refusal
    for text, kind in (("tool server busy", "APIConnectionError"), ("function missing", "BadRequestError")):
        odd = LLM(client=client_of(lambda **kw: (_ for _ in ()).throw(type(kind, (Exception,), {})(text))),
                  model="m", retries=1)                       # another error naming tools; a 400 with no tools sent
        with pytest.raises(Exception, match=text):
            odd.chat(USER, tools=TOOLS if kind != "BadRequestError" else None)
        assert odd.tool_mode is None
    # the request sent again through the prompt keeps the reasoning switch and stays interruptible (Esc)
    asked, stopped = [], []

    def refuse_then_wait(**kw):
        asked.append(kw)
        if kw.get("tools"):
            stopped.append(1)                                 # the user presses Esc while it is refused
            raise type("BadRequestError", (Exception,), {})(REFUSED)
        return iter(streamed(content="late"))
    keen = LLM(client=client_of(refuse_then_wait), model="m", retries=1, thinking=False)
    with pytest.raises(Exception) as e:
        keen.chat(convo, tools=TOOLS, stop=lambda: bool(stopped))
    assert type(e.value).__name__ == "Interrupted" and len(asked) == 2
    assert asked[1]["extra_body"] == switch(False) and CALL_FORMAT_IN(asked[1])
    # a server whose own tool-call parser fails on the model's output ends the stream with an error event (llama.cpp
    # 11351 + Llama-3.2-1B, recorded 2026-10-06; the OpenAI client raises APIError): that request is sent again
    # through the prompt, where peresearch reads the call itself; the next one tries the native way again (one
    # unreadable output does not make the server unable to carry tools)
    PEG = "The model produced output that does not match the expected peg-native format"

    class Broken:
        def __iter__(self):
            yield chunk(role="assistant", content=None)
            raise type("APIError", (Exception,), {})(PEG)

        def close(self):
            pass
    sent.clear()

    def parse_fails(**kw):
        sent.append(kw)
        if kw.get("tools") and len(sent) == 1:
            return Broken()
        return streamed(content='<tool_call>{"name": "tree", "arguments": {"q": "y"}}</tool_call>')
    llm = LLM(client=client_of(parse_fails), model="m", retries=1)
    got = llm.chat(convo, tools=TOOLS)
    assert [(c.name, c.args) for c in got.calls] == [("tree", {"q": "y"})] and llm.tool_mode is None
    assert [bool(k.get("tools")) for k in sent] == [True, False] and CALL_FORMAT_IN(sent[1])
    llm.chat(convo, tools=TOOLS)
    assert bool(sent[2].get("tools"))
    # the parser may fail only on what the model wrote AFTER a whole call (llama.cpp + Llama-3.2-1B streamed tree with
    # {"type": "object", "properties": {}}, then "$": "unparsed peg-native output: $", recorded 2026-10-07): the
    # calls received whole are used, nothing is sent again; a call cut by the error still goes through the prompt
    class AfterCall:
        def __init__(self, args):
            self.args = args

        def __iter__(self):
            yield chunk(tool_calls=[NS(index=0, id="c0", function=NS(name="tree", arguments=self.args[:3]))])
            yield chunk(tool_calls=[NS(index=0, id=None, function=NS(name=None, arguments=self.args[3:]))])
            raise type("APIError", (Exception,), {})(PEG)

        def close(self):
            pass
    for args, whole in (('{"q": "z"}', True), ('{"q": "z', False), ("", False)):      # cut mid-arguments, after the name
        sent.clear()
        made = LLM(client=client_of(lambda **kw: sent.append(kw) or (AfterCall(args) if kw.get("tools") else
                                                                     streamed(content="answer"))),
                   model="m", retries=1)
        got = made.chat(convo, tools=TOOLS)
        if whole:
            assert [(c.name, c.args, c.error) for c in got.calls] == [("tree", {"q": "z"}, "")] and len(sent) == 1
        else:
            assert got.calls == [] and got.text == "answer" and len(sent) == 2 and CALL_FORMAT_IN(sent[1])
    sent.clear()                                    # sent again: the reasoning switch kept, Esc still heard
    pressed = []

    def fails_then_streams(**kw):
        sent.append(kw)
        if kw.get("tools"):
            return Broken()
        pressed.append(1)                             # Esc while the request sent again streams
        return iter(streamed(content="late"))
    llm = LLM(client=client_of(fails_then_streams), model="m", retries=1, thinking=False)
    with pytest.raises(Exception) as e:
        llm.chat(convo, tools=TOOLS, stop=lambda: bool(pressed))
    assert type(e.value).__name__ == "Interrupted" and len(sent) == 2 and CALL_FORMAT_IN(sent[1])
    assert sent[1]["extra_body"] == switch(False)
    for other_error in ("Connection reset", "The model produced output"):     # not a parser's failure: raised
        bad = LLM(client=client_of(lambda **kw: (_ for _ in ()).throw(type("APIError", (Exception,), {})(other_error))),
                  model="m", retries=1)
        with pytest.raises(Exception, match=other_error):
            bad.chat(USER, tools=TOOLS)
    sent.clear()
    plain = LLM(client=client_of(lambda **kw: sent.append(kw) or Broken()), model="m", retries=1)   # no tools sent
    with pytest.raises(Exception, match="peg-native") as raised:
        plain.chat(USER)
    assert type(raised.value).__name__ == "APIError" and len(sent) == 1


def CALL_FORMAT_IN(kw):
    from peresearch.llm import CALL_FORMAT
    return CALL_FORMAT in kw["messages"][0]["content"]


NO_SPLIT = ("reasoning: none came back separately; if answers start with the model's thinking, turn on the server's "
            "reasoning parser (vLLM: --reasoning-parser)")


@pytest.mark.parametrize("replies, listed, works, want", [
    ([dict(tool_calls=[("tree", "{}")], usage=NS(prompt_tokens=9, completion_tokens=99)),
      dict(content="391", reasoning="17 times 23", usage=NS(prompt_tokens=5, completion_tokens=300))],
     [NS(id="llm", root="Qwen/Qwen3.5-0.8B", max_model_len=16384)], True,
     ["model: llm · qwen family (model table)", "tool calls: native", "reasoning: in its own field",
      "context: 16,384 tokens (from the server)", "speed: 600 tokens/s (300 tokens in 0.5 s)"]),
    ([dict(content="I would list them."),
      dict(content='<tool_call>{"name": "tree", "arguments": {}}</tool_call>', usage=NS(prompt_tokens=9, completion_tokens=99)),
      dict(content="Thinking Process: 17 * 23 …\n\n\n391", usage=NS(prompt_tokens=5, completion_tokens=10))],
     [], True,
     ["model: my-model · not in the model table: defaults",
      "tool calls: through the prompt (set PERESEARCH_LLM_TOOLS=prompt to keep it)", NO_SPLIT,
      "context: unknown (set PERESEARCH_LLM_CONTEXT, or the conversation is trimmed when refused)",
      "speed: 20 tokens/s (10 tokens in 0.5 s)"]),
    ([dict(content="No."), dict(content="Still no."), dict(content="391")], [], False,
     ["model: my-model · not in the model table: defaults",
      "tool calls: none: the model did not call the test tool; the agent can only answer from memory, without your "
      "files or the web", NO_SPLIT,
      "context: unknown (set PERESEARCH_LLM_CONTEXT, or the conversation is trimmed when refused)",
      "speed: unknown (the server reports no token counts)"]),
    ([dict(content=qwen("tree", {}), usage=NS(prompt_tokens=9, completion_tokens=99)),
      dict(content="17 times 23</think>391", usage=NS(prompt_tokens=5, completion_tokens=8))],
     [], True,
     ["model: my-model · not in the model table: defaults",
      "tool calls: native, written as text and recovered (the server's tool-call parser may be off)",
      "reasoning: inside the text, moved out of the answer",
      "context: unknown (set PERESEARCH_LLM_CONTEXT, or the conversation is trimmed when refused)",
      "speed: 16 tokens/s (8 tokens in 0.5 s)"]),
    # the model table says no tool calls: through the prompt first; when that brings no call the native way is tried
    # (the table can be wrong: Ollama's llama3.2:1b, 2026-10-06). Its "no reasoning" is not trusted either: models.dev
    # says it of Qwen3.5-0.8B, which reasons (recorded 2026-10-06), so the parser advice stays
    ([dict(content="I would list them."), dict(tool_calls=[("tree", "{}")], usage=NS(prompt_tokens=9, completion_tokens=99)),
      dict(content="391", usage=NS(prompt_tokens=5, completion_tokens=6))],
     [NS(id="gemma2-9b-it")], True,
     ["model: gemma2-9b-it · gemma family (model table)",
      "tool calls: native (set PERESEARCH_LLM_TOOLS=native to keep it)", NO_SPLIT,
      "context: 8,192 tokens (from the model table); small for the agent: long results are shown in parts and earlier turns summarized sooner — run the model with 16,384 or more (Ollama: OLLAMA_CONTEXT_LENGTH=16384)", "speed: 12 tokens/s (6 tokens in 0.5 s)"]),
    # a server that refuses `tools` (vLLM without its tool-call parser; Ollama for gemma3, recorded 2026-10-06): the
    # prompt way, and no native retry after a refusal — first, or when the table's prompt way brought no call
    ([dict(refuse=True), dict(content="I would list them."), dict(content="391")], [], False,
     ["model: my-model · not in the model table: defaults",
      "tool calls: none: the model did not call the test tool; the agent can only answer from memory, without your "
      "files or the web", NO_SPLIT,
      "context: unknown (set PERESEARCH_LLM_CONTEXT, or the conversation is trimmed when refused)",
      "speed: unknown (the server reports no token counts)"]),
    ([dict(content="I would list them."), dict(refuse=True), dict(content="Still no."), dict(content="391")],
     [NS(id="gemma2-9b-it")], False,
     ["model: gemma2-9b-it · gemma family (model table)",
      "tool calls: none: the model did not call the test tool; the agent can only answer from memory, without your "
      "files or the web", NO_SPLIT,
      "context: 8,192 tokens (from the model table); small for the agent: long results are shown in parts and earlier turns summarized sooner — run the model with 16,384 or more (Ollama: OLLAMA_CONTEXT_LENGTH=16384)", "speed: unknown (the server reports no token counts)"])])
def test_the_model_check_says_how_the_model_calls_tools_reasons_fits_and_runs(home, monkeypatch, replies, listed,
                                                                             works, want):
    """One small request with one tool (reasoning off); a second through the prompt when the model made no native
    call (kept only if it worked); one without tools and with reasoning on, to see where reasoning comes back — a
    server that removes the closing tag (vLLM without --reasoning-parser, measured 2026-10-06) cannot be split by
    the client, so the user is told how to fix the server; the context and its source; the speed, measured on that
    last request (the first may wait for the server to load the model: Ollama, 8 tokens/s cold against 150 warm,
    2026-10-07). Not a gate."""
    from peresearch.llm import check_model
    for k in ("PERESEARCH_LLM_CONTEXT", "PERESEARCH_LLM_TOOLS", "PERESEARCH_LLM_TEMPERATURE", "PERESEARCH_LLM_TOP_P"):
        monkeypatch.delenv(k, raising=False)
    queue, sent = list(replies), []

    def answer(**kw):
        sent.append(kw)
        reply = queue.pop(0)
        if "refuse" in reply:                                  # the server refuses `tools`
            raise type("BadRequestError", (Exception,), {})(REFUSED)
        return streamed(**reply)
    client = NS(chat=client_of(answer).chat,
                models=NS(list=lambda: NS(data=listed)))
    llm = LLM(client=client, model=listed[0].id if listed else "my-model", max_tokens=4096)
    clock = iter(10 + 0.5 * i for i in range(200))       # each duration the check measures: 0.5 s
    monkeypatch.setattr(llm_module.time, "monotonic", lambda: float(next(clock)))
    assert check_model(llm) == (works, want)
    assert llm.max_tokens == 4096 and all(k["max_tokens"] == 1024 for k in sent) and not queue
    probe = sent[-1]
    gemma = want[0].startswith("model: gemma")                        # only to a family with it, or an unknown one
    assert not probe.get("tools") and probe.get("extra_body") == (None if gemma else switch(True))
    off = None if gemma else switch(False)
    assert all(k.get("extra_body") == off for k in sent[:-1])         # the tool requests: reasoning off, quicker
    refused = any("refuse" in r for r in replies)                       # remembered: never tried natively again
    assert llm.tool_mode == ("prompt" if refused or want[1].startswith("tool calls: through the prompt") else
                             "native" if "PERESEARCH_LLM_TOOLS=native" in want[1] else None)


def test_without_a_model_name_the_servers_only_model_is_used_else_the_user_is_told(home, monkeypatch):
    """No default name (a default only fits one person's server): the server's only model is used (vLLM, llama.cpp,
    one model in LM Studio); several, or none listed, and the user is told what to set, with the names offered."""
    monkeypatch.delenv("PERESEARCH_LLM_MODEL", raising=False)
    monkeypatch.delenv("PERESEARCH_LLM_CONTEXT", raising=False)

    def server(*ids, fail=False):
        sent = []

        def listing():
            if fail:
                raise error("NotFoundError")
            return NS(data=[NS(id=i) for i in ids])
        return sent, NS(chat=client_of(lambda **kw: sent.append(kw) or streamed(content="ok")).chat,
                        models=NS(list=listing))
    sent, client = server("qwen3:8b")
    llm = LLM(client=client)
    llm.chat(USER)
    assert llm.model == "qwen3:8b" and sent[0]["model"] == "qwen3:8b"
    for ids, fail, said in ((("a", "b", "c", "d", "e", "f", "g"), False, "the server offers 7 models (a, b, c, d, e, …)"),
                            (("a", "b", "c", "d", "e"), False, "the server offers 5 models (a, b, c, d, e)"),
                            (("a", "b", "c", "d", "e", "f"), False, "the server offers 6 models (a, b, c, d, e, …)"),
                            ((), False, "the server lists no model"), ((), True, "the server lists no model")):
        sent, client = server(*ids, fail=fail)
        with pytest.raises(RuntimeError, match=r"^no model name: " + re.escape(said) +
                           r"; set it with /setup \(in a terminal: `peresearch setup`\)"):
            LLM(client=client).chat(USER)
        assert sent == []
    monkeypatch.setenv("PERESEARCH_LLM_MODEL", "chosen")
    sent, client = server("a", "b")
    LLM(client=client).chat(USER)
    assert sent[0]["model"] == "chosen"


# --- retries -----------------------------------------------------------------------------------------------------

def error(name, headers=None):
    e = type(name, (Exception,), {})(f"{name} happened")
    if headers is not None:                                      # where the OpenAI client keeps the HTTP answer
        e.response = NS(headers=headers)
    return e


# a 429 with the wait the server asks for (the OpenAI SDK's rule: honoured when 0 < wait ≤ 60 s)
RETRY_AFTER = {"RateLimit3s": {"retry-after": "3"}, "RateLimitMs": {"retry-after-ms": "1500"},
               "RateLimitLong": {"retry-after": "120"}}


def policy(outcomes, retries, local, thinking, fields):
    """The reference policy for one question: (each outcome met, what each request carried as (reasoning switch,
    stream usage), what the caller was told, the pauses, the optional fields after it)."""
    template, reasoning, usage = fields
    met, sent, told, pauses, refusals, attempt = [], [], [], [], 0, 0
    while True:
        o = outcomes[min(len(met), len(outcomes) - 1)]
        met.append(o)
        sent.append((thinking is not None and template, thinking is False and reasoning, usage))
        if o == "ok":
            return met, sent, told, pauses, (template, reasoning, usage)
        if o == "TooLong":                                     # over the context: no field is to blame, raised now
            return met, sent, told, pauses, fields
        if o == "BadRequestError" and thinking is not None and template:
            template = False                                   # asked again without it: not a try
            continue
        if o == "BadRequestError" and thinking is False and reasoning:
            reasoning = False
            continue
        if o == "BadRequestError" and usage:
            usage = False
            continue
        refused_here = local and o == "APIConnectionError"
        refusals += refused_here
        if o in ("ValueError", "BadRequestError") or attempt == retries - 1 or refusals == 2:
            return met, sent, told, pauses, fields             # raised: nothing worked without them, both restored
        pause = 2.0 if refused_here else 2.0 * 2 ** attempt
        asked = {"RateLimit3s": 3.0, "RateLimitMs": 1.5}.get(o)     # RateLimitLong: over 60 s, the backoff instead
        pause = asked if asked is not None else pause
        pauses.append(pause)
        told.append(f"model not ready ({'RateLimitError' if o in RETRY_AFTER else o}); retrying in {pause:.0f} s")
        attempt += 1


OUTCOMES = ["ok", "APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError", "StreamCut",
            "ValueError", "BadRequestError", "TooLong", *RETRY_AFTER]
TOO_LONG = ("Error code: 400 - {'error': {'message': \"This model's maximum context length is 16384 tokens. However, you "
            "requested 4096 output tokens and your prompt contains 12289 input tokens\"}}")   # vLLM's, measured


@given(st.lists(st.sampled_from(OUTCOMES), min_size=1, max_size=7), st.integers(1, 5), st.booleans(),
       st.sampled_from([None, True, False]), st.sampled_from(["ok", "BadRequestError"]))
@hsettings(max_examples=300, deadline=None)
@example(["APIConnectionError", "APIConnectionError", "ok"], 5, True, None, "ok")    # nothing listening here: 2 tries
@example(["APITimeoutError", "APITimeoutError", "APIConnectionError"], 3, True, None, "ok")   # refused at the last try
@example(["APIConnectionError"], 1, True, None, "ok")                   # one try: its error, not UnboundLocalError
@example(["APIConnectionError"] * 4 + ["BadRequestError", "ok"], 5, False, None, "ok")   # bug 11: a 400 at the last try
@example(["BadRequestError", "BadRequestError"], 1, False, None, "ok")   # a 400 with nothing left to drop: raised
@example(["BadRequestError", "BadRequestError", "ok"], 5, False, True, "BadRequestError")   # a server without both:
# dropped and remembered; a later 400 is a real error, and they stay dropped
@example(["BadRequestError"] * 3, 5, False, True, "ok")       # a 400 about something else (context length): both kept
@example(["BadRequestError", "BadRequestError", "APITimeoutError"], 3, False, True, "ok")   # retries + 2 requests
@example(["TooLong"], 5, False, True, "ok")       # over the context: one request, both fields kept (not two resends)
@example(["RateLimit3s", "RateLimitMs", "RateLimitLong", "ok"], 5, False, None, "ok")   # the server's wait, ≤ 60 s
def test_retries_follow_one_policy(outcomes, retries, local, thinking, then):
    """Retryable errors are tried again after wait × 2^attempt seconds; a refused local server gets one more try
    (after 2 s), and a second refusal raises; any other error, or the last try, raises at once. A 400 (BadRequest)
    explicitly naming an unsupported optional field drops only that field (template, reasoning_effort or usage):
    the request is sent again without it, which is not a try; the field stays dropped only if a request without it
    succeeds. Two questions: the second (`then`) starts from what the first left."""
    met, sent, told, slept, results, current = [], [], [], [], [], {"seq": outcomes, "n": 0}

    def create(**kw):
        seq = current["seq"]
        o = seq[min(current["n"], len(seq) - 1)]
        current["n"] += 1
        met.append(o)
        payload = kw.get("extra_body", {})
        sent.append(("chat_template_kwargs" in payload, "reasoning_effort" in payload, "stream_options" in kw))
        if o == "ok":
            return streamed(content="answer")
        if o == "StreamCut":
            return [chunk(content="half")]                       # ends without a finish reason
        if o == "TooLong":
            raise type("BadRequestError", (Exception,), {})(TOO_LONG)
        if o in RETRY_AFTER:
            raise error("RateLimitError", RETRY_AFTER[o])
        if o == "BadRequestError":
            field = next((f for f in ("chat_template_kwargs", "reasoning_effort") if f in payload),
                         "stream_options" if "stream_options" in kw else "unknown_field")
            raise type("BadRequestError", (Exception,), {})(f"Unsupported parameter: '{field}'")
        raise error(o)
    llm = LLM(client=client_of(create), model="m", retries=retries, wait=2.0, on_wait=told.append, thinking=thinking)
    llm.local = local
    real_time, real_sleep = time.time, time.sleep
    clock = [0.0]
    llm_module.time.time = lambda: clock[0]
    llm_module.time.sleep = lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s))
    try:
        for seq in (outcomes, [then]):
            current.update(seq=seq, n=0)
            try:
                results.append(llm.chat(USER))
            except Exception as e:
                results.append(e)
    finally:
        llm_module.time.time, llm_module.time.sleep = real_time, real_sleep
    first = policy(outcomes, retries, local, thinking, (True, True, True))
    second = policy([then], retries, local, thinking, first[4])
    assert met == first[0] + second[0] and sent == first[1] + second[1] and told == first[2] + second[2]
    assert (llm.template_kwargs, llm.reasoning_option, llm.usage_option) == second[4]
    for result, ref in zip(results, (first, second)):
        want = {"ok": None, "TooLong": "BadRequestError", **dict.fromkeys(RETRY_AFTER, "RateLimitError")}.get(
            ref[0][-1], ref[0][-1])
        assert result.text == "answer" if want is None else type(result).__name__ == want
    assert abs(sum(slept) - sum(first[3] + second[3])) < 1e-6 and all(0 < x <= 0.1 for x in slept)  # 0.1 s steps


def test_the_wait_a_server_asks_for_is_read_from_its_headers():
    """retry-after-ms first, then Retry-After in seconds or as an HTTP date; kept only when 0 < wait <= 60 s."""
    import email.utils
    from peresearch.llm import _asked_wait
    soon = email.utils.format_datetime(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=30),
                                       usegmt=True)
    late = email.utils.format_datetime(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=2),
                                       usegmt=True)
    ask = lambda headers: _asked_wait(error("RateLimitError", headers))    # noqa: E731
    assert ask({"retry-after": "7"}) == 7.0 and ask({"retry-after-ms": "250", "retry-after": "9"}) == 0.25
    assert ask({"retry-after": "60"}) == 60.0                                   # the longest wait taken
    assert 25 < ask({"retry-after": soon}) <= 30
    for unusable in ({"retry-after": late}, {"retry-after": "0"}, {"retry-after": "-3"}, {"retry-after": "inf"},
                     {"retry-after": "soon"}, {"retry-after-ms": "x"}, {"retry-after": "61"}, {}):
        assert ask(unusable) is None, unusable
    assert _asked_wait(error("RateLimitError")) is None and _asked_wait(ValueError()) is None    # no response at all


@pytest.mark.parametrize("message, want", [
    (TOO_LONG, (16384, 12289)),                                                        # vLLM (seen here)
    ("Error code: 400 - {'error': {'code': 400, 'message': 'request (36013 tokens) exceeds the available context size "
     "(8192 tokens), try increasing it', 'type': 'exceed_context_size_error', 'n_prompt_tokens': 36013, 'n_ctx': 8192}}",
     (8192, 36013)),                                                       # llama.cpp 11351 (recorded 2026-10-06)
    ("This model's maximum context length is 128000 tokens. However, your messages resulted in 130512 tokens.",
     (128000, 130512)),                                       # OpenAI's wording as known to me, not seen here
    ("prompt is too long: 210123 tokens > 200000 maximum", (200000, 210123)),    # Anthropic's, likewise
    ("the input exceeds the model's context window", (None, None)),                                    # numbers unsaid
    ("Invalid 'tools': an unknown field", None), ("max_tokens must be at least 1", None)])            # other 400s
def test_a_conversation_too_long_for_the_model_is_recognized_with_its_numbers(message, want):
    """The agent trims a conversation the server refused for its length, to what the server counted; any other 400
    is not mistaken for it, and an error of another type never is."""
    assert llm_module.too_long(type("BadRequestError", (Exception,), {})(message)) == want
    assert llm_module.too_long(error("APITimeoutError")) is None and llm_module.too_long(ValueError(TOO_LONG)) is None


@pytest.mark.parametrize("seq, retries", [(["APIConnectionError"], 1),
                                          (["APITimeoutError", "APITimeoutError", "APIConnectionError"], 3)])
def test_a_refused_local_server_always_raises_the_real_error(seq, retries):
    it = iter(seq)
    llm = LLM(client=client_of(lambda **kw: (_ for _ in ()).throw(error(next(it, seq[-1])))), model="m",
              retries=retries, wait=0)
    llm.local = True
    with pytest.raises(Exception) as err:
        llm.chat(USER)
    assert type(err.value).__name__ == seq[-1]


def test_without_an_endpoint_there_is_no_model_and_a_local_one_is_recognized(monkeypatch, home):
    monkeypatch.delenv("PERESEARCH_LLM_URL", raising=False)
    with pytest.raises(RuntimeError, match="no model endpoint: set PERESEARCH_LLM_URL"):
        LLM()
    for url, local in (("http://localhost:8000/v1", True), ("http://127.0.0.1:9/v1", True), ("http://[::1]:8/v1", True),
                       ("https://ws--peresearch-llm-server.modal.run/v1", False)):
        assert LLM(url=url, model="m").local is local
    monkeypatch.setenv("PERESEARCH_LLM_MODEL", "served")
    assert LLM(url="http://localhost:1/v1").model == "served"


# --- interrupts and real servers -------------------------------------------------------------------------------

class BlockingClient:
    """An OpenAI-like client whose answer streams a chunk every 50 ms for 30 s, like a model writing a long answer;
    `closed` is set when the stream is closed (what makes vLLM stop generating)."""

    def __init__(self, error=None):
        self.closed, self.closed_by, self.calls, self.error = threading.Event(), None, 0, error
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls += 1
        if self.error:
            raise self.error
        client = self

        class S:
            def __iter__(self):
                end = time.time() + 30
                while time.time() < end and not client.closed.is_set():
                    time.sleep(0.05)
                    yield streamed(content="word ")[0]

            def close(self):
                client.closed_by = threading.current_thread()
                client.closed.set()
        return S()


def test_an_interrupt_stops_a_model_call_in_flight_and_its_reader_closes_the_stream():
    """Closing the stream is what makes vLLM stop generating (0.27 s later, measured); it must be closed by the
    thread reading it (a close from another thread left vLLM generating)."""
    client = BlockingClient()
    t0 = time.time()
    with pytest.raises(Interrupted) as err:
        LLM(client=client, model="m").chat(USER, stop=lambda: time.time() - t0 > 0.3)
    assert time.time() - t0 < 1.5 and str(err.value) == "stopped by the user"
    assert client.closed.wait(1) and client.closed_by is not threading.main_thread()
    # a server that never sends a first chunk blocks the reader until the 600 s timeout: once the caller is freed,
    # that thread must be a daemon, or quitting the program would wait for it
    release, started = threading.Event(), threading.Event()

    def silent(**kw):
        started.set()
        release.wait(30)
        raise error("APITimeoutError")
    before = set(threading.enumerate())
    try:
        with pytest.raises(Interrupted):
            LLM(client=client_of(silent), model="m").chat(USER, stop=lambda: started.is_set())
        stuck = [t for t in threading.enumerate() if t not in before and t.is_alive()]
        assert stuck and all(t.daemon for t in stuck)
    finally:
        release.set()


def test_an_interrupt_also_ends_the_wait_between_retries():
    llm = LLM(client=BlockingClient(error=error("APIConnectionError")), model="m", wait=10)
    t0 = time.time()
    with pytest.raises(Interrupted) as err:
        llm.chat(USER, stop=lambda: time.time() - t0 > 0.3)
    assert str(err.value) == "stopped by the user"
    assert time.time() - t0 < 1.5


def test_a_request_that_fails_in_its_thread_raises_in_the_caller():
    with pytest.raises(ValueError, match="bad"):
        LLM(client=client_of(lambda **kw: (_ for _ in ()).throw(ValueError("bad"))), model="m").chat(USER, stop=lambda: False)


@pytest.fixture
def flaky_server():
    """An OpenAI-compatible server on localhost whose first `drops` answers are cut off mid-stream (a container
    that dies while answering); the next ones stream to the end. The real OpenAI client talks to it."""
    import http.server
    import socket

    state = {"drops": 1, "requests": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            state["requests"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def send(delta, finish=None):
                c = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.flush()
            send({"content": "Hel"})
            if state["requests"] <= state["drops"]:
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            send({"content": "lo."}, finish="stop")
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", state
    server.shutdown()


def test_a_real_cut_stream_is_asked_again_and_a_stream_cut_every_time_is_an_error(flaky_server):
    """The OpenAI client raises nothing when a stream ends early; only the missing finish reason tells."""
    url, state = flaky_server
    reply = LLM(url=url, model="m", wait=0).chat(USER)
    assert reply.text == "Hello." and reply.finish == "stop" and state["requests"] == 2
    state.update(drops=99, requests=0)
    with pytest.raises(StreamCut):
        LLM(url=url, model="m", wait=0, retries=3).chat(USER)
    assert state["requests"] == 3


def test_a_local_server_that_is_not_running_fails_fast():
    """Nothing listening on localhost will not start listening by waiting: 2 tries, not 2.5 minutes of retries."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]                                 # closed again: nothing listens there
    waits, t0 = [], time.time()
    with pytest.raises(Exception) as err:
        LLM(url=f"http://127.0.0.1:{port}/v1", model="m", on_wait=waits.append).chat(USER)
    assert type(err.value).__name__ == "APIConnectionError" and time.time() - t0 < 6
    assert waits == ["model not ready (APIConnectionError); retrying in 2 s"]


@pytest.mark.parametrize("delay, rest, want", [
    (0.8, 0.0, ["the model has not started answering yet (a server that is starting can take minutes)", "Thinking"]),
    (0.0, 0.0, []), (0.3, 0.0, [])])                                  # 0.3 s is within slow_start (0.5 s)
def test_a_model_slow_to_start_answering_is_announced_once_each_way(delay, rest, want):
    def create(**kw):
        def chunks():
            time.sleep(delay)
            parts = streamed(content="Hello.")
            yield parts[0]
            time.sleep(rest)
            yield from parts[1:]
        return chunks()
    waits = []
    r = LLM(client=client_of(create), model="m", slow_start=0.5, on_wait=waits.append).chat(USER, stop=lambda: False)
    assert r.text == "Hello." and waits == want


def test_the_answer_starting_is_announced_while_it_streams_not_after():
    def create(**kw):
        def chunks():
            time.sleep(0.4)
            parts = streamed(content="Hello.")
            yield parts[0]
            time.sleep(0.8)                                         # the rest takes a while
            yield from parts[1:]
        return chunks()
    said, t0 = [], time.time()
    LLM(client=client_of(create), model="m", slow_start=0.2, on_wait=lambda m: said.append((m, time.time() - t0))).chat(
        USER, stop=lambda: False)
    assert [m for m, _ in said][-1] == "Thinking" and said[-1][1] < 1.0


# --- recorded real servers -----------------------------------------------------------------------------------
# Exchanges recorded from real servers (runs/serve/compat-matrix-2026-10-06.py: Ollama, llama.cpp, vLLM with small
# models; runs/serve/compat-curate-2026-10-06.py copies them here), replayed offline through the real OpenAI client.

RECORDED = sorted((Path(__file__).parent / "data" / "servers").glob("*.json.gz"))


def replay(rec):
    """An OpenAI client whose server is the recording: each request must be the next one recorded (method, path)."""
    from openai import OpenAI
    queue = list(rec["exchanges"])

    def answer(request):
        assert queue, f"a request the real run did not make: {request.method} {request.url.path}"
        x = queue.pop(0)
        assert (request.method, request.url.path) == (x["method"], x["path"])
        return httpx.Response(x["status"], headers={"content-type": x["content_type"]}, content=x["body"].encode())
    client = OpenAI(base_url="http://recorded.test/v1", api_key="none", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(answer)))
    return client, queue


def as_live(r) -> dict:
    """A Reply as the matrix wrote it down."""
    import dataclasses
    d = dataclasses.asdict(r)
    d["calls"] = [{**c, "recovered": c["id"].startswith("text-")} for c in d["calls"]]
    return json.loads(json.dumps(d))


@pytest.mark.parametrize("path", RECORDED, ids=lambda p: p.name.removesuffix(".json.gz"))
def test_a_recorded_real_server_exchange_reads_back_as_it_did_live(home, monkeypatch, path):
    """What peresearch made of a real server's answers — the model list, the context it runs with, the model check,
    an agent's tool call and the answer to its result, a refusal for length, a parser failure — is what it makes of
    the same bytes now: a change that misreads any real server's stream, error or listing fails here."""
    import gzip
    from peresearch import agent
    from peresearch.llm import check_model, too_long
    from peresearch.tools import Toolbox
    for k in ("PERESEARCH_LLM_MODEL", "PERESEARCH_LLM_CONTEXT", "PERESEARCH_LLM_TOOLS", "PERESEARCH_LLM_THINKING",
              "PERESEARCH_LLM_TEMPERATURE", "PERESEARCH_LLM_TOP_P"):
        monkeypatch.delenv(k, raising=False)
    rec = json.loads(gzip.decompress(path.read_bytes()))
    client, queue = replay(rec)
    model = rec["used"] if rec["named_by_us"] else None
    schemas = Toolbox(None, None, roots=[]).schemas(web=False)
    live = rec["outcome"]
    if rec["case"] == "models":
        llm = LLM(client=client, model=model)
        assert ([m.id for m in llm._models()], llm._name()) == (live["listed"], live["used"])
        assert (llm.context_length(), llm.context_source()) == (rec["context"]["tokens"], rec["context"]["source"])
    elif rec["case"] == "check":
        llm = LLM(client=client, model=rec["used"])
        monkeypatch.setattr(llm_module.time, "monotonic", lambda: 0.0)     # the speed line is the clock's
        works, lines = check_model(llm)
        assert (works, lines[:-1], llm.tool_mode) == (live["works"], live["lines"][:-1], live["tool_mode_after"])
    elif rec["case"] == "agent":
        llm = LLM(client=client, model=rec["used"], max_tokens=2048)
        assert as_live(llm.chat(rec["messages"][:2], schemas)) == {"refusal": "", **live["first"]}
        if "second" in live:
            assert as_live(llm.chat(rec["messages"], schemas)) == {"refusal": "", **live["second"]}
        assert llm.tool_mode == live["tool_mode"]
    elif rec["case"] == "too-long":
        llm = LLM(client=client, model=rec["used"], max_tokens=16, retries=1)
        long = [{"role": "user", "content": "Summarize: " + "the quick brown fox jumps over the lazy dog " * 4000}]
        if live["refused"]:
            with pytest.raises(Exception) as e:
                llm.chat(long)
            assert list(too_long(e.value)) == live["read"]
        else:                                  # a server that cuts silently: only the count shows it
            r = llm.chat(long)
            assert (r.prompt_tokens, r.finish) == (live["prompt_tokens"], live["finish"])
    else:                                  # the server's tool-call parser failing, before a call or after a whole one
        assert rec["case"] in ("parse-failure", "parse-failure-after-call")
        llm = LLM(client=client, model=None, max_tokens=rec["max_tokens"])
        assert as_live(llm.chat(rec["messages"], rec["tools"], thinking=rec["thinking"])) == {"refusal": "", **live["reply"]}
        assert any("peg-native" in x["body"] for x in rec["exchanges"]) and llm.tool_mode is None
    assert queue == [], "the real run made requests this one did not"
