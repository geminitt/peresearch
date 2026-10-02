"""Contracts of the model client (peresearch.llm).

- Tool calls a model leaves as text are recovered exactly and removed from the text (a property).
- A streamed reply is put back together exactly, whatever the pieces; the stream is always closed by its reader;
  one without a finish reason is a cut, not an answer (a property).
- Everything sent is redacted; nothing else in a message changes.
- Retries follow one policy: which errors, how many tries, which pauses, what the caller is told (a property).
- The daily budget is the union of request spans plus the idle time, today only (a property against a reference).
- An interrupt frees the caller at once, in flight or between retries; real servers: a cut stream, nothing
  listening on this machine, a slow start.
"""

import json
import threading
import time
from types import SimpleNamespace as NS

import pytest
from hypothesis import example, given, settings as hsettings
from hypothesis import strategies as st

from peresearch import llm as llm_module
from peresearch.llm import (LLM, Budget, BudgetExceeded, Interrupted, StreamCut, ToolCall, _outbound, _read_stream,
                            parse_text_calls, strip_text_calls)
from tests.helpers import streamed
from tests.test_guard import CANARIES

USER = [{"role": "user", "content": "x"}]


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


@given(calls, prose, prose, st.sampled_from([qwen, hermes]))
@example([("web_search", {"query": "bpe history"})], "", "", qwen)
@example([("read", {"path": "a.md", "start": 3})], "Looking. ", "", hermes)
def test_tool_calls_left_as_text_are_recovered_and_removed(cs, before, after, write):
    text = before + "\n".join(write(n, a) for n, a in cs) + after
    got = parse_text_calls(text)
    assert [(c.id, c.name, c.args) for c in got] == [(f"text-{i}", n, a) for i, (n, a) in enumerate(cs)]
    assert strip_text_calls(text) == (before + "\n" * (len(cs) - 1) + after).strip()


def test_qwen_calls_win_over_hermes_ones_and_broken_json_is_skipped():
    mixed = qwen("a", {"x": 1}) + hermes("b", {"y": 2})
    assert [c.name for c in parse_text_calls(mixed)] == ["a"]
    assert parse_text_calls('<tool_call>{"name": "b", "arguments": {}}</tool_call><tool_call>{broken</tool_call>') == \
        [ToolCall("text-0", "b", {})]
    assert parse_text_calls('<tool_call>["not a call"]</tool_call>') == []
    assert parse_text_calls('<tool_call>{"arguments": {}}</tool_call>') == [] and parse_text_calls(None) == []
    assert strip_text_calls(None) == "" and parse_text_calls(hermes("n", None))[0].args == {}


@pytest.mark.xfail(strict=True, reason="BUG found 2026-10-02: a broken Hermes call followed by a valid one is read as one "
                   "match running across both tags, so the valid call is lost; fix awaits the owner's OK")
def test_a_broken_text_call_does_not_hide_the_next_one():
    assert parse_text_calls('<tool_call>{broken</tool_call><tool_call>{"name": "b", "arguments": {}}</tool_call>') == \
        [ToolCall("text-0", "b", {})]


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
    (True, "off", None, True), (None, "on", False, False), (None, "maybe", None, None)])
def test_the_request_carries_the_model_settings_and_the_reasoning_switch(home, monkeypatch, own, setting, call, want):
    monkeypatch.setenv("PERESEARCH_LLM_THINKING", setting)
    sent = []
    llm = LLM(client=client_of(lambda **kw: sent.append(kw) or streamed(content="ok")), model="m", thinking=own,
              max_tokens=99, temperature=0.1, top_p=0.5)
    llm.chat(USER, tools=[{"type": "function"}], thinking=call)
    llm.chat(USER, tools=[])
    kw = sent[0]
    assert {k: kw[k] for k in ("model", "max_tokens", "temperature", "top_p", "tools", "stream", "stream_options")} == {
        "model": "m", "max_tokens": 99, "temperature": 0.1, "top_p": 0.5, "tools": [{"type": "function"}],
        "stream": True, "stream_options": {"include_usage": True}}
    assert kw.get("extra_body") == (None if want is None else {"chat_template_kwargs": {"enable_thinking": want}})
    assert sent[1]["tools"] is None


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
    assert (llm.max_tokens, llm.temperature, llm.top_p, llm.retries, llm.wait, llm.slow_start) == (4096, 0.6, 0.95, 5, 10.0, 5.0)
    injected = LLM(client=client_of(None), model="m")
    assert injected.local is False and injected.budget is None and injected.on_wait("x") is None


# --- retries -----------------------------------------------------------------------------------------------------

def error(name):
    return type(name, (Exception,), {})(f"{name} happened")


@given(st.lists(st.sampled_from(["ok", "APIConnectionError", "APITimeoutError", "InternalServerError",
                                 "RateLimitError", "StreamCut", "ValueError"]), min_size=1, max_size=6),
       st.integers(1, 5), st.booleans())
@hsettings(max_examples=150, deadline=None)
@example(["APIConnectionError", "APIConnectionError", "ok"], 5, True)     # nothing listening here: 2 tries
def test_retries_follow_one_policy(outcomes, retries, local):
    """Retryable errors are tried again after wait × 2^attempt seconds (2 s on a refused local server, which gets one
    more try only); any other error, or the last try, raises at once."""
    if local:                                         # a refusal after the second try: see the bug test below
        retries = max(retries, 2)
        outcomes = outcomes[:2] + [o for o in outcomes[2:] if o != "APIConnectionError"] or ["ok"]
    tries, waits, slept = [], [], []

    def create(**kw):
        o = outcomes[min(len(tries), len(outcomes) - 1)]
        tries.append(o)
        if o == "ok":
            return streamed(content="answer")
        if o == "StreamCut":
            return [chunk(content="half")]                       # ends without a finish reason
        raise error(o)
    llm = LLM(client=client_of(create), model="m", retries=retries, wait=2.0, on_wait=waits.append)
    llm.local = local
    real_time, real_sleep = time.time, time.sleep
    clock = [0.0]
    llm_module.time.time = lambda: clock[0]
    llm_module.time.sleep = lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s))
    try:
        result = llm.chat(USER)
    except Exception as e:
        result = e
    finally:
        llm_module.time.time, llm_module.time.sleep = real_time, real_sleep
    # the reference policy
    want_tries, want_waits, pauses = [], [], []
    for attempt in range(retries):
        o = outcomes[min(attempt, len(outcomes) - 1)]
        want_tries.append(o)
        if o == "ok":
            break
        refused_here = local and o == "APIConnectionError"
        if o == "ValueError" or attempt == (1 if refused_here else retries - 1):
            break
        pause = 2.0 if refused_here else 2.0 * 2 ** attempt
        pauses.append(pause)
        want_waits.append(f"model not ready ({o}); retrying in {pause:.0f} s")
    assert tries == want_tries and waits == want_waits
    last = want_tries[-1]
    if last == "ok":
        assert result.text == "answer"
    else:
        assert type(result).__name__ == last
    assert abs(sum(slept) - sum(pauses)) < 1e-6 and all(0 < x <= 0.1 for x in slept)   # waited in steps of 0.1 s


@pytest.mark.xfail(strict=True, reason="BUG found 2026-10-02: a refused local server is raised only at exactly the second "
                   "try; refused at the first of one try, or after other errors at a later try, the retry loop runs out "
                   "and UnboundLocalError comes out instead of the real error; fix awaits the owner's OK")
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


def test_a_server_without_template_switches_or_stream_usage_is_asked_without_them():
    seen = []

    def create(**kw):
        seen.append(("extra_body" in kw, "stream_options" in kw))
        if "extra_body" in kw or "stream_options" in kw:
            raise error("BadRequestError")
        return streamed(content="ok")
    llm = LLM(client=client_of(create), model="m", thinking=False)
    assert llm.chat(USER).text == "ok" and seen == [(True, True), (False, True), (False, False)]
    llm.chat(USER)
    assert seen[-1] == (False, False) and not llm.template_kwargs and not llm.usage_option   # remembered
    llm.client.chat.completions.create = lambda **kw: (_ for _ in ()).throw(error("BadRequestError"))
    with pytest.raises(Exception, match="BadRequestError happened"):     # nothing left to drop: a real error
        llm.chat(USER)


def test_without_an_endpoint_there_is_no_model_and_a_local_one_is_recognized(monkeypatch, home):
    monkeypatch.delenv("PERESEARCH_LLM_URL", raising=False)
    with pytest.raises(RuntimeError, match="no model endpoint: set PERESEARCH_LLM_URL"):
        LLM()
    for url, local in (("http://localhost:8000/v1", True), ("http://127.0.0.1:9/v1", True), ("http://[::1]:8/v1", True),
                       ("https://ws--peresearch-llm-server.modal.run/v1", False)):
        assert LLM(url=url, model="m").local is local
    monkeypatch.setenv("PERESEARCH_LLM_MODEL", "served")
    assert LLM(url="http://localhost:1/v1").model == "served"
    assert LLM(url="https://x.modal.run/v1").budget is not None and LLM(url="http://localhost:1/v1").budget is None


# --- the daily budget -----------------------------------------------------------------------------------------------

DAY = 1_800_000_000.0                       # a fixed instant; spans are placed around it


@given(st.lists(st.tuples(st.floats(-90_000, 90_000), st.floats(0, 4000)), max_size=8), st.floats(0, 900))
def test_the_budget_bills_the_union_of_request_spans_plus_idle_today_only(tmp_path_factory, spans, scaledown):
    path = tmp_path_factory.mktemp("b") / "usage.jsonl"
    b = Budget(path, usd_per_hour=3.6, cap_usd=1e9, scaledown=scaledown, clock=lambda: DAY)
    import datetime
    today = datetime.date.fromtimestamp(DAY)
    for start, length in spans:
        b.record(DAY + start, DAY + start + length)
    ivs = sorted((DAY + s, DAY + s + n + scaledown) for s, n in spans if datetime.date.fromtimestamp(DAY + s) == today)
    total, end = 0.0, -1e18                                            # union length, merged left to right
    for a, z in ivs:
        if a > end:
            total, end = total + z - a, z
        elif z > end:
            total, end = total + z - end, z
    assert abs(b.spent() - total / 3600 * 3.6) < 1e-6


def test_the_cap_stops_the_model_before_it_is_called_and_every_request_is_counted(home):
    now = [DAY]
    b = Budget(home / "usage.jsonl", usd_per_hour=3.6, cap_usd=0.99, scaledown=300, clock=lambda: now[0])
    b.record(DAY, DAY + 600)                                           # 900 s at $3.6/h = $0.90: under the cap
    b.check()
    b.record(DAY + 600, DAY + 700)                                     # 1,000 s = $1.00
    with pytest.raises(BudgetExceeded, match=r"at \$1\.00 of the \$0\.99 cap \(PERESEARCH_BUDGET_USD_PER_DAY\)"):
        b.check()
    now[0] += 86_400
    b.check()                                                          # a new day starts from zero
    with open(home / "usage.jsonl", "a") as f:                         # a torn line does not hide what follows
        f.write("torn line\n{\"x\": 1}\n" + json.dumps({"span": [now[0], now[0] + 600]}) + "\n")
    assert abs(b.spent() - 900 / 3600 * 3.6) < 1e-9
    exact = Budget(home / "deep" / "er" / "usage.jsonl", usd_per_hour=3600, cap_usd=1.0, scaledown=0, clock=lambda: DAY)
    exact.record(DAY, DAY + 1)                                         # $1 exactly: the cap is reached
    with pytest.raises(BudgetExceeded):
        exact.check()
    calls = []
    llm = LLM(client=client_of(lambda **kw: calls.append(1) or (_ for _ in ()).throw(error("ValueError"))), model="m",
              budget=Budget(home / "u2.jsonl", 3.6, 1.0, 0))
    before = time.time()
    with pytest.raises(Exception, match="ValueError"):
        llm.chat(USER)
    lines = (home / "u2.jsonl").read_text().splitlines()               # a failed request is billed too
    assert len(lines) == 1
    a, z = json.loads(lines[0])["span"]
    assert before <= a <= z <= time.time()
    capped = LLM(client=client_of(lambda **kw: calls.append(1)), model="m", budget=Budget(home / "u3.jsonl", 3600, 0.5, 0))
    capped.budget.record(time.time(), time.time() + 1)
    calls.clear()
    with pytest.raises(BudgetExceeded):
        capped.chat(USER)
    assert calls == []


def test_the_budget_applies_to_a_modal_endpoint_or_a_configured_rate(home, monkeypatch):
    for k in ("PERESEARCH_GPU_USD_PER_HOUR", "PERESEARCH_BUDGET_USD_PER_DAY", "PERESEARCH_SCALEDOWN_S"):
        monkeypatch.delenv(k, raising=False)
    assert Budget.from_settings("http://localhost:8000/v1") is None and Budget.from_settings(None) is None
    b = Budget.from_settings("https://ws--peresearch-llm-server.modal.run/v1")
    assert (b.path, b.rate, b.cap, b.scaledown) == (home / "usage.jsonl", 1.95, 1.0, 300)
    monkeypatch.setenv("PERESEARCH_GPU_USD_PER_HOUR", "2.5")
    monkeypatch.setenv("PERESEARCH_BUDGET_USD_PER_DAY", "3")
    monkeypatch.setenv("PERESEARCH_SCALEDOWN_S", "60")
    b = Budget.from_settings("http://localhost:8000/v1")
    assert (b.rate, b.cap, b.scaledown) == (2.5, 3.0, 60.0)


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
    (0.8, 0.0, ["the model has not started answering yet (a Modal container can take minutes to wake up)", "Thinking"]),
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
