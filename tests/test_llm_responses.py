"""The Responses API route behind the same gateway (2026-10-14): a model that takes tools only through the Responses
API (OpenAI documents GPT 6.1 Sol so) still gets native tool calls, the user still giving only a URL and a key.

Offline fixtures only, shaped by the installed OpenAI SDK's event types (openai 3.20.0); no provider was called."""

import json
from types import SimpleNamespace as NS

import pytest

from peresearch.llm import LLM

USER = [{"role": "system", "content": "rules"}, {"role": "user", "content": "what is bpe?"}]
TOOLS = [{"type": "function", "function": {"name": "search_notes", "description": "search the notes",
                                           "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}]


class BadRequestError(Exception):
    def __init__(self, message, body=None):
        super().__init__(message)
        self.body = body


class NotFoundError(Exception):
    pass


TOOLS_UNSUPPORTED = BadRequestError("Error code: 400 - tools is not supported with this model",
                                    {"error": {"message": "tools is not supported with this model",
                                               "param": "tools", "code": "unsupported_parameter"}})


def events(text="", calls=(), usage=(12, 5), end="response.completed", reason=None, reasoning=""):
    out = [NS(type="response.reasoning_summary_text.delta", delta=reasoning)] if reasoning else []
    out += [NS(type="response.output_text.delta", delta=text[i:i + 4]) for i in range(0, len(text), 4)]
    for n, (call_id, name, args) in enumerate(calls, start=1):
        out.append(NS(type="response.output_item.added", output_index=n,
                      item=NS(type="function_call", call_id=call_id, name=name, arguments="")))
        out += [NS(type="response.function_call_arguments.delta", output_index=n, delta=args[i:i + 5])
                for i in range(0, len(args), 5)]
        out.append(NS(type="response.function_call_arguments.done", output_index=n, arguments=args))
    out.append(NS(type=end, response=NS(usage=NS(input_tokens=usage[0], output_tokens=usage[1]),
                                        incomplete_details=NS(reason=reason) if reason else None)))
    return out


def client(chat, responses=None, base_url="https://api.openai.com/v1"):
    c = NS(chat=NS(completions=NS(create=chat)), base_url=base_url)
    if responses is not None:
        c.responses = NS(create=responses)
    return c


def refuse_tools(**kw):
    if kw.get("tools"):
        raise TOOLS_UNSUPPORTED
    raise AssertionError("the chat route is not asked again once Responses works")


@pytest.fixture(autouse=True)
def plain(monkeypatch):
    for name in ("PERESEARCH_LLM_TOOLS", "PERESEARCH_LLM_CONTEXT", "PERESEARCH_LLM_API"):
        monkeypatch.delenv(name, raising=False)


def test_tools_refused_by_chat_completions_go_through_the_responses_api_from_then_on(home):
    sent = []

    def responses(**kw):
        sent.append(kw)
        if len(sent) == 1:
            return iter(events(calls=[("call_1", "search_notes", '{"query": "bpe"}')]))
        return iter(events(text="BPE merges pairs [N1].", reasoning="look first"))
    llm = LLM(client=client(refuse_tools, responses), model="gpt-6.1-sol", retries=1)
    first = llm.chat(USER, tools=TOOLS)
    assert [(c.id, c.name, c.args) for c in first.calls] == [("call_1", "search_notes", {"query": "bpe"})]
    assert (first.prompt_tokens, first.completion_tokens, first.finish) == (12, 5, "tool_calls")
    request = sent[0]
    assert request["model"] == "gpt-6.1-sol" and request["stream"] is True and request["store"] is False
    assert request["instructions"] == "rules" and request["input"] == [{"role": "user", "content": "what is bpe?"}]
    assert request["tools"] == [{"type": "function", "name": "search_notes", "description": "search the notes",
                                 "parameters": TOOLS[0]["function"]["parameters"]}]
    followed = USER + [{"role": "assistant", "content": "", "tool_calls": [
                           {"id": "call_1", "type": "function",
                            "function": {"name": "search_notes", "arguments": '{"query": "bpe"}'}}]},
                       {"role": "tool", "tool_call_id": "call_1", "content": "[N1] bpe.md: BPE merges pairs"}]
    second = llm.chat(followed, tools=TOOLS)                          # remembered: straight to Responses
    assert second.text == "BPE merges pairs [N1]." and second.reasoning == "look first" and second.finish == "stop"
    assert sent[1]["input"][1:] == [{"type": "function_call", "call_id": "call_1", "name": "search_notes",
                                     "arguments": '{"query": "bpe"}'},
                                    {"type": "function_call_output", "call_id": "call_1",
                                     "output": "[N1] bpe.md: BPE merges pairs"}]


def test_an_endpoint_without_the_responses_api_falls_back_to_tools_in_the_prompt(home):
    chat_sent = []

    def chat(**kw):
        chat_sent.append(kw)
        if kw.get("tools"):
            raise TOOLS_UNSUPPORTED
        from tests.helpers import streamed
        return streamed(content='<tool_call>{"name": "search_notes", "arguments": {"query": "bpe"}}</tool_call>')

    def responses(**kw):
        raise NotFoundError("Error code: 404 - not found")
    llm = LLM(client=client(chat, responses), model="local-model", retries=1)
    reply = llm.chat(USER, tools=TOOLS)
    assert [(c.name, c.args) for c in reply.calls] == [("search_notes", {"query": "bpe"})]
    assert llm.tool_mode == "prompt" and [bool(k.get("tools")) for k in chat_sent] == [True, False]
    llm.chat(USER, tools=TOOLS)                                       # remembered: no second Responses try
    assert len(chat_sent) == 3


def test_a_client_without_responses_keeps_the_prompt_fallback(home):
    from tests.helpers import streamed
    calls = []

    def chat(**kw):
        calls.append(bool(kw.get("tools")))
        if kw.get("tools"):
            raise TOOLS_UNSUPPORTED
        return streamed(content="no tools here")
    llm = LLM(client=client(chat), model="m", retries=1)
    assert llm.chat(USER, tools=TOOLS).text == "no tools here" and calls == [True, False]


def test_a_reply_cut_at_its_token_limit_and_a_failure_are_reported(home, monkeypatch):
    monkeypatch.setenv("PERESEARCH_LLM_API", "responses")              # chosen: chat completions never asked
    replies = iter([events(text="partial answer", end="response.incomplete", reason="max_output_tokens"),
                    [NS(type="response.output_text.delta", delta="half"), NS(type="error", message="server overloaded")]])
    llm = LLM(client=client(lambda **kw: pytest.fail("chat completions asked"),
                            lambda **kw: iter(next(replies))), model="m", retries=1)
    cut = llm.chat(USER, tools=TOOLS)
    assert cut.text == "partial answer" and cut.finish == "length"
    with pytest.raises(Exception, match="server overloaded"):
        llm.chat(USER, tools=TOOLS)


def test_the_route_setting_is_checked(home, monkeypatch):
    monkeypatch.setenv("PERESEARCH_LLM_API", "soap")
    with pytest.raises(ValueError, match="PERESEARCH_LLM_API"):
        LLM(client=client(lambda **kw: None), model="m")


def test_what_leaves_through_responses_passes_the_guard_first(home):
    from tests.test_guard import CANARIES
    sent = []
    secret = CANARIES["openai-key"]
    llm = LLM(client=client(refuse_tools, lambda **kw: sent.append(kw) or iter(events(text="ok"))), model="m", retries=1)
    llm.chat([{"role": "user", "content": f"my key is {secret}"}], tools=TOOLS)
    assert secret not in json.dumps(sent[0])


def test_a_stream_without_its_end_is_a_cut_connection(home, monkeypatch):
    monkeypatch.setenv("PERESEARCH_LLM_API", "responses")
    llm = LLM(client=client(lambda **kw: None, lambda **kw: iter([NS(type="response.output_text.delta", delta="x")])),
              model="m", retries=1)
    with pytest.raises(Exception, match="cut mid-answer"):
        llm.chat(USER, tools=TOOLS)


def test_another_server_refusing_tools_goes_straight_to_the_prompt(home):
    """Ollama refuses `tools` for a model without them ("gemma3:1b does not support tools", recorded live): the
    Responses API would not help, so only OpenAI's own endpoint, where Responses-only models live, gets that try."""
    from tests.helpers import streamed
    asked = []

    def chat(**kw):
        if kw.get("tools"):
            raise TOOLS_UNSUPPORTED
        return streamed(content="no tools here")
    llm = LLM(client=client(chat, lambda **kw: asked.append(kw), base_url="http://localhost:11434/v1"), model="m",
              retries=1)
    assert llm.chat(USER, tools=TOOLS).text == "no tools here" and asked == [] and llm.tool_mode == "prompt"


# Paths a mutation run of the route found untested (2026-10-14): request shapes, events, failures and settings.

from peresearch.llm import Interrupted, StreamCut, _read_responses, responses_request


def test_a_request_keeps_every_part_and_fills_what_is_missing():
    messages = [{"role": "system", "content": "rules"}, {"role": "system", "content": "more rules"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "looking", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a"}'}},
                    {"type": "function", "function": {}}]},
                {"role": "tool", "content": "text of a"}, {"role": "assistant", "content": None}]
    tools = [{"type": "function", "function": {"name": "read"}}, {"type": "web", "function": {"name": "skip"}}]
    assert responses_request(messages, tools) == {
        "input": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "looking"},
                  {"type": "function_call", "call_id": "c1", "name": "read", "arguments": '{"path": "a"}'},
                  {"type": "function_call", "call_id": "", "name": "", "arguments": "{}"},
                  {"type": "function_call_output", "call_id": "", "output": "text of a"}],
        "store": False, "instructions": "rules\n\nmore rules",
        "tools": [{"type": "function", "name": "read", "description": "",
                   "parameters": {"type": "object", "properties": {}}}]}
    assert responses_request([{"role": "user", "content": "q"}], None) == {
        "input": [{"role": "user", "content": "q"}], "store": False}


class Stream(list):
    closed = False

    def close(self):
        self.closed = True


def test_events_become_the_reply_as_a_chat_stream_would():
    stream = Stream([NS(type="response.reasoning_text.delta", delta="think "),
                     NS(type="response.reasoning_summary_text.delta", delta="sum"),
                     NS(type="response.refusal.delta", delta="I can't"),
                     NS(type="response.output_item.added", output_index=0, item=NS(type="message")),
                     NS(type="response.output_item.added", output_index=1,
                        item=NS(type="function_call", call_id="c9", name=None, arguments=None)),
                     NS(type="response.function_call_arguments.delta", output_index=2, delta='{"a"'),
                     NS(type="response.function_call_arguments.delta", output_index=2, delta=': 1}'),
                     NS(type="response.function_call_arguments.done", output_index=3, arguments='{"b": 2}'),
                     NS(no_type=True),
                     NS(type="response.completed", response=NS(usage=None))])
    first = []
    got = _read_responses(stream, lambda: False, lambda: first.append(1))
    assert got == {"text": "", "reasoning": "think sum", "refusal": "I can't", "finish": "tool_calls",
                   "prompt_tokens": 0, "completion_tokens": 0,
                   "calls": [{"id": "c9", "name": "", "args": ""},           # no id: named by output index
                             {"id": "call-2", "name": "", "args": '{"a": 1}'},
                             {"id": "call-3", "name": "", "args": '{"b": 2}'}]}
    assert stream.closed and first == [1]


@pytest.mark.parametrize("details, finish", [(NS(reason="content_filter"), "content_filter"),
                                             (NS(reason="max_output_tokens"), "length"), (None, "length")])
def test_an_incomplete_reply_says_why(details, finish):
    stream = [NS(type="response.incomplete", response=NS(usage=NS(input_tokens=3, output_tokens=4),
                                                         incomplete_details=details))]
    got = _read_responses(stream, None)
    assert (got["finish"], got["prompt_tokens"], got["completion_tokens"]) == (finish, 3, 4)


@pytest.mark.parametrize("event, said", [(NS(type="response.failed", response=NS(error="quota exceeded")), "quota exceeded"),
                                         (NS(type="response.failed", response=NS(error=None)),
                                          "the Responses API reported a failure"),
                                         (NS(type="error", message="bad"), "bad")])
def test_a_failure_raises_with_what_came_before(event, said):
    stream = Stream([NS(type="response.output_text.delta", delta="half"),
                     NS(type="response.refusal.delta", delta="no"), event])
    with pytest.raises(Exception) as caught:
        _read_responses(stream, None)
    assert str(caught.value) == said and stream.closed
    assert caught.value.partial == {"text": "half", "reasoning": "", "refusal": "no", "calls": [], "finish": "",
                                    "prompt_tokens": 0, "completion_tokens": 0}


def test_a_cut_stream_keeps_its_part_and_a_stop_interrupts():
    stream = [NS(type="response.output_text.delta", delta="half"),
              NS(type="response.completed", response=NS(usage=NS(input_tokens=1, output_tokens=2))),
              NS(type="response.output_text.delta", delta="more")]
    assert _read_responses(stream, None)["text"] == "halfmore"
    with pytest.raises(StreamCut) as cut:
        _read_responses([NS(type="response.output_text.delta", delta="x")], None)
    assert str(cut.value) == "the reply stream ended without a finish reason: the connection was cut mid-answer"
    assert cut.value.partial["text"] == "x"
    with pytest.raises(Interrupted, match="^stopped by the user$"):
        _read_responses([NS(type="response.output_text.delta", delta="x")], lambda: True)


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_the_reply_budget_and_sampling_reach_the_responses_api(home, monkeypatch, field):
    monkeypatch.setenv("PERESEARCH_LLM_API", "responses")
    sent = []
    llm = LLM(client=client(lambda **kw: None, lambda **kw: sent.append(kw) or iter(events(text="ok"))),
              model="m", retries=1, max_tokens=64, temperature=0.25, top_p=0.5)
    llm._token_field = field
    llm.chat(USER)
    assert {k: sent[0][k] for k in ("max_output_tokens", "temperature", "top_p", "stream")} == {
        "max_output_tokens": 64, "temperature": 0.25, "top_p": 0.5, "stream": True}


def test_chosen_routes_are_accepted_and_chat_never_tries_responses(home, monkeypatch):
    for value in ("auto", "chat", "responses"):
        monkeypatch.setenv("PERESEARCH_LLM_API", value)
        assert LLM(client=client(lambda **kw: None), model="m").api == value
    monkeypatch.setenv("PERESEARCH_LLM_API", "chat")
    from tests.helpers import streamed
    asked = []

    def chat(**kw):
        if kw.get("tools"):
            raise TOOLS_UNSUPPORTED
        return streamed(content="prompt route")
    llm = LLM(client=client(chat, lambda **kw: asked.append(kw)), model="m", retries=1)
    assert llm.chat(USER, tools=TOOLS).text == "prompt route" and asked == [] and llm.tool_mode == "prompt"


def test_a_parser_refusal_at_openai_and_a_refused_responses_request_keep_the_prompt(home):
    from tests.helpers import streamed
    vllm_refusal = BadRequestError("Error code: 400 - \"auto\" tool choice requires --enable-auto-tool-choice and "
                                   "--tool-call-parser to be set")
    for refusal, responses_error in ((vllm_refusal, None), (TOOLS_UNSUPPORTED, BadRequestError("no responses here"))):
        asked = []

        def chat(**kw):
            if kw.get("tools"):
                raise refusal
            return streamed(content="prompt route")

        def responses(**kw):
            asked.append(kw)
            raise responses_error
        llm = LLM(client=client(chat, responses), model="m", retries=1)
        assert llm.chat(USER, tools=TOOLS).text == "prompt route" and llm.route == "chat"
        assert len(asked) == (0 if responses_error is None else 1)


def test_esc_while_the_responses_route_is_tried_stops_the_question(home):
    stopping = []

    def responses(**kw):
        stopping.append(1)
        return iter(events(text="never read"))
    llm = LLM(client=client(refuse_tools, responses), model="m", retries=1)
    with pytest.raises(Interrupted):
        llm.chat(USER, tools=TOOLS, stop=lambda: bool(stopping))


def test_tools_the_user_put_in_the_prompt_stay_there_on_the_responses_route(home, monkeypatch):
    monkeypatch.setenv("PERESEARCH_LLM_API", "responses")
    monkeypatch.setenv("PERESEARCH_LLM_TOOLS", "prompt")
    sent = []
    llm = LLM(client=client(lambda **kw: None, lambda **kw: sent.append(kw) or iter(events(text="ok"))),
              model="m", retries=1)
    llm.chat(USER, tools=TOOLS)
    assert "tools" not in sent[0] and "search_notes" in sent[0]["instructions"]   # described in the prompt


# Found by the second mutation run (2026-10-14).

def test_no_reply_budget_sends_no_token_field(home, monkeypatch):
    monkeypatch.setenv("PERESEARCH_LLM_API", "responses")
    sent = []
    llm = LLM(client=client(lambda **kw: None, lambda **kw: sent.append(kw) or iter(events(text="ok"))), model="m",
              retries=1)
    llm.reply_tokens = lambda: None                                   # nothing known: the server's own limit
    llm.chat(USER)
    assert "max_output_tokens" not in sent[0]


def test_a_client_that_names_no_endpoint_keeps_the_prompt(home):
    from tests.helpers import streamed

    def chat(**kw):
        if kw.get("tools"):
            raise TOOLS_UNSUPPORTED
        return streamed(content="prompt route")
    unnamed = NS(chat=NS(completions=NS(create=chat)), responses=NS(create=lambda **kw: pytest.fail("tried")))
    assert LLM(client=unnamed, model="m", retries=1).chat(USER, tools=TOOLS).text == "prompt route"


@pytest.mark.parametrize("name", ["NotFoundError", "BadRequestError", "UnprocessableEntityError",
                                  "PermissionDeniedError", "ResponsesFailed"])
def test_each_refusal_of_the_responses_route_falls_back_to_the_prompt(home, name):
    from tests.helpers import streamed

    def chat(**kw):
        if kw.get("tools"):
            raise TOOLS_UNSUPPORTED
        return streamed(content="prompt route")

    def responses(**kw):
        raise type(name, (Exception,), {})("no such route")
    llm = LLM(client=client(chat, responses), model="m", retries=1)
    assert llm.chat(USER, tools=TOOLS).text == "prompt route" and llm.route == "chat" and llm.tool_mode == "prompt"
    other = LLM(client=client(chat, lambda **kw: (_ for _ in ()).throw(type("APIConnectionError", (Exception,), {})(
        "unreachable"))), model="m", retries=1)
    with pytest.raises(Exception, match="unreachable"):               # not a refusal of the route: raised
        other.chat(USER, tools=TOOLS)
