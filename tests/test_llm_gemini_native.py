"""Gemini through Google's Interactions API (model gateway step 2, owner decision 2026-10-15). Offline: the stream's
events are built with google-genai's own types (2.31, TypeAdapter over InteractionSSEEvent) and the expected request
follows Google's stateless example (ai.google.dev/gemini-api/docs/interactions/function-calling, read 2026-10-15:
user_input, the model's steps exactly as received, function_result by call_id, store=false). No call to Google."""

import json
from types import SimpleNamespace as NS

import pytest
from pydantic import TypeAdapter

from peresearch.llm import LLM
from peresearch.messages import Message, Request, Tool
from peresearch.protocols import StreamCut, gemini

SEARCH = Tool("search_notes", "search the notes", {"type": "object", "properties": {"query": {"type": "string"}}})
QUESTION = Request("rules", (Message.user("what is bpe?"),), (SEARCH,))
_EVENT = None


def event(raw: dict):
    global _EVENT
    from google.genai._gaos.types import interactions as T
    _EVENT = _EVENT or TypeAdapter(T.InteractionSSEEvent)
    return _EVENT.validate_python(raw)


def completed(status, usage=(40, 9, 3)):
    return event({"event_type": "interaction.completed", "interaction": {"id": "i1", "status": status, "usage": {
        "total_input_tokens": usage[0], "total_output_tokens": usage[1], "total_thought_tokens": usage[2]}}})


def call_round():
    """A thought (its summary, then its signature) and a call whose arguments come in two pieces."""
    return [event(e) for e in (
        {"event_type": "interaction.created", "interaction": {"id": "i1", "status": "in_progress"}},
        {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
        {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_summary",
                                                             "content": {"type": "text", "text": "search first"}}},
        {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": "c2lnLTE="}},
        {"event_type": "step.stop", "index": 0},
        {"event_type": "step.start", "index": 1, "step": {"type": "function_call", "id": "fc1",
                                                           "name": "search_notes", "arguments": {}}},
        {"event_type": "step.delta", "index": 1, "delta": {"type": "arguments_delta", "arguments": '{"query": '}},
        {"event_type": "step.delta", "index": 1, "delta": {"type": "arguments_delta", "arguments": '"bpe"}'}},
        {"event_type": "step.stop", "index": 1})] + [completed("requires_action")]


def answer(text="BPE merges pairs [N1].", status="completed"):
    return [event(e) for e in (
        {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
        *({"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": text[i:i + 6]}}
          for i in range(0, len(text), 6)),
        {"event_type": "step.stop", "index": 0})] + [completed(status, (60, 7, 0))]


def google(sent, replies):
    def create(**kw):
        sent.append(json.loads(json.dumps(kw)))
        nxt = replies.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return iter(nxt)
    return NS(interactions=NS(create=create), models=NS(list=lambda: []))


def ask(sent, replies, **kw):
    return LLM(client=google(sent, replies), model="gemini-3.5-flash-lite", retries=kw.pop("retries", 1),
               provider="google", wait=0, **kw)


def continued(first):
    call = first.calls[0]
    return Request(QUESTION.system, QUESTION.messages + (first.message, Message.tool(call.id, call.name, "[N1] BPE")),
                   QUESTION.tools)


@pytest.fixture(autouse=True)
def plain(monkeypatch):
    for name in ("PERESEARCH_LLM_TOOLS", "PERESEARCH_LLM_CONTEXT", "PERESEARCH_LLM_API", "PERESEARCH_LLM_PROTOCOL"):
        monkeypatch.delenv(name, raising=False)


def test_a_tool_round_goes_through_the_interactions_api_statelessly(home):
    sent = []
    first = ask(sent, [call_round(), answer()]).chat(QUESTION)
    assert [(c.id, c.name, c.args) for c in first.calls] == [("fc1", "search_notes", {"query": "bpe"})]
    assert first.reasoning == "search first" and first.finish == "tool_calls"
    assert (first.prompt_tokens, first.completion_tokens) == (40, 12)          # output and thought tokens
    assert first.message.provider == {"gemini": {"steps": [
        {"type": "thought", "summary": [{"type": "text", "text": "search first"}], "signature": "c2lnLTE="},
        {"type": "function_call", "id": "fc1", "name": "search_notes", "arguments": {"query": "bpe"}}]}}
    body = sent[0]
    assert (body["model"], body["stream"], body["store"], body["system_instruction"]) == \
        ("gemini-3.5-flash-lite", True, False, "rules")
    assert body["input"] == [{"type": "user_input", "content": [{"type": "text", "text": "what is bpe?"}]}]
    assert body["tools"] == [{"type": "function", "name": "search_notes", "description": "search the notes",
                              "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}]
    assert body["generation_config"]["thinking_summaries"] == "auto" and body["generation_config"]["max_output_tokens"]


def test_the_next_request_resends_the_turn_exactly_and_the_result_by_call_id(home):
    sent = []
    llm = ask(sent, [call_round(), answer()])
    second = llm.chat(continued(llm.chat(QUESTION)))
    assert second.text == "BPE merges pairs [N1]." and second.finish == "stop" and second.calls == []
    assert sent[1]["input"] == [
        {"type": "user_input", "content": [{"type": "text", "text": "what is bpe?"}]},
        {"type": "thought", "summary": [{"type": "text", "text": "search first"}], "signature": "c2lnLTE="},
        {"type": "function_call", "id": "fc1", "name": "search_notes", "arguments": {"query": "bpe"}},
        {"type": "function_result", "call_id": "fc1", "name": "search_notes", "result": [{"type": "text", "text": "[N1] BPE"}]}]
    assert sent[1]["store"] is False


def test_the_turn_goes_back_through_a_new_gateway(home):
    sent = []
    first = ask(sent, [call_round()]).chat(QUESTION)
    ask(sent, [answer()]).chat(continued(first))
    assert sent[1]["input"][1]["signature"] == "c2lnLTE="                       # carried by the message, not the LLM


def test_a_turn_without_its_steps_is_rebuilt_as_text_and_calls(home):
    from peresearch.llm import Reply
    from peresearch.messages import ToolCall
    hand = Reply("looking", [ToolCall("fc9", "search_notes", {"query": "x"})])
    sent = []
    ask(sent, [answer()]).chat(Request("rules", (Message.user("q"), hand.message, Message.tool("fc9", "search_notes", "r")),
                                       (SEARCH,)))
    assert sent[0]["input"][1:] == [
        {"type": "model_output", "content": [{"type": "text", "text": "looking"}]},
        {"type": "function_call", "id": "fc9", "name": "search_notes", "arguments": {"query": "x"}},
        {"type": "function_result", "call_id": "fc9", "name": "search_notes", "result": [{"type": "text", "text": "r"}]}]


def test_the_answer_is_handed_over_as_it_streams(home):
    pieces = []
    llm = ask([], [answer()])
    llm.on_text = pieces.append
    assert llm.chat(QUESTION).text == "BPE merges pairs [N1]." and "".join(pieces) == "BPE merges pairs [N1]."


def test_a_reply_cut_at_its_budget_is_marked_and_a_stream_cut_short_is_retried(home):
    assert ask([], [answer(status="incomplete")]).chat(QUESTION).finish == "length"
    cut = [e for e in answer() if e.event_type != "interaction.completed"]
    sent = []
    assert ask(sent, [cut, answer()], retries=2).chat(QUESTION).text == "BPE merges pairs [N1]." and len(sent) == 2
    with pytest.raises(StreamCut):
        ask([], [cut]).chat(QUESTION)


def test_an_error_event_or_a_failed_interaction_is_raised_not_answered(home):
    error = event({"event_type": "error", "error": {"code": "invalid_argument", "message": "bad request"}})
    with pytest.raises(gemini.GeminiFailed, match="bad request"):
        ask([], [answer()[:1] + [error]]).chat(QUESTION)
    with pytest.raises(gemini.GeminiFailed, match="failed"):
        ask([], [answer(status="failed")]).chat(QUESTION)


def test_server_errors_and_rate_limits_are_retried_other_client_errors_are_not(home):
    from google.genai import errors
    server = errors.ServerError(503, {"error": {"code": 503, "message": "unavailable", "status": "UNAVAILABLE"}})
    limited = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    refused = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
    sent = []
    assert ask(sent, [server, limited, answer()], retries=3).chat(QUESTION).text and len(sent) == 3
    sent = []
    with pytest.raises(errors.ClientError):
        ask(sent, [refused, answer()], retries=3).chat(QUESTION)
    assert len(sent) == 1


def test_the_client_reaches_the_api_host_with_one_attempt():
    made = gemini.client("https://generativelanguage.googleapis.com/v1beta/openai/", "AIza-synthetic")
    options = made._api_client._http_options
    assert options.base_url == "https://generativelanguage.googleapis.com/" and options.retry_options.attempts == 1
    assert gemini.list_models(NS(models=NS(list=lambda: [NS(name="models/gemini-3.6-flash"), NS(name=None)]))) == \
        ["gemini-3.6-flash"]


# --- the request, built directly ---------------------------------------------------------------------------------

def test_the_request_body_follows_the_conversation_exactly():
    messages = [{"role": "system", "content": "one"}, {"role": "system", "content": "two"},
                {"role": "user", "content": "q"}, {"role": "assistant", "content": "plain answer"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "fc1", "type": "function", "function": {"name": "tree", "arguments": '{"a": 1}'}},
                    {"id": "fc2", "type": "function", "function": {"name": "read", "arguments": "not json"}},
                    {"id": "fc3", "type": "function", "function": {"name": "grep", "arguments": "[1]"}},
                    {"id": "fc4", "type": "function", "function": {"name": "glob"}},
                    {"id": "fc5", "type": "function", "function": {}},
                    {"type": "function", "function": {}}]},
                {"role": "tool", "tool_call_id": "fc1", "content": "result"},
                {"role": "tool", "tool_call_id": "fc5", "content": "nameless"},
                {"role": "tool", "content": "stray"}]
    tools = [{"type": "function", "function": {"name": "t1"}},
             {"type": "function", "function": {"name": "t2", "description": "d", "parameters": {"type": "object"}}},
             {"type": "other"}]
    out = gemini.body(messages, tools, {})
    assert out["system_instruction"] == "one\n\ntwo" and out["store"] is False
    assert out["input"] == [
        {"type": "user_input", "content": [{"type": "text", "text": "q"}]},
        {"type": "model_output", "content": [{"type": "text", "text": "plain answer"}]},
        {"type": "function_call", "id": "fc1", "name": "tree", "arguments": {"a": 1}},
        {"type": "function_call", "id": "fc2", "name": "read", "arguments": {}},          # unreadable: none
        {"type": "function_call", "id": "fc3", "name": "grep", "arguments": {}},          # not an object: none
        {"type": "function_call", "id": "fc4", "name": "glob", "arguments": {}},
        {"type": "function_call", "id": "fc5", "name": "", "arguments": {}},
        {"type": "function_call", "id": "", "name": "", "arguments": {}},
        {"type": "function_result", "call_id": "fc1", "name": "tree", "result": [{"type": "text", "text": "result"}]},
        {"type": "function_result", "call_id": "fc5", "name": "", "result": [{"type": "text", "text": "nameless"}]},
        {"type": "function_result", "call_id": "", "name": "", "result": [{"type": "text", "text": "stray"}]}]
    assert out["tools"] == [
        {"type": "function", "name": "t1", "description": "", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "t2", "description": "d", "parameters": {"type": "object"}}]
    assert "system_instruction" not in gemini.body([{"role": "user", "content": "q"}], None, {}) and \
        "tools" not in gemini.body([{"role": "user", "content": "q"}], None, {})


def test_an_empty_message_content_is_sent_as_empty_text_not_a_placeholder():
    out = gemini.body([{"role": "user", "content": None}, {"role": "assistant", "content": None},
                       {"role": "tool", "tool_call_id": "x", "content": None}], None, {})
    assert out["input"] == [{"type": "user_input", "content": [{"type": "text", "text": ""}]},
                            {"type": "function_result", "call_id": "x", "name": "",
                             "result": [{"type": "text", "text": ""}]}]


def test_kept_steps_are_sent_as_copies_in_place_of_the_rebuilt_calls():
    kept = [{"type": "thought", "signature": "s"}, {"type": "function_call", "id": "fc1", "name": "tree",
                                                    "arguments": {}}]
    call = {"id": "fc1", "type": "function", "function": {"name": "tree", "arguments": "{}"}}
    out = gemini.body([{"role": "assistant", "content": "ignored", "tool_calls": [call]}], None, {"fc1": kept})
    assert out["input"] == kept and out["input"][0] is not kept[0]


def test_sampling_and_the_reply_budget_reach_the_generation_config(home):
    sent = []
    llm = ask(sent, [answer()], max_tokens=777, temperature=0.25, top_p=0.5)
    llm.chat(QUESTION)
    assert sent[0]["generation_config"] == {"thinking_summaries": "auto", "max_output_tokens": 777,
                                            "temperature": 0.25, "top_p": 0.5}
    config = gemini.send(NS(interactions=NS(create=lambda **kw: kw)), {"model": "m", "messages": [], "tools": None,
                                                                        "max_completion_tokens": 55}, {})
    assert config["generation_config"]["max_output_tokens"] == 55 and "temperature" not in config["generation_config"]


# --- the stream, read directly -----------------------------------------------------------------------------------

def read(events, **kw):
    return gemini.read(iter(events), kw.pop("stop", None), kw.pop("on_first", None), kw.pop("on_text", None))


def step(index, kind, **fields):
    return event({"event_type": "step.start", "index": index, "step": {"type": kind, **fields}})


def delta(index, kind, **fields):
    return event({"event_type": "step.delta", "index": index, "delta": {"type": kind, **fields}})


def stop_step(index):
    return event({"event_type": "step.stop", "index": index})


def test_a_call_complete_at_its_start_needs_no_argument_deltas():
    got = read([step(0, "function_call", id="fc1", name="tree", arguments={"path": "a"}), stop_step(0),
                completed("requires_action")])
    assert got["calls"] == [{"id": "fc1", "name": "tree", "args": '{"path": "a"}'}] and got["finish"] == "tool_calls"
    assert got["steps"] == [{"type": "function_call", "id": "fc1", "name": "tree", "arguments": {"path": "a"}}]


def test_a_call_without_arguments_has_none_and_a_call_may_come_without_its_start():
    got = read([step(0, "function_call", id="fc1", name="tree", arguments={}), stop_step(0),
                delta(1, "arguments_delta", arguments='{"q": 1}'), stop_step(1), completed("requires_action")])
    assert got["calls"][0] == {"id": "fc1", "name": "tree", "args": ""}
    assert got["calls"][1] == {"id": "call-1", "name": "", "args": '{"q": 1}'}      # no id, no name: the gateway's own
    assert got["steps"][0]["arguments"] == {}


def test_arguments_that_are_not_json_are_kept_for_the_gateway_to_report():
    got = read([step(0, "function_call", id="fc1", name="tree", arguments={}),
                delta(0, "arguments_delta", arguments='{"q": '), stop_step(0), completed("requires_action")])
    assert got["calls"][0]["args"] == '{"q": ' and got["steps"][0]["arguments"] == {}


def test_a_turn_with_text_and_a_call_keeps_both_steps_in_order():
    got = read([step(0, "model_output"), delta(0, "text", text="Look"), delta(0, "text", text="ing."), stop_step(0),
                step(1, "function_call", id="fc1", name="tree", arguments={}), delta(1, "arguments_delta", arguments="{}"),
                stop_step(1), completed("requires_action")])
    assert got["text"] == "Looking." and got["steps"] == [
        {"type": "model_output", "content": [{"type": "text", "text": "Looking."}]},
        {"type": "function_call", "id": "fc1", "name": "tree", "arguments": {}}]


def test_a_text_delta_for_a_step_never_started_is_still_the_answer():
    got = read([delta(0, "text", text="hi"), completed("completed")])
    assert got["text"] == "hi" and got["steps"] == []                                # nothing to send back with it


def test_summaries_signatures_and_other_steps_are_told_apart():
    got = read([step(0, "thought"), delta(0, "thought_summary", content={"type": "text", "text": "a"}),
                delta(0, "thought_summary"), delta(0, "thought_summary", content={"type": "text", "text": "b"}),
                delta(0, "thought_signature", signature="sig"), stop_step(0),
                step(1, "google_search_call", id="x", arguments={}), stop_step(1), completed("completed")])
    assert got["reasoning"] == "ab"
    assert got["steps"] == [{"type": "thought", "summary": [{"type": "text", "text": "a"}, {"type": "text", "text": ""},
                                                            {"type": "text", "text": "b"}], "signature": "sig"}]


def test_usage_is_counted_when_given_and_zero_when_not():
    assert (lambda g: (g["prompt_tokens"], g["completion_tokens"]))(read([completed("completed", (10, 4, 2))])) == (10, 6)
    bare = event({"event_type": "interaction.completed", "interaction": {"id": "i", "status": "completed"}})
    assert (lambda g: (g["prompt_tokens"], g["completion_tokens"]))(read([bare])) == (0, 0)
    part = event({"event_type": "interaction.completed", "interaction": {"id": "i", "status": "completed",
                                                                         "usage": {"total_input_tokens": 5}}})
    assert (lambda g: (g["prompt_tokens"], g["completion_tokens"]))(read([part])) == (5, 0)


def test_how_an_interaction_ended_decides_the_finish():
    assert read([completed("completed")])["finish"] == "stop"
    assert read([completed("incomplete")])["finish"] == "length"
    for status in ("failed", "cancelled"):
        with pytest.raises(gemini.GeminiFailed, match=status):
            read([completed(status)])


def test_an_error_event_says_its_message_or_a_default():
    with pytest.raises(gemini.GeminiFailed, match="^upstream timed out$"):
        read([event({"event_type": "error", "error": {"code": "gateway_timeout", "message": "upstream timed out"}})])
    with pytest.raises(gemini.GeminiFailed, match="^the Interactions API reported an error$"):
        read([event({"event_type": "error"})])


def test_a_stream_that_fails_or_is_cut_carries_what_came_before_it_and_is_closed():
    closed = []

    class Source:
        def __iter__(self):
            yield from answer()[:2]                                   # a step start, one piece of text
            raise StreamCut("cut")

        def close(self):
            closed.append(True)
    with pytest.raises(StreamCut) as raised:
        gemini.read(Source(), None)
    assert raised.value.partial["text"] == "BPE me" and closed == [True]
    with pytest.raises(StreamCut, match="^the reply stream ended without a finish reason: the connection was cut "
                                        "mid-answer$") as raised:
        read(answer()[:2] + [event({"event_type": "step.stop", "index": 0})])
    assert raised.value.partial["text"] == "BPE me" and raised.value.partial["finish"] == ""
    with pytest.raises(gemini.GeminiFailed) as raised:
        read([completed("completed", (7, 1, 0)), event({"event_type": "error"})])
    assert (raised.value.partial["prompt_tokens"], raised.value.partial["finish"]) == (7, "stop")


def test_the_reader_polls_stop_and_announces_the_first_event_once():
    from peresearch.protocols import Interrupted
    firsts = []
    read(answer(), on_first=lambda: firsts.append(1))
    assert firsts == [1]
    with pytest.raises(Interrupted, match="^stopped by the user$"):
        read(answer(), stop=lambda: True)


def test_a_step_is_dumped_by_alias_without_empty_fields_and_a_plain_dict_as_it_is():
    from google.genai._gaos.types import interactions as T
    thought = TypeAdapter(T.InteractionSSEEvent).validate_python(
        {"event_type": "step.start", "index": 0, "step": {"type": "thought", "signature": "abc"}}).step
    assert gemini._dump(thought) == {"type": "thought", "signature": "abc"}
    assert gemini._dump({"type": "x"}) == {"type": "x"}


def test_the_client_has_the_hosts_base_a_ten_minute_timeout_and_the_key():
    made = gemini.client("https://generativelanguage.googleapis.com/v1beta/openai/", "AIza-synthetic")
    options = made._api_client._http_options
    assert (options.timeout, options.retry_options.attempts) == (600_000, 1)
    assert made._api_client.api_key == "AIza-synthetic"
    assert made._api_client._http_options.base_url == "https://generativelanguage.googleapis.com/"
    with pytest.raises(ValueError, match="No API key was provided"):          # the SDK's own refusal, not a request
        gemini.client("https://generativelanguage.googleapis.com/", "")
    from google import genai
    default = genai.Client(api_key="k")._api_client._http_options.base_url
    assert gemini.client("not a url", "k")._api_client._http_options.base_url == default      # no host: the SDK's own
    proxy = gemini.client("https://proxy.example/v1beta/openai/", "k")._api_client._http_options
    assert proxy.base_url == "https://proxy.example/"                          # the host the user gave, not Google's
    assert gemini.client("http://localhost:8080/x", "k")._api_client._http_options.base_url == "http://localhost:8080/"


def test_listing_models_drops_the_prefix_and_the_unnamed():
    listed = NS(models=NS(list=lambda: [NS(name="models/b"), NS(name="models/a"), NS(name=None), NS(name="")]))
    assert gemini.list_models(listed) == ["a", "b"]


def test_arguments_that_arrive_unreadable_replace_those_the_step_began_with():
    got = read([step(0, "function_call", id="fc1", name="tree", arguments={"old": 1}),
                delta(0, "arguments_delta", arguments='{"q": '), stop_step(0), completed("requires_action")])
    assert got["steps"][0]["arguments"] == {} and got["calls"][0]["args"] == '{"q": '


def test_an_arguments_delta_without_arguments_adds_nothing_and_a_stop_without_a_start_is_ignored():
    got = read([step(0, "function_call", id="fc1", name="tree", arguments={}), delta(0, "arguments_delta"),
                delta(0, "arguments_delta", arguments="{}"), stop_step(0), stop_step(7), completed("requires_action")])
    assert got["calls"][0]["args"] == "{}" and got["steps"] == [
        {"type": "function_call", "id": "fc1", "name": "tree", "arguments": {}}]


def test_input_tokens_the_interaction_does_not_report_are_zero():
    only_output = event({"event_type": "interaction.completed", "interaction": {
        "id": "i", "status": "completed", "usage": {"total_output_tokens": 3}}})
    got = read([only_output])
    assert (got["prompt_tokens"], got["completion_tokens"]) == (0, 3)
