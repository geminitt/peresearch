"""The Anthropic Messages route (2026-10-14): Claude through its native API and the official SDK (anthropic 1.13),
chosen by the provider registry, behind the same LLM.chat contract. Offline: a client shaped like the SDK's
(messages.stream → events + get_final_message, models.list); no call to Anthropic is made.

Claude's thinking blocks carry a signature bound to the conversation before them: they go back unchanged with the
tool calls of their turn, and a 400 naming an invalid thinking signature is answered once by sending the history
without thinking blocks (the documented recovery that needs no beta header)."""

import json
from types import SimpleNamespace as NS

import pytest

from peresearch import profiles, providers
from peresearch.llm import LLM
from peresearch.messages import Message, Request
from tests.helpers import request

TOOLS = [{"type": "function", "function": {"name": "search_notes", "description": "search the notes",
                                           "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}]
QUESTION = [{"role": "system", "content": "rules"}, {"role": "user", "content": "what is bpe?"}]


class Block:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def model_dump(self, exclude_none=False, **_):
        return {k: v for k, v in self.__dict__.items() if not (exclude_none and v is None)}


class Stream:
    def __init__(self, events, final):
        self.events, self.final = events, final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.final


def reply(blocks, stop="end_turn", usage=(100, 20, 0, 0), details=None):
    events = [NS(type="thinking", thinking=b.thinking) for b in blocks if b.type == "thinking"]
    events += [NS(type="text", text=b.text) for b in blocks if b.type == "text"]
    return Stream(events, NS(content=blocks, stop_reason=stop, stop_details=details,
                             usage=NS(input_tokens=usage[0], output_tokens=usage[1], cache_read_input_tokens=usage[2],
                                      cache_creation_input_tokens=usage[3])))


THINK = Block(type="thinking", thinking="search first", signature="sig-1")
CALL = Block(type="tool_use", id="toolu_1", name="search_notes", input={"query": "bpe"})


def claude(sent, replies, models=()):
    def stream(**kw):
        sent.append(json.loads(json.dumps(kw)))
        nxt = replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt
    return NS(messages=NS(stream=stream), models=NS(list=lambda: NS(data=list(models))),
              base_url="https://api.anthropic.com")


def ask_with_tool(llm):
    """A question, then the conversation continued as the agent continues it: the model's turn as its message (what
    the provider attached travels with it), then the tool's result."""
    question = request(QUESTION, TOOLS)
    first = llm.chat(question)
    call = first.calls[0]
    followed = Request(question.system, question.messages + (
        first.message, Message.tool(call.id, call.name, "[N1] bpe.md: BPE merges pairs")), question.tools)
    return first, llm.chat(followed)


@pytest.fixture(autouse=True)
def plain(monkeypatch):
    for name in ("PERESEARCH_LLM_TOOLS", "PERESEARCH_LLM_CONTEXT", "PERESEARCH_LLM_API"):
        monkeypatch.delenv(name, raising=False)


def test_the_registry_names_each_provider_and_its_protocol():
    anthropic = providers.get("anthropic")
    assert (anthropic.url, anthropic.protocol, anthropic.roles) == ("https://api.anthropic.com", "anthropic", ("chat",))
    google = providers.get("google")
    assert google.url == "https://generativelanguage.googleapis.com/v1beta/openai/" and google.protocol == "gemini"
    assert providers.get("openai").protocol == "chat" and profiles.profile("gpt-6.1-sol").tools == "responses"
    assert providers.guess("https://api.anthropic.com/v1") == "anthropic" and providers.guess("http://localhost:8000/v1") == ""


def test_a_question_with_a_tool_round_goes_through_the_messages_api(home):
    sent = []
    replies = [reply([THINK, Block(type="text", text="Looking."), CALL], stop="tool_use", usage=(100, 20, 30, 5)),
               reply([Block(type="text", text="BPE merges pairs [N1].")])]
    llm = LLM(client=claude(sent, replies), model="claude-opus-5-5", retries=1, provider="anthropic")
    first, second = ask_with_tool(llm)
    assert [(c.id, c.name, c.args) for c in first.calls] == [("toolu_1", "search_notes", {"query": "bpe"})]
    assert (first.text, first.reasoning, first.finish) == ("Looking.", "search first", "tool_calls")
    assert (first.prompt_tokens, first.completion_tokens) == (135, 20)          # cached input counted
    body = sent[0]
    assert body["model"] == "claude-opus-5-5" and body["system"] == "rules" and body["max_tokens"] > 0
    assert body["messages"] == [{"role": "user", "content": "what is bpe?"}]
    assert body["tools"] == [{"name": "search_notes", "description": "search the notes",
                                 "input_schema": TOOLS[0]["function"]["parameters"]}]
    assert body["cache_control"] == {"type": "ephemeral"}
    assert not {"temperature", "top_p"} & set(body)                           # current models refuse them
    assert second.text == "BPE merges pairs [N1]." and second.finish == "stop"
    assert sent[1]["messages"][1:] == [
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "search first", "signature": "sig-1"},
                                          {"type": "text", "text": "Looking."},
                                          {"type": "tool_use", "id": "toolu_1", "name": "search_notes",
                                           "input": {"query": "bpe"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                      "content": "[N1] bpe.md: BPE merges pairs"}]}]


def test_an_invalid_thinking_signature_is_answered_once_without_thinking_blocks(home):
    class BadRequestError(Exception):
        pass
    invalid = BadRequestError("Error code: 400 - messages.1.content.0: Invalid `signature` in `thinking` block. The "
                              "block is bound to a different conversation.")
    sent = []
    replies = [reply([THINK, CALL], stop="tool_use"), invalid, reply([Block(type="text", text="ok")])]
    llm = LLM(client=claude(sent, replies), model="claude-opus-5-5", retries=1, provider="anthropic")
    first, second = ask_with_tool(llm)
    assert second.text == "ok" and len(sent) == 3
    replayed = sent[2]["messages"][1]["content"]
    assert [b["type"] for b in replayed] == ["tool_use"]                          # text and calls stay, thinking goes


def test_a_refusal_a_cut_and_the_context_are_reported(home):
    sent = []
    replies = [reply([], stop="refusal", details=NS(category="cyber", explanation="declined for safety")),
               reply([Block(type="text", text="half")], stop="max_tokens")]
    models = [NS(id="claude-opus-5-5", max_input_tokens=1_000_000, max_tokens=128_000)]
    llm = LLM(client=claude(sent, replies, models), model="claude-opus-5-5", retries=1, provider="anthropic")
    refused = llm.chat(request(QUESTION))
    assert refused.refusal == "declined for safety" and refused.calls == []
    cut = llm.chat(request(QUESTION))
    assert cut.text == "half" and cut.finish == "length"
    assert llm.context_length() == 1_000_000


def test_consecutive_tool_results_go_back_in_one_user_message(home):
    from peresearch.protocols.anthropic import anthropic_request
    messages = QUESTION + [{"role": "assistant", "content": "", "tool_calls": [
                               {"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                               {"id": "b", "type": "function", "function": {"name": "y", "arguments": '{"q": 1}'}}]},
                           {"role": "tool", "tool_call_id": "a", "content": "A"},
                           {"role": "tool", "tool_call_id": "b", "content": "B"},
                           {"role": "user", "content": "Limit reached: answer now."}]
    body = anthropic_request(messages, None, {})
    assert body["messages"][1] == {"role": "assistant", "content": [
        {"type": "tool_use", "id": "a", "name": "x", "input": {}},
        {"type": "tool_use", "id": "b", "name": "y", "input": {"q": 1}}]}
    assert body["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "a", "content": "A"},
        {"type": "tool_result", "tool_use_id": "b", "content": "B"}]}
    assert body["messages"][3] == {"role": "user", "content": "Limit reached: answer now."}


# Paths the mutation run of the registry and the Anthropic route found untested (2026-10-14).

from peresearch import connections, settings
from peresearch.llm import Interrupted
from peresearch.protocols.anthropic import _read_anthropic, anthropic_request


def test_the_registry_table_is_exact():
    plain, local = providers.Compatibility(), providers.Compatibility(ollama_probe=True)
    assert [(p.id, p.name, p.url, p.protocol, p.kind, p.roles, p.compatibility) for p in providers.PROVIDERS] == [
        ("openai", "OpenAI", "https://api.openai.com/v1", "chat", "cloud", ("chat", "embedding"), plain),
        ("anthropic", "Anthropic (Claude)", "https://api.anthropic.com", "anthropic", "cloud", ("chat",), plain),
        ("google", "Google AI Studio (Gemini)", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini",
         "cloud", ("chat", "embedding"), plain),
        ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "chat", "cloud", ("chat",), plain),
        ("groq", "Groq", "https://api.groq.com/openai/v1", "chat", "cloud", ("chat",), plain),
        ("ollama", "Ollama", "http://localhost:11434/v1", "chat", "local", ("chat", "embedding"),
         providers.Compatibility(thinking_switch="reasoning_effort", ollama_probe=True)),
        ("lmstudio", "LM Studio", "http://localhost:1234/v1", "chat", "local", ("chat", "embedding"), local),
        ("local", "vLLM, llama.cpp or another server", "http://localhost:8000/v1", "chat", "local",
         ("chat", "embedding", "rerank"), local)]
    assert providers.get("") is None and providers.get(None) is None and providers.get("nope") is None
    assert providers.guess("") == providers.guess(None) == "" and providers.guess("not a url") == ""
    assert providers.protocol("nope") == "chat" and providers.protocol("anthropic") == "anthropic"
    assert providers.protocol("", "gemini") == "gemini" and providers.protocol("", "bogus") == "chat"
    assert providers.guess("https://openrouter.ai/api/v1") == "openrouter"


def test_a_request_fills_what_a_history_lacks():
    messages = [{"role": "system", "content": "one"}, {"role": "system", "content": "two"},
                {"role": "user", "content": "q"}, {"role": "assistant", "content": "plain answer"},
                {"role": "assistant", "content": ""},                             # empty, no call: not sent
                {"role": "assistant", "content": "calling", "tool_calls": [
                    {"type": "function", "function": {"arguments": "not json"}},
                    {"id": "c2", "type": "function", "function": {"name": "n", "arguments": "[1, 2]"}}]},
                {"role": "tool", "content": "r"}]
    tools = [{"type": "function", "function": {"name": "bare"}}, {"type": "web"}]
    assert anthropic_request(messages, tools, {}) == {
        "system": "one\n\ntwo",
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "plain answer"},
                     {"role": "assistant", "content": [
                         {"type": "text", "text": "calling"},
                         {"type": "tool_use", "id": "", "name": "", "input": {}},
                         {"type": "tool_use", "id": "c2", "name": "n", "input": {}}]},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "", "content": "r"}]}],
        "tools": [{"name": "bare", "description": "", "input_schema": {"type": "object", "properties": {}}}]}


def final(content=(), stop="end_turn", usage=None, details=None):
    return NS(content=list(content), stop_reason=stop, stop_details=details,
              usage=usage or NS(input_tokens=3, output_tokens=4, cache_read_input_tokens=None,
                                cache_creation_input_tokens=None))


@pytest.mark.parametrize("stop, finish", [("end_turn", "stop"), ("tool_use", "tool_calls"), ("max_tokens", "length"),
                                          ("stop_sequence", "stop"), ("pause_turn", "stop"), ("new_reason", "new_reason"),
                                          (None, "stop")])
def test_every_stop_reason_has_its_finish(stop, finish):
    assert _read_anthropic(Stream([], final(stop=stop)), None)["finish"] == finish


def test_blocks_calls_and_counts_come_back_as_sent():
    call = Block(type="tool_use", id="t1", name="read", input={"path": "Phở.md"}, caller=None)
    got = _read_anthropic(Stream([], final([call], stop="tool_use",
                                           usage=NS(input_tokens=None, output_tokens=None, cache_read_input_tokens=7,
                                                    cache_creation_input_tokens=2))), None)
    assert got["calls"] == [{"id": "t1", "name": "read", "args": '{"path": "Phở.md"}'}]    # not \\u escaped
    assert got["blocks"] == [{"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "Phở.md"}}]
    assert (got["prompt_tokens"], got["completion_tokens"]) == (9, 0)
    plain = _read_anthropic(Stream([], final()), None)
    assert (plain["prompt_tokens"], plain["completion_tokens"]) == (3, 4)


def test_a_refusal_without_an_explanation_and_a_refusal_after_a_call():
    got = _read_anthropic(Stream([], final([CALL], stop="refusal", details=None)), None)
    assert got["refusal"] == "the model declined to answer" and got["calls"] == []


def test_a_stream_cut_or_stopped_keeps_what_came_and_says_why():
    class Cut(Stream):
        def __iter__(self):
            yield NS(type="text", text="half ")
            yield NS(type="thinking", thinking="mid")
            raise ConnectionError("reset")
    with pytest.raises(ConnectionError) as caught:
        _read_anthropic(Cut([], None), None)
    assert caught.value.partial == {"text": "half ", "reasoning": "mid", "calls": [], "finish": "",
                                    "prompt_tokens": 0, "completion_tokens": 0}
    first = []
    with pytest.raises(Interrupted, match="^stopped by the user$"):
        _read_anthropic(Stream([NS(type="text", text="x")], final()), lambda: True, lambda: first.append(1))
    assert first == [1]


def test_the_reply_budget_reaches_max_tokens(home):
    sent = []
    llm = LLM(client=claude(sent, [reply([Block(type="text", text="a")])] * 3), model="m", retries=1,
              provider="anthropic")
    llm.reply_tokens = lambda: None
    llm.chat(request(QUESTION))
    assert sent[0]["max_tokens"] == 32_000                             # required: the default when nothing is known
    for field in ("max_tokens", "max_completion_tokens"):
        llm.reply_tokens, llm._token_field = (lambda: 777), field
        llm.chat(request(QUESTION))
        assert sent[-1]["max_tokens"] == 777


def test_the_thinking_recovery_is_once_and_only_on_the_anthropic_route(home):
    class BadRequestError(Exception):
        pass
    invalid = BadRequestError("Invalid `signature` in `thinking` block")
    redacted = Block(type="redacted_thinking", data="opaque")
    sent = []
    llm = LLM(client=claude(sent, [reply([THINK, redacted, CALL], stop="tool_use"), invalid, invalid]), model="m",
              retries=1, provider="anthropic")
    with pytest.raises(BadRequestError):                               # stripped once; a second refusal is raised
        ask_with_tool(llm)
    assert [b["type"] for b in sent[2]["messages"][1]["content"]] == ["tool_use"] and len(sent) == 3
    chat_sent = []

    def chat(**kw):
        chat_sent.append(kw)
        raise invalid
    other = LLM(client=NS(chat=NS(completions=NS(create=chat))), model="m", retries=1)
    with pytest.raises(BadRequestError):
        other.chat(request(QUESTION))
    assert len(chat_sent) == 1                                         # no recovery on another route


def test_the_provider_and_its_client_come_from_the_settings(home, monkeypatch):
    settings.save({"PERESEARCH_LLM_URL": "https://api.anthropic.com/v1/", "PERESEARCH_LLM_KEY": "sk-ant-synthetic",
                   "PERESEARCH_LLM_MODEL": "claude-opus-5-5"})
    guessed = LLM()                                                    # no provider saved: guessed from the host
    assert guessed.provider == "anthropic" and guessed.route == "anthropic"
    client = guessed.client
    assert str(client.base_url).rstrip("/") == "https://api.anthropic.com" and client.api_key == "sk-ant-synthetic"
    assert client.max_retries == 0 and client.timeout == 600.0
    settings.save({"PERESEARCH_LLM_URL": "https://proxy.example/v1", "PERESEARCH_LLM_PROVIDER": "anthropic",
                   "PERESEARCH_LLM_MODEL": "m"})
    assert LLM().route == "anthropic"                                  # a saved provider wins over the host
    settings.save({"PERESEARCH_LLM_URL": "http://localhost:8000/v1", "PERESEARCH_LLM_MODEL": "m"})
    assert LLM().route == "chat" and LLM().provider == ""


def test_the_context_comes_from_the_models_api_for_an_unknown_model(home):
    models = [NS(id="claude-unlisted-test", max_input_tokens=123_456)]
    llm = LLM(client=claude([], [], models), model="claude-unlisted-test", retries=1, provider="anthropic")
    assert llm.context_length() == 123_456


def test_connections_record_and_list_through_the_provider(home, monkeypatch):
    assert connections.Connection("i", "chat", "n", "http://x/v1").provider == ""
    plain = connections.add("chat", "", "https://api.groq.com/openai/v1", "k")
    assert plain.provider == "groq" and plain.name == "api.groq.com"
    named = connections.connect("chat", "Claude", "https://api.anthropic.com", "k2", provider="anthropic")
    assert named.provider == "anthropic"
    assert connections.selection(named, "claude-opus-5-5")["PERESEARCH_LLM_PROVIDER"] == "anthropic"
    unnamed = connections.Connection("j", "chat", "n", "https://generativelanguage.googleapis.com/v1beta/openai/")
    assert connections.selection(unnamed, "g")["PERESEARCH_LLM_PROVIDER"] == "google"
    embed = connections.Connection("e", "embedding", "n", "http://localhost:11434/v1")
    assert "PERESEARCH_LLM_PROVIDER" not in connections.selection(embed, "bge-m3")
    made = []
    import anthropic as sdk

    closed = []

    def fake(**kw):
        made.append(kw)
        return NS(models=NS(list=lambda: NS(data=[NS(id="claude-b"), NS(id="claude-a")])),
                  close=lambda: closed.append(1))
    monkeypatch.setattr(sdk, "Anthropic", fake)
    assert connections.discover(named) == ["claude-a", "claude-b"]
    assert made == [{"base_url": "https://api.anthropic.com", "api_key": "k2", "timeout": 15, "max_retries": 0}]
    assert closed == [1]                                               # the client it made, closed
    v1 = connections.Connection("v", "chat", "n", "https://api.anthropic.com/v1/", "", provider="anthropic")
    connections.discover(v1)
    assert made[-1]["base_url"] == "https://api.anthropic.com" and made[-1]["api_key"] is None


def test_a_selection_saved_before_providers_is_adopted_with_its_provider(home):
    settings.save({"PERESEARCH_LLM_URL": "https://api.anthropic.com", "PERESEARCH_LLM_MODEL": "claude-opus-5-5",
                   "PERESEARCH_LLM_PROVIDER": "anthropic", "PERESEARCH_EMBED_URL": "http://localhost:11434/v1"})
    connections.adopt_saved_selection()
    got = {e.role: (e.provider, e.name, e.models) for e in connections.load()}
    assert got == {"chat": ("anthropic", "api.anthropic.com", ["claude-opus-5-5"]),
                   "embedding": ("", "localhost", [])}


# Found by the second mutation run (2026-10-14).

def test_the_first_answer_is_announced_once_whatever_follows(home):
    first = []
    stream = Stream([NS(type="thinking", thinking="a"), NS(type="text", text="b"), NS(type="text", text="c")], final())
    assert _read_anthropic(stream, lambda: False, lambda: first.append(1))["text"] == "bc" and first == [1]


def test_a_claude_endpoint_behind_a_proxy_keeps_its_address_and_provider(home, monkeypatch):
    settings.save({"PERESEARCH_LLM_URL": "https://proxy.example/anthropic/v1/", "PERESEARCH_LLM_PROVIDER": "anthropic",
                   "PERESEARCH_LLM_MODEL": "m", "PERESEARCH_LLM_KEY": "k"})
    assert str(LLM().client.base_url).rstrip("/") == "https://proxy.example/anthropic"
    entry = connections.connect("chat", "Proxy", "https://proxy.example/anthropic/v1", "k", provider="anthropic")
    assert entry.provider == "anthropic"
    made = []
    import anthropic as sdk
    monkeypatch.setattr(sdk, "Anthropic", lambda **kw: made.append(kw) or NS(
        models=NS(list=lambda: NS(data=[])), close=lambda: None))
    connections.discover(entry)
    unnamed = connections.Connection("u", "chat", "n", "https://api.anthropic.com")       # no provider recorded
    connections.discover(unnamed)
    assert [m["base_url"] for m in made] == ["https://proxy.example/anthropic", "https://api.anthropic.com"]


def test_connecting_without_a_provider_guesses_it_and_keeps_a_model_id(home):
    assert connections.connect("chat", "", "https://api.groq.com/openai/v1", "k").provider == "groq"
    assert connections.add("chat", "", "http://localhost:8000/v1", "", "served-id").models == ["served-id"]


def test_adopted_selections_keep_the_saved_provider_or_guess_it(home):
    settings.save({"PERESEARCH_LLM_URL": "https://proxy.example/v1", "PERESEARCH_LLM_MODEL": "m",
                   "PERESEARCH_LLM_PROVIDER": "anthropic", "PERESEARCH_EMBED_URL": "https://api.openai.com/v1",
                   "PERESEARCH_EMBED_MODEL": "text-embedding-4"})
    connections.adopt_saved_selection()
    assert {e.role: e.provider for e in connections.load()} == {"chat": "anthropic", "embedding": "openai"}


def test_a_refused_token_field_is_switched_once_and_requests_are_bounded(home):
    """Two counters of the chat route that the Anthropic change touched: the token field is switched once when the
    server refuses it, and one question sends at most retries + 4 requests (each capability fallback once)."""
    from tests.helpers import streamed

    class BadRequestError(Exception):
        def __init__(self, message, param):
            super().__init__(message)
            self.body = {"error": {"message": message, "param": param, "code": "unsupported_parameter"}}
    sent = []

    def create(**kw):
        sent.append(sorted(k for k in kw if k in ("max_tokens", "max_completion_tokens", "extra_body",
                                                  "stream_options")))
        for field in ("max_tokens", "max_completion_tokens"):
            if field in kw and len(sent) == 1:
                raise BadRequestError(f"Unsupported parameter: '{field}' is not supported with this model.", field)
        return streamed(content="ok")
    llm = LLM(client=NS(chat=NS(completions=NS(create=create))), model="m", retries=1, max_tokens=50)
    assert llm.chat(request(QUESTION)).text == "ok" and len(sent) == 2
    assert {"max_tokens", "max_completion_tokens"} == {sent[0][0], sent[1][0]}

    refused = []

    def refuse_all(**kw):                      # each capability refused once, in turn; the fifth request is served
        refused.append(1)
        body = kw.get("extra_body") or {}
        order = [("max_tokens" in kw or "max_completion_tokens" in kw) and len(refused) == 1 and
                 ("max_tokens" if "max_tokens" in kw else "max_completion_tokens"),
                 "chat_template_kwargs" in body and "chat_template_kwargs",
                 "reasoning_effort" in body and "reasoning_effort",
                 "stream_options" in kw and "stream_options"]
        field = next((f for f in order if f), None)
        if field is None:
            return streamed(content="done")
        raise BadRequestError(f"Unsupported parameter: '{field}' is not supported with this model.", field)
    bounded = LLM(client=NS(chat=NS(completions=NS(create=refuse_all))), model="unknown-model", retries=1,
                  max_tokens=50, thinking=False)
    assert bounded.chat(request(QUESTION)).text == "done" and len(refused) == 5      # retries (1) + four fallbacks


def test_a_selection_saved_without_a_provider_is_adopted_with_the_guessed_one(home):
    settings.save({"PERESEARCH_LLM_URL": "https://api.groq.com/openai/v1", "PERESEARCH_LLM_MODEL": "llama"})
    connections.adopt_saved_selection()
    assert [e.provider for e in connections.load()] == ["groq"]
