"""Token capabilities, refusal identity and complete MCP initialization on retry."""
import json
import sys
from types import SimpleNamespace as NS

import httpx
from openai import BadRequestError, OpenAI
import pytest

from peresearch.llm import LLM, Reply, StreamCut, _read_stream, unsupported
from peresearch.mcp import MCPError, PROTOCOL, Server
from tests.helpers import streamed
from tests.test_llm import Stream, chunk, client_of

USER = [{"role": "user", "content": "A synthetic protocol question."}]
TOOLS = [{"type": "function", "function": {"name": "tree", "parameters": {"type": "object"}}}]


@pytest.mark.parametrize("family", ["o", "o-mini", "o-pro", "gpt", "gpt-mini", "gpt-nano", "gpt-pro",
    "gpt-codex", "gpt-codex-spark", "gpt-astra", "gpt-luna", "gpt-sol", "gpt-terra"])
def test_known_chat_api_families_use_the_completion_budget(family):
    from peresearch.profiles import Profile
    assert Profile(family=family).token_field == "max_completion_tokens"


@pytest.mark.parametrize("family", ["", "qwen", "llama", "gemma", "gpt-oss", "gpt-unknown"])
def test_local_and_unknown_families_keep_the_legacy_budget(family):
    from peresearch.profiles import Profile
    assert Profile(family=family).token_field == "max_tokens"


def rejected(field, message, code="unsupported_parameter"):
    response = httpx.Response(400, request=httpx.Request("POST", "https://fixture.test/v1/chat/completions"))
    return BadRequestError(message, response=response, body={"param": field, "code": code, "message": message})


def test_plain_reply_preserves_the_public_empty_refusal_default():
    assert Reply("An ordinary answer.").refusal == ""


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("code", ["unsupported_parameter", "unknown_parameter", "unrecognized_parameter", "extra_forbidden"])
def test_structured_error_identity_is_enough_without_a_keyword_in_its_message(wrapped, code):
    error = rejected("stream_options.include_usage", "This endpoint rejected the request field.", code)
    if wrapped:
        error.body = {"error": error.body}
    assert unsupported(error, "stream_options")
    assert not unsupported(error, "max_tokens")


@pytest.mark.parametrize("param", ["model", "max_tokens_limit", 12, ["max_tokens"]])
def test_a_different_structured_parameter_does_not_alias_a_named_budget(param):
    error = rejected(param, "Unsupported parameter: max_tokens")
    assert not unsupported(error, "max_tokens")


@pytest.mark.parametrize("code", ["unsupported_value", "invalid_value"])
def test_unsupported_values_do_not_become_unsupported_fields(code):
    assert not unsupported(rejected("max_tokens", "Unsupported parameter: max_tokens", code), "max_tokens")


@pytest.mark.parametrize("message", ["Unsupported parameter: 'max_tokens'", "Unknown request field: max_tokens",
    "max_tokens is not supported", "This endpoint does not support max_tokens"])
def test_explicit_unstructured_field_diagnostics_remain_supported(message):
    error = rejected(None, message)
    error.body = None
    assert unsupported(error, "max_tokens")


@pytest.mark.parametrize("message", ["max_tokens must be at least 1", "Unsupported model", "Invalid tools schema",
    "The request is invalid", "Unsupported parameter: max_tokens_limit"])
def test_unstructured_unrelated_errors_do_not_disable_budget_fields(message):
    error = rejected(None, message)
    error.body = None
    assert not unsupported(error, "max_tokens")


def test_explicit_tool_parser_capability_diagnostic_is_case_insensitive(home):
    sent = []
    def create(**kwargs):
        sent.append(kwargs)
        if kwargs.get("tools"):
            raise rejected(None, '"Auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser', None)
        return streamed(content="Accepted.")
    llm = LLM(client=client_of(create), model="audit-local", retries=1)
    assert llm.chat(USER, TOOLS).text == "Accepted."
    assert len(sent) == 2 and llm.tool_mode == "prompt" and not sent[-1].get("tools")


@pytest.mark.parametrize("message", ["not supported", "UNSUPPORTED", "not permitted", "not allowed"])
def test_named_field_diagnostics_work_without_a_structured_code(message):
    error = rejected("max_tokens", f"max_tokens is {message}", None)
    error.args = ("The transport reported a rejected request.",)
    assert unsupported(error, "max_tokens")


@pytest.mark.parametrize("hint", ["must be positive", "Must be positive", "maximum 2", "Maximum 2"])
def test_value_constraints_in_named_field_errors_do_not_trigger_capability_fallback(hint):
    assert not unsupported(rejected("max_tokens", f"Unsupported max_tokens value: {hint}", None), "max_tokens")


def test_all_four_capability_fallbacks_can_complete_without_a_transport_retry(home):
    sent = []
    def create(**kwargs):
        sent.append(kwargs)
        payload = kwargs.get("extra_body", {})
        field = next((key for key in ("chat_template_kwargs", "reasoning_effort", "stream_options", "max_tokens")
                      if key in kwargs or key in payload), None)
        if field:
            raise rejected(field, f"Unsupported parameter: {field}")
        return streamed(content="Accepted.")
    llm = LLM(client=client_of(create), model="audit-local", thinking=False, max_tokens=128, retries=1)
    assert llm.chat(USER).text == "Accepted." and len(sent) == 5
    assert sent[-1]["max_completion_tokens"] == 128


@pytest.mark.parametrize("field", ["enable_thinking", "include_usage", "tools"])
def test_nested_field_and_tool_capability_diagnostics_apply_to_the_actual_request(home, field):
    sent = []
    def create(**kwargs):
        sent.append(kwargs)
        offered = kwargs.get("tools") if field == "tools" else kwargs.get("stream_options") if field == "include_usage" \
            else kwargs.get("extra_body", {}).get("chat_template_kwargs")
        if offered:
            raise rejected(field, f"Unsupported parameter: {field}")
        return streamed(content="Accepted.")
    llm = LLM(client=client_of(create), model="audit-local", thinking=False, retries=1)
    assert llm.chat(USER, TOOLS).text == "Accepted." and len(sent) == 2
    assert llm.reasoning_option
    assert llm.template_kwargs == (field != "enable_thinking")
    assert llm.usage_option == (field != "include_usage")
    assert llm.tool_mode == ("prompt" if field == "tools" else None)


@pytest.mark.parametrize("model,field", [("o3", "max_completion_tokens"), ("o4-mini", "max_completion_tokens"),
    ("gpt-5", "max_completion_tokens"), ("gpt-5-mini", "max_completion_tokens"),
    ("audit-local", "max_tokens"), ("gpt-oss-20b", "max_tokens")])
def test_chat_budget_uses_the_selected_model_capability_through_the_real_sdk(home, model, field):
    sent = []
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []}) if request.url.path.endswith("/models") else httpx.Response(404)
        body = json.loads(request.content)
        sent.append(body)
        if body.get(field) != 128 or ({"max_tokens", "max_completion_tokens"} - {field}) & body.keys():
            return httpx.Response(400, json={"error": {"param": next(iter({"max_tokens", "max_completion_tokens"} - {field})),
                "code": "unsupported_parameter", "message": "Unsupported token-budget parameter"}})
        data = {"id": "fixture", "object": "chat.completion.chunk", "created": 0, "model": model,
                "choices": [{"index": 0, "delta": {"content": "Accepted."}, "finish_reason": "stop"}]}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps(data) + "\n\ndata: [DONE]\n\n")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = OpenAI(base_url="https://fixture.test/v1", api_key="synthetic", http_client=http, max_retries=0)
        reply = LLM(client=client, model=model, max_tokens=128, retries=1).chat(USER)
        assert reply.text == "Accepted."
        assert len(sent) == 1 and sent[0][field] == 128


@pytest.mark.parametrize("model,refused,accepted", [("audit-local", "max_tokens", "max_completion_tokens"),
    ("o3", "max_completion_tokens", "max_tokens")])
def test_precise_token_field_fallback_preserves_unrelated_options(home, model, refused, accepted):
    sent = []
    def create(**kwargs):
        sent.append(kwargs)
        if refused in kwargs:
            raise rejected(refused, f"Unsupported parameter: '{refused}'; use '{accepted}' instead.")
        assert kwargs[accepted] == 128
        return streamed(content="Accepted.")
    llm = LLM(client=client_of(create), model=model, max_tokens=128, retries=1, thinking=True)
    assert llm.chat(USER).text == "Accepted."
    assert len(sent) == 2 and sent[0][refused] == 128 and sent[1][accepted] == 128
    assert sent[0].get("extra_body") == sent[1].get("extra_body")
    assert all(request["stream_options"] == {"include_usage": True} for request in sent)
    assert llm.template_kwargs and llm.usage_option
    assert llm.chat(USER).text == "Accepted." and len(sent) == 3 and accepted in sent[-1]


@pytest.mark.parametrize("field,message,code", [
    ("max_tokens", "max_tokens must be at least 1", "invalid_request_error"),
    ("model", "Unsupported model for this endpoint", "model_not_found"),
    ("tools", "Invalid tools: function parameters schema is malformed", "invalid_request_error"),
])
def test_unrelated_bad_requests_do_not_retry_or_downgrade_capabilities(home, field, message, code):
    sent = []
    error = rejected(field, message, code)
    def create(**kwargs):
        sent.append(kwargs)
        raise error
    llm = LLM(client=client_of(create), model="audit-local", max_tokens=128, thinking=False, retries=1)
    with pytest.raises(BadRequestError) as got:
        llm.chat(USER, TOOLS)
    assert got.value is error and len(sent) == 1
    assert llm.template_kwargs and llm.usage_option and llm.tool_mode is None


@pytest.mark.parametrize("content", ["", "Partial explanation."])
def test_structured_refusal_is_preserved_and_its_examples_are_not_executable(home, content):
    refusal = 'Cannot execute <tool_call>{"name":"tree","arguments":{}}</tool_call>.'
    stream = Stream([chunk(content=content), chunk(refusal=refusal[:20]), chunk(refusal=refusal[20:]),
                     NS(choices=[NS(delta=NS(), finish_reason="content_filter")], usage=None)])
    reply = LLM(client=client_of(lambda **kwargs: stream), model="audit-local").chat(USER, TOOLS)
    assert reply.refusal == refusal and refusal in reply.text
    assert reply.text == (content + "\n" if content else "") + refusal
    assert content in reply.text and reply.calls == [] and reply.finish == "content_filter"
    assert stream.closed == 1


def test_refusal_prevents_dispatch_even_when_a_server_also_sends_native_calls(home):
    stream = Stream([chunk(refusal="Cannot answer.", tool_calls=[NS(index=0, id="refused-call",
        function=NS(name="tree", arguments="{}"))]),
        NS(choices=[NS(delta=NS(), finish_reason="content_filter")], usage=None)])
    reply = LLM(client=client_of(lambda **kwargs: stream), model="audit-local").chat(USER, TOOLS)
    assert reply.calls == [] and reply.text == reply.refusal == "Cannot answer."


def test_refusal_already_present_in_content_is_not_duplicated(home):
    reply = LLM(client=client_of(lambda **kwargs: Stream([chunk(content="Cannot answer.", refusal="Cannot answer."),
        NS(choices=[NS(delta=NS(), finish_reason="content_filter")], usage=None)])), model="audit-local").chat(USER)
    assert reply.text == reply.refusal == "Cannot answer."


def test_partial_stream_errors_retain_structured_refusal(home):
    class Broken:
        closed = False
        def __iter__(self):
            yield chunk(refusal="Cannot answer.")
            raise RuntimeError("Connection failed after a refusal.")
        def close(self):
            self.closed = True
    stream = Broken()
    with pytest.raises(RuntimeError) as got:
        _read_stream(stream, None)
    assert got.value.partial["refusal"] == "Cannot answer." and stream.closed


def test_a_cut_stream_retains_the_structured_refusal_in_error_metadata():
    stream = Stream([chunk(refusal="Cannot answer.")])
    with pytest.raises(StreamCut) as got:
        _read_stream(stream, None)
    assert got.value.partial["refusal"] == "Cannot answer."
    assert stream.closed == 1


@pytest.mark.parametrize("field", ["chat_template_kwargs", "reasoning_effort", "stream_options"])
def test_optional_capabilities_fall_back_independently(home, field):
    sent = []
    def create(**kwargs):
        sent.append(kwargs)
        payload = kwargs.get("extra_body", {})
        if field in kwargs or field in payload:
            raise rejected(field, f"Unsupported parameter: '{field}'")
        return streamed(content="Accepted.")
    llm = LLM(client=client_of(create), model="audit-local", thinking=False, retries=1)
    assert llm.chat(USER).text == "Accepted." and len(sent) == 2
    assert llm.template_kwargs == (field != "chat_template_kwargs")
    assert llm.reasoning_option == (field != "reasoning_effort")
    assert llm.usage_option == (field != "stream_options")
    assert sent[-1].get("extra_body", {}) == {key: value for key, value in sent[0]["extra_body"].items() if key != field}


def test_rejected_alternate_budget_stops_after_two_requests_and_restores_capabilities(home):
    sent = []
    def create(**kwargs):
        sent.append(kwargs)
        field = "max_tokens" if "max_tokens" in kwargs else "max_completion_tokens"
        raise rejected(field, f"Unsupported parameter: '{field}'")
    llm = LLM(client=client_of(create), model="audit-local", max_tokens=128, thinking=False, retries=1)
    with pytest.raises(BadRequestError):
        llm.chat(USER)
    assert len(sent) == 2 and llm._token_field is None
    assert llm.template_kwargs and llm.reasoning_option and llm.usage_option


def test_a_later_unrelated_failure_does_not_keep_a_tentative_budget_override(home):
    sent = []
    def create(**kwargs):
        sent.append(kwargs)
        if "max_tokens" in kwargs:
            raise rejected("max_tokens", "Unsupported parameter: max_tokens")
        raise rejected("model", "Unsupported model", "model_not_found")
    llm = LLM(client=client_of(create), model="audit-local", max_tokens=128, thinking=False, retries=1)
    with pytest.raises(BadRequestError, match="Unsupported model"):
        llm.chat(USER)
    assert len(sent) == 2 and llm._token_field is None
    assert llm.template_kwargs and llm.reasoning_option and llm.usage_option


def test_a_parser_failure_after_a_complete_refusal_keeps_the_refusal_without_resending(home):
    requests = []
    class Broken:
        def __iter__(self):
            yield NS(choices=[NS(delta=NS(refusal="Cannot answer."), finish_reason=None)],
                     usage=NS(prompt_tokens=17, completion_tokens=9))
            yield NS(choices=[NS(delta=NS(), finish_reason="content_filter")], usage=None)
            raise type("APIError", (Exception,), {})("Tool parser failed")
        def close(self):
            pass
    llm = LLM(client=client_of(lambda **kwargs: requests.append(kwargs) or Broken()), model="audit-local", retries=1)
    reply = llm.chat(USER, TOOLS)
    assert reply.refusal == reply.text == "Cannot answer." and reply.calls == []
    assert reply.finish == "content_filter" and len(requests) == 1
    assert reply.prompt_tokens == 17 and reply.completion_tokens == 9


def transient_http(phase, protocol_version=PROTOCOL):
    posts, failed = [], False
    def handler(request):
        nonlocal failed
        if request.method == "DELETE":
            return httpx.Response(202)
        body = json.loads(request.content)
        if body["method"] == "initialize":
            params = body["params"]
            assert params["protocolVersion"] == PROTOCOL and params["capabilities"] == {}
            assert isinstance(params["clientInfo"]["name"], str) and params["clientInfo"]["name"]
            assert isinstance(params["clientInfo"]["version"], str) and params["clientInfo"]["version"]
        posts.append((body["method"], dict(request.headers)))
        if body["method"] == phase and not failed:
            failed = True
            return httpx.Response(503, text="Temporary initialization failure.")
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        result = {"protocolVersion": protocol_version, "serverInfo": {"name": "fixture"}, "capabilities": {}} \
            if body["method"] == "initialize" else {"tools": [{"name": "search"}]}
        return httpx.Response(200, headers={"mcp-session-id": "synthetic-session"},
                              json={"jsonrpc": "2.0", "id": body["id"], "result": result})
    return httpx.Client(transport=httpx.MockTransport(handler)), posts


@pytest.mark.parametrize("phase", ["initialize", "notifications/initialized"])
def test_injected_http_mcp_retry_repeats_the_complete_handshake_without_stale_session(phase):
    client, posts = transient_http(phase)
    server = Server("https://fixture.test/mcp", client=client)
    try:
        with pytest.raises(MCPError, match="503"):
            server.tools()
        assert not server._started and server.info is None and server._session is None
        assert not client.is_closed
        assert server.tools() == [{"name": "search"}]
        initial = ["initialize"] if phase == "initialize" else ["initialize", "notifications/initialized"]
        assert [method for method, _ in posts] == initial + ["initialize", "notifications/initialized", "tools/list"]
        assert "mcp-session-id" not in posts[len(initial)][1]
        assert posts[-1][1]["mcp-protocol-version"] == PROTOCOL
    finally:
        server.close()
        assert not client.is_closed
        client.close()


def test_owned_http_client_is_closed_after_failed_initialization_and_replaced_on_retry(monkeypatch):
    first, _ = transient_http("initialize")
    second, posts = transient_http("never")
    clients = iter([first, second])
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: next(clients))
    server = Server("https://fixture.test/mcp")
    try:
        with pytest.raises(MCPError):
            server.tools()
        assert first.is_closed and not second.is_closed
        assert server.tools() == [{"name": "search"}]
        assert [method for method, _ in posts] == ["initialize", "notifications/initialized", "tools/list"]
    finally:
        server.close()
        server.close()
        first.close()
        second.close()


def test_http_session_cleanup_carries_the_negotiated_protocol_version():
    deletes = []
    client, _ = transient_http("never", protocol_version="2025-03-26")
    original = client.delete
    def deleting(url, **kwargs):
        assert url == "https://fixture.test/mcp"
        deletes.append(kwargs["headers"])
        return original(url, **kwargs)
    client.delete = deleting
    server = Server("https://fixture.test/mcp", client=client)
    try:
        server.tools()
        server.close()
        assert deletes == [{"Mcp-Session-Id": "synthetic-session", "MCP-Protocol-Version": "2025-03-26"}]
        assert not client.is_closed
        assert server.tools() == [{"name": "search"}]
    finally:
        server.close()
        client.close()


def test_failed_http_initialization_cleans_an_allocated_session_with_the_proposed_version():
    deletes = []
    def handler(request):
        if request.method == "DELETE":
            deletes.append(dict(request.headers))
            return httpx.Response(202)
        body = json.loads(request.content)
        return httpx.Response(200, headers={"mcp-session-id": "failed-session"},
            json={"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32000, "message": "Initialization failed"}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        server = Server("https://fixture.test/mcp", client=client)
        with pytest.raises(MCPError, match="Initialization failed"):
            server.tools()
        assert len(deletes) == 1
        assert deletes[0]["mcp-session-id"] == "failed-session" and deletes[0]["mcp-protocol-version"] == PROTOCOL
        assert not client.is_closed and server.info is None and server._session is None
        server.close()


def test_owned_http_transport_follows_redirects_and_keeps_the_configured_deadline(monkeypatch):
    original = httpx.Client
    seen = []
    def handler(request):
        seen.append(request)
        if request.url.path == "/redirect":
            return httpx.Response(307, headers={"location": "/mcp"})
        assert request.url.path == "/mcp"
        if request.method == "DELETE":
            return httpx.Response(202)
        body = json.loads(request.content)
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        result = {"protocolVersion": PROTOCOL, "capabilities": {}} if body["method"] == "initialize" else {"tools": []}
        return httpx.Response(200, headers={"mcp-session-id": "session"}, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    server = Server("https://fixture.test/redirect", timeout=2)
    try:
        assert server.tools() == []
        assert any(request.url.path == "/mcp" for request in seen)
        assert all(set(request.extensions["timeout"].values()) == {2} for request in seen)
    finally:
        server.close()


def test_stdio_reader_accepts_corrupt_log_bytes_and_stderr_retention_stays_bounded():
    import queue
    from tests.test_mcp import FAKE
    class Lines:
        def __init__(self, values):
            self.values = iter([*values, b""])
        def readline(self):
            return next(self.values)  # EOF must end the reader, never read beyond it.
    server = Server(FAKE, timeout=2)
    try:
        server.tools()
        lines = queue.Queue()
        server._read_out(NS(stdout=Lines([b"corrupt log \xff\n", b'{"jsonrpc":"2.0","id":4,"result":{"tools":[]}}\n'])), lines)
        assert "corrupt log" in lines.get()
        assert json.loads(lines.get())["result"] == {"tools": []}
        assert lines.get() is None and lines.empty()
        server._stderr.clear()
        server._read_err(NS(stderr=Lines([b"failure \xff X\n"] * 100)), server._stderr)
        retained = len(server._stderr)
        server._read_err(NS(stderr=Lines([b"failure \xff X\n"] * 1000)), server._stderr)
        assert len(server._stderr) == retained and retained > 0
        assert all(line.strip() == "failure \ufffd X" for line in server._stderr)
    finally:
        server.close()


def test_mcp_readers_allow_interpreter_exit_when_a_descendant_keeps_a_pipe_open(tmp_path):
    import os
    import signal
    import subprocess
    pid_file = tmp_path / "owned-mcp.pid"
    service = tmp_path / "descendant_mcp.py"
    service.write_text('''import json,os,subprocess,sys
from pathlib import Path
Path(sys.argv[1]).write_text(str(os.getpid()))
subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])
for line in sys.stdin:
    body=json.loads(line)
    if "id" not in body: continue
    result={"protocolVersion":"2025-06-18","capabilities":{}} if body["method"]=="initialize" else {"tools":[]}
    print(json.dumps({"jsonrpc":"2.0","id":body["id"],"result":result}),flush=True)
''')
    program = ("from pathlib import Path; from peresearch.mcp import Server; "
               f"s=Server([{sys.executable!r},{str(service)!r},{str(pid_file)!r}], timeout=2); s.tools(); s.close()")
    caller = subprocess.Popen([sys.executable, "-c", program], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              start_new_session=True)
    try:
        caller.communicate(timeout=5)
        assert caller.returncode == 0
    finally:
        if pid_file.exists():
            try:
                os.killpg(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if caller.poll() is None:
            os.killpg(caller.pid, signal.SIGKILL)
        caller.communicate(timeout=5)


def test_stdio_server_has_its_own_process_group_for_owned_cleanup():
    import os
    from tests.test_mcp import FAKE
    with Server(FAKE) as server:
        server.tools()
        assert os.getpgid(server.proc.pid) == server.proc.pid


def test_transport_open_failure_does_not_skip_initialization_on_retry(monkeypatch):
    import subprocess
    from tests.test_mcp import FAKE
    opening, calls = subprocess.Popen, []
    def open_once_failing(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 1:
            raise FileNotFoundError("Temporarily unavailable executable")
        return opening(*args, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", open_once_failing)
    server = Server(FAKE, timeout=2)
    try:
        with pytest.raises(MCPError, match="cannot start") as got:
            server.tools()
        assert "Temporarily unavailable executable" in str(got.value)
        assert [tool["name"] for tool in server.tools()] == ["web_search", "fetch"]
        assert len(calls) == 2
    finally:
        server.close()


def test_owned_stdio_failed_initialization_closes_the_old_process_and_restarts_the_handshake(tmp_path):
    script = tmp_path / "retry_mcp.py"
    marker = tmp_path / "initialization-attempted"
    script.write_text('''
import json, sys
from pathlib import Path
marker = Path(sys.argv[1])
initialized = False
for line in sys.stdin:
    body = json.loads(line)
    method = body["method"]
    if method == "notifications/initialized":
        initialized = True
        continue
    if method == "initialize" and not marker.exists():
        marker.write_text("attempted")
        answer = {"error": {"code": -32000, "message": "Temporary initialization failure"}}
    elif method == "initialize":
        answer = {"result": {"protocolVersion": body["params"]["protocolVersion"], "capabilities": {}}}
    elif initialized:
        answer = {"result": {"tools": [{"name": "search"}]}}
    else:
        answer = {"error": {"code": -32000, "message": "Handshake was skipped"}}
    print(json.dumps({"jsonrpc": "2.0", "id": body["id"], **answer}), flush=True)
''')
    server = Server([sys.executable, str(script), str(marker)], timeout=2)
    try:
        with pytest.raises(MCPError, match="Temporary initialization failure"):
            server.tools()
        previous = server.proc
        assert previous.poll() is not None and not server._started
        assert server.tools() == [{"name": "search"}]
        assert server.proc.pid != previous.pid and previous.poll() is not None
    finally:
        server.close()
