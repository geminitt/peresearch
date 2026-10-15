"""OpenAI's Responses API: the request as that API takes it, and its event stream read back."""

from types import SimpleNamespace

from peresearch.protocols import Interrupted, StreamCut, _assembled


class ResponsesFailed(Exception):
    """The Responses API ended a reply with a failure or an error event."""


def responses_request(messages: list[dict], tools: list[dict] | None) -> dict:
    """A chat request (messages, function tools) as the Responses API takes it: the system prompt as `instructions`,
    the assistant's calls as `function_call` items and tool results as `function_call_output` items matched by call
    id, function tools flat (no "function" wrapper). `store` is off: the provider keeps nothing of it."""
    instructions, items = [], []
    for m in messages:
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            instructions.append(content)
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id") or "", "output": content})
        elif role == "assistant":
            if content:
                items.append({"role": "assistant", "content": content})
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or {}
                items.append({"type": "function_call", "call_id": call.get("id") or "", "name": fn.get("name") or "",
                              "arguments": fn.get("arguments") or "{}"})
        else:
            items.append({"role": "user", "content": content})
    request = {"input": items, "store": False}
    if instructions:
        request["instructions"] = "\n\n".join(instructions)
    if tools:
        request["tools"] = [{"type": "function", "name": t["function"]["name"],
                             "description": t["function"].get("description", ""),
                             "parameters": t["function"].get("parameters") or {"type": "object", "properties": {}}}
                            for t in tools if t.get("type") == "function"]
    return request


def _read_responses(stream, stop, on_first=None, on_text=None) -> dict:
    """A streamed Responses API reply put back together as `_read_stream` returns a chat completion's: text,
    reasoning (its text or its summary), calls by output index with their call ids, token counts, and why it ended
    ("length" for a reply cut at its token limit). An error or failure event raises with what came before it."""
    text, reasoning, refusal, calls, usage, finish = [], [], [], {}, None, ""
    try:
        for event in stream:
            if on_first is not None:
                on_first()
                on_first = None
            if stop is not None and stop():
                raise Interrupted("stopped by the user")
            kind = getattr(event, "type", "")
            if kind == "response.output_text.delta":
                text.append(event.delta)
                if on_text is not None:
                    on_text(event.delta)
            elif kind == "response.refusal.delta":
                refusal.append(event.delta)
            elif kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                reasoning.append(event.delta)
            elif kind == "response.output_item.added" and getattr(event.item, "type", "") == "function_call":
                calls[event.output_index] = {"id": event.item.call_id, "name": event.item.name or "",
                                             "args": event.item.arguments or ""}
            elif kind == "response.function_call_arguments.delta":
                calls.setdefault(event.output_index, {"id": None, "name": "", "args": ""})["args"] += event.delta
            elif kind == "response.function_call_arguments.done":     # the whole arguments: they win
                calls.setdefault(event.output_index, {"id": None, "name": "", "args": ""})["args"] = event.arguments
            elif kind in ("response.completed", "response.incomplete"):
                done = event.response
                got = getattr(done, "usage", None)
                if got is not None:
                    usage = SimpleNamespace(prompt_tokens=got.input_tokens, completion_tokens=got.output_tokens)
                if kind == "response.incomplete":
                    why = getattr(getattr(done, "incomplete_details", None), "reason", None)
                    finish = "content_filter" if why == "content_filter" else "length"
                else:
                    finish = "tool_calls" if calls else "stop"
            elif kind in ("response.failed", "error"):
                said = getattr(event, "message", None) or getattr(getattr(event, "response", None), "error", None)
                raise ResponsesFailed(str(said or "the Responses API reported a failure"))
        if not finish:
            raise StreamCut("the reply stream ended without a finish reason: the connection was cut mid-answer")
    except Exception as e:
        e.partial = _assembled(text, reasoning, calls, usage, finish, refusal)
        raise
    finally:
        close = getattr(stream, "close", None)
        if close:
            close()
    return _assembled(text, reasoning, calls, usage, finish, refusal)


def send(client, kw: dict, extras: dict):
    """The request as the Responses API takes it: the messages and tools translated, the reply budget as
    max_output_tokens, sampling kept."""
    request = responses_request(kw["messages"], kw.get("tools"))
    tokens = kw.get("max_tokens") or kw.get("max_completion_tokens")
    request.update(model=kw["model"], stream=True, **({"max_output_tokens": tokens} if tokens else {}),
                   **{k: kw[k] for k in ("temperature", "top_p") if k in kw})
    return client.responses.create(**request)


read = _read_responses


def client(url: str, key: str):
    """The OpenAI SDK, which serves both of OpenAI's APIs."""
    from peresearch.protocols.chat import client as openai_client
    return openai_client(url, key)


def extras(request) -> dict:
    """Nothing beyond the Chat Completions form travels to this API."""
    return {}
