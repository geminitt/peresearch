"""OpenAI Chat Completions: what a request looks like on that wire and how its stream is read. Also the form the agent's
size estimate is counted on (`request_size`)."""

import json

from peresearch.messages import Request
from peresearch.protocols import Interrupted, StreamCut, _assembled


def _read_stream(stream, stop, on_first=None, on_text=None) -> dict:
    """A streamed chat completion put back together: text, reasoning, tool calls (their pieces joined by index) and
    token counts. `stop()` is checked between chunks; the stream is always closed by this, the reading thread. A
    stream ended by an error carries what came before it as the error's `partial`."""
    text, reasoning, refusal, calls, usage, finish = [], [], [], {}, None, ""
    try:
        for chunk in stream:
            if on_first is not None:
                on_first()
                on_first = None
            if stop is not None and stop():
                raise Interrupted("stopped by the user")
            usage = getattr(chunk, "usage", None) or usage
            for choice in getattr(chunk, "choices", None) or []:
                finish = getattr(choice, "finish_reason", None) or finish
                delta = choice.delta
                if getattr(delta, "content", None):
                    text.append(delta.content)
                    if on_text is not None:        # the answer as it is written (the interface's draft)
                        on_text(delta.content)
                if getattr(delta, "refusal", None):
                    refusal.append(delta.refusal)
                thought = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if thought:
                    reasoning.append(thought)
                for piece in getattr(delta, "tool_calls", None) or []:
                    key = piece.index if getattr(piece, "index", None) is not None else len(calls)
                    call = calls.setdefault(key, {"id": None, "name": "", "args": ""})
                    call["id"] = getattr(piece, "id", None) or call["id"]
                    fn = getattr(piece, "function", None)
                    call["name"] += (getattr(fn, "name", None) or "") if fn else ""
                    call["args"] += (getattr(fn, "arguments", None) or "") if fn else ""
                    extra = getattr(piece, "extra_content", None)
                    if extra:                       # Gemini's thought signature: sent back with this call
                        call["extra"] = extra
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


def chat_messages(request: Request, provider: bool = True) -> list[dict]:
    """A request's messages as Chat Completions takes them: the system prompt first, an assistant turn's calls as
    `tool_calls` (with what the endpoint attached to each, when `provider`: Gemini's thought signature), each tool
    result as a `tool` message. Reasoning is not sent back on this protocol."""
    out = [{"role": "system", "content": request.system}] if request.system else []
    for m in request.messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.call_id, "content": m.text})
        elif m.role == "assistant" and m.calls:
            calls = []
            for c in m.calls:
                call = {"id": c.id, "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.args, ensure_ascii=False)}}
                extra = (c.provider.get("chat") or {}).get("extra_content") if provider else None
                if extra is not None:
                    call["extra_content"] = extra
                calls.append(call)
            out.append({"role": "assistant", "content": m.text, "tool_calls": calls})
        else:
            out.append({"role": m.role, "content": m.text})
    return out


def request_size(request: Request) -> int:
    """The characters a request sends, counted on its Chat Completions body (the measure the agent's characters per
    token and framing were fitted on): every message's text and calls, and the tool schemas."""
    return sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or "", ensure_ascii=False))
               for m in chat_messages(request, provider=False)) + \
        len(json.dumps(chat_tools(request.tools), ensure_ascii=False))


def chat_tools(tools) -> list[dict]:
    """Tools as Chat Completions declares them: function tools."""
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.parameters}} for t in tools]


def send(client, kw: dict, extras: dict):
    """The request, sent: the Chat Completions body is the gateway's own form."""
    return client.chat.completions.create(**kw)


read = _read_stream


def client(url: str, key: str):
    """The OpenAI SDK against any Chat Completions endpoint; no retries of its own."""
    from openai import OpenAI
    return OpenAI(base_url=url, api_key=key or "none", timeout=600.0, max_retries=0)


def extras(request: Request) -> dict:
    """Nothing beyond the Chat Completions form travels to this API."""
    return {}
