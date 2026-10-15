"""Anthropic Messages through the official SDK: the request as the API takes it — content blocks, the system prompt
apart, thinking blocks sent back with their signatures — and its stream read back."""

import json
from types import SimpleNamespace

from peresearch.messages import Request
from peresearch.protocols import Interrupted, _assembled


ANTHROPIC_REPLY_TOKENS = 32_000    # max_tokens is required by the Messages API; used only when the model says nothing


def anthropic_request(messages: list[dict], tools: list[dict] | None, blocks: dict) -> dict:
    """A chat request as the Anthropic Messages API takes it: system messages as the top-level `system`; an
    assistant turn with calls as its content blocks — the ones the API returned for it (`blocks`, by the id of its
    first call: thinking blocks with their signatures go back unchanged), else rebuilt as text and tool_use blocks;
    the tool results of a round together in one user message; function tools as name, description, input_schema."""
    system, out = [], []
    for m in messages:
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            system.append(content)
        elif role == "tool":
            result = {"type": "tool_result", "tool_use_id": m.get("tool_call_id") or "", "content": content}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(result)
            else:
                out.append({"role": "user", "content": [result]})
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            kept = blocks.get(calls[0].get("id")) if calls else None
            if kept is not None:
                out.append({"role": "assistant", "content": [dict(b) for b in kept]})
                continue
            parts = [{"type": "text", "text": content}] if content else []
            for call in calls:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                parts.append({"type": "tool_use", "id": call.get("id") or "", "name": fn.get("name") or "",
                              "input": args if isinstance(args, dict) else {}})
            if parts:                          # an empty assistant text is refused by the API
                out.append({"role": "assistant", "content": parts if calls else content})
        else:
            out.append({"role": "user", "content": content})
    request = {"messages": out}
    if system:
        request["system"] = "\n\n".join(system)
    if tools:
        request["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                             "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
                            for t in tools if t.get("type") == "function"]
    return request


def _read_anthropic(manager, stop, on_first=None, on_text=None) -> dict:
    """A streamed Messages API reply put back together as `_read_stream` returns a chat completion's, plus `blocks`:
    the reply's content blocks as sent back with its calls. Reasoning comes from thinking deltas; a refusal's
    explanation is the refusal; the input tokens count the cached ones too."""
    text, reasoning, calls, usage, finish, refusal, blocks = [], [], {}, None, "", [], []
    try:
        with manager as stream:
            for event in stream:
                if on_first is not None:
                    on_first()
                    on_first = None
                if stop is not None and stop():
                    raise Interrupted("stopped by the user")
                if event.type == "text":
                    text.append(event.text)
                    if on_text is not None:
                        on_text(event.text)
                elif event.type == "thinking":
                    reasoning.append(event.thinking)
            final = stream.get_final_message()
        blocks = [b.model_dump(exclude_none=True) for b in final.content]
        for i, b in enumerate(b for b in final.content if b.type == "tool_use"):
            calls[i] = {"id": b.id, "name": b.name, "args": json.dumps(b.input, ensure_ascii=False)}
        got = final.usage
        usage = SimpleNamespace(
            prompt_tokens=(got.input_tokens or 0) + (getattr(got, "cache_read_input_tokens", 0) or 0)
                          + (getattr(got, "cache_creation_input_tokens", 0) or 0),
            completion_tokens=got.output_tokens or 0)
        finish = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop",
                  "refusal": "refusal", "pause_turn": "stop"}.get(final.stop_reason, final.stop_reason or "stop")
        if final.stop_reason == "refusal":
            details = getattr(final, "stop_details", None)
            refusal.append(getattr(details, "explanation", None) or "the model declined to answer")
            calls = {}
    except Exception as e:
        e.partial = _assembled(text, reasoning, calls, usage, finish, refusal)
        raise
    result = _assembled(text, reasoning, calls, usage, finish, refusal)
    result["blocks"] = blocks
    return result


def anthropic_blocks(request: Request) -> dict:
    """The content blocks the Messages API returned for each assistant turn with calls, by its first call's id: they
    go back unchanged (thinking blocks keep their signatures)."""
    return {m.calls[0].id: m.provider["anthropic"]["content"] for m in request.messages
            if m.role == "assistant" and m.calls and "anthropic" in m.provider}


def send(client, kw: dict, extras: dict):
    """The request as the Messages API takes it, `extras` the content blocks of earlier turns by first call id;
    sampling and template fields are not sent: current Claude refuses them."""
    request = anthropic_request(kw["messages"], kw.get("tools"), extras)
    request.update(model=kw["model"], cache_control={"type": "ephemeral"},
                   max_tokens=kw.get("max_tokens") or kw.get("max_completion_tokens") or ANTHROPIC_REPLY_TOKENS)
    return client.messages.stream(**request)


read = _read_anthropic


def client(url: str, key: str):
    """The official Anthropic SDK; the API's base is the host (a "/v1" the user typed is dropped)."""
    import anthropic
    return anthropic.Anthropic(base_url=url.rstrip("/").removesuffix("/v1"), api_key=key or None, timeout=600.0,
                               max_retries=0)


extras = anthropic_blocks
