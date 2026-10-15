"""Gemini through Google's Interactions API with the official SDK (`google-genai`): the request as that API takes it,
and its event stream read back (owner decision 2026-10-15).

WARNING, read before relying on it. Google calls the Interactions API "recommended for all new projects" and
generateContent "legacy" (ai.google.dev/gemini-api/docs/interactions, read 2026-10-15), but the API changed in May 2026
(Google's breaking-changes page could not be read that day) and OpenCode still offers it as an initial, opt-in
protocol. peresearch always sends `store=false`: Google keeps stored interactions 55 days on paid tiers (1 day free),
and peresearch keeps the conversation itself. Without stored interactions the whole history is resent each turn;
whether Google's implicit caching applies then is unknown, so each turn may bill the whole history. Checked only with
fixtures shaped by the SDK's types (google-genai 2.31) until a live call. The SDK is asked for one attempt; its
Interactions client may still retry a 5xx or 429 once by itself, under peresearch's own retries.

Stateless use, as Google documents it: the input is the conversation as steps — `user_input`, the model's own steps of
each turn "exactly as received" (thought steps with their signatures, `model_output`, `function_call`), then
`function_result` items matched by `call_id`. The model's steps of a turn with calls travel in the message's provider
data (messages.Message.provider["gemini"]["steps"]); a turn without them is rebuilt as text and calls."""

import json
from types import SimpleNamespace
from urllib.parse import urlsplit

from peresearch.messages import Request
from peresearch.protocols import Interrupted, StreamCut, _assembled


class GeminiFailed(Exception):
    """The stream reported an error or a failed interaction."""


def client(url: str, key: str):
    """The google-genai client; its base is the endpoint's host (a saved address of the OpenAI-compatible endpoint,
    …/v1beta/openai/, reaches the same host)."""
    from google import genai
    from google.genai import types
    parts = urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}/" if parts.netloc else None
    return genai.Client(api_key=key or None, http_options=types.HttpOptions(
        base_url=base, timeout=600_000, retry_options=types.HttpRetryOptions(attempts=1)))


def extras(request: Request) -> dict:
    """The model's own steps of each earlier turn with calls, by the id of its first call."""
    return {m.calls[0].id: m.provider["gemini"]["steps"] for m in request.messages
            if m.role == "assistant" and m.calls and "gemini" in m.provider}


def body(messages: list[dict], tools: list[dict] | None, steps: dict) -> dict:
    """A request (Chat Completions messages, as the gateway holds them after `guard`) as the Interactions API takes it."""
    system, items, names = [], [], {}
    for m in messages:
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            system.append(content)
        elif role == "user":
            items.append({"type": "user_input", "content": [{"type": "text", "text": content}]})
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            for call in calls:
                names[call.get("id")] = (call.get("function") or {}).get("name") or ""
            kept = steps.get(calls[0].get("id")) if calls else None
            if kept is not None:                   # exactly as received: thought signatures included
                items.extend(dict(step) for step in kept)
                continue
            if content:
                items.append({"type": "model_output", "content": [{"type": "text", "text": content}]})
            for call in calls:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                items.append({"type": "function_call", "id": call.get("id") or "", "name": fn.get("name") or "",
                              "arguments": args if isinstance(args, dict) else {}})
        elif role == "tool":
            call_id = m.get("tool_call_id") or ""
            items.append({"type": "function_result", "call_id": call_id, "name": names.get(call_id, ""),
                          "result": [{"type": "text", "text": content}]})
    request = {"input": items, "store": False}
    if system:
        request["system_instruction"] = "\n\n".join(system)
    if tools:
        request["tools"] = [{"type": "function", "name": t["function"]["name"],
                             "description": t["function"].get("description", ""),
                             "parameters": t["function"].get("parameters") or {"type": "object", "properties": {}}}
                            for t in tools if t.get("type") == "function"]
    return request


def send(client, kw: dict, extras: dict):
    """The request sent, streamed: the reply budget as max_output_tokens, sampling kept, thought summaries asked for
    (Ctrl+R shows them)."""
    request = body(kw["messages"], kw.get("tools"), extras)
    config = {"thinking_summaries": "auto"}
    tokens = kw.get("max_tokens") or kw.get("max_completion_tokens")
    if tokens:
        config["max_output_tokens"] = tokens
    config.update({k: kw[k] for k in ("temperature", "top_p") if k in kw})
    request.update(model=kw["model"], stream=True, generation_config=config)
    return client.interactions.create(**request)


def _dump(value) -> dict:
    return value.model_dump(exclude_none=True, by_alias=True) if hasattr(value, "model_dump") else dict(value)


def read(stream, stop, on_first=None, on_text=None) -> dict:
    """A streamed interaction put back together as the other readers return theirs, plus `steps`: the model's own steps
    as they are to be sent back with its calls. Text and thought summaries arrive as deltas, a call's arguments as
    pieces of JSON, a thought's signature as its last delta; usage and the end come with `interaction.completed`. An
    `error` event, or an interaction that failed, raises with what came before it."""
    text, reasoning, calls, usage, finish, steps = [], [], {}, None, "", {}
    streamed = set()                   # the calls whose arguments came as deltas: those replace what the start held
    try:
        for event in stream:
            if on_first is not None:
                on_first()
                on_first = None
            if stop is not None and stop():
                raise Interrupted("stopped by the user")
            kind = getattr(event, "event_type", "")
            if kind == "step.start":
                step = _dump(event.step)
                steps[event.index] = step
                if step.get("type") == "function_call":
                    calls[event.index] = {"id": step.get("id"), "name": step.get("name") or "",
                                          "args": json.dumps(step["arguments"]) if step.get("arguments") else ""}
                elif step.get("type") == "model_output":
                    step["content"] = [{"type": "text", "text": ""}]
            elif kind == "step.delta":
                delta, step = event.delta, steps.setdefault(event.index, {})
                what = getattr(delta, "type", "")
                if what == "text":
                    text.append(delta.text)
                    if on_text is not None:
                        on_text(delta.text)
                    if step.get("type") == "model_output":
                        step["content"][0]["text"] += delta.text
                elif what == "thought_summary":
                    piece = getattr(getattr(delta, "content", None), "text", None) or ""
                    reasoning.append(piece)
                    step.setdefault("summary", []).append({"type": "text", "text": piece})
                elif what == "thought_signature":
                    step["signature"] = delta.signature
                elif what == "arguments_delta":
                    calls.setdefault(event.index, {"id": step.get("id"), "name": step.get("name") or "", "args": ""})
                    if event.index not in streamed:
                        streamed.add(event.index)
                        calls[event.index]["args"] = ""
                    calls[event.index]["args"] += delta.arguments or ""
            elif kind == "step.stop":
                step = steps.get(event.index, {})
                if step.get("type") == "function_call" and event.index in calls:
                    try:
                        step["arguments"] = json.loads(calls[event.index]["args"] or "{}")
                    except ValueError:
                        step["arguments"] = {}     # the gateway reports the unreadable arguments (llm._arguments)
            elif kind == "interaction.completed":
                done = event.interaction
                got = getattr(done, "usage", None)
                if got is not None:
                    usage = SimpleNamespace(prompt_tokens=got.total_input_tokens or 0,
                                            completion_tokens=(got.total_output_tokens or 0)
                                            + (got.total_thought_tokens or 0))
                status = getattr(done, "status", "")
                if status in ("failed", "cancelled"):
                    raise GeminiFailed(f"the interaction ended {status}")
                finish = "length" if status == "incomplete" else "tool_calls" if calls else "stop"
            elif kind == "error":
                error = getattr(event, "error", None)
                raise GeminiFailed(getattr(error, "message", None) or "the Interactions API reported an error")
        if not finish:
            raise StreamCut("the reply stream ended without a finish reason: the connection was cut mid-answer")
    except Exception as e:
        e.partial = _assembled(text, reasoning, calls, usage, finish)
        raise
    finally:
        close = getattr(stream, "close", None)
        if close:
            close()
    result = _assembled(text, reasoning, calls, usage, finish)
    result["steps"] = [steps[i] for i in sorted(steps) if steps[i].get("type") in ("thought", "model_output",
                                                                                   "function_call")]
    return result


def retryable(error: BaseException) -> bool:
    """A server error or a rate limit, as google-genai raises them (its APIConnectionError and APITimeoutError are
    retried by name like the other SDKs')."""
    from google.genai import errors
    return isinstance(error, errors.ServerError) or isinstance(error, errors.ClientError) and error.code == 429


def list_models(client) -> list[str]:
    """The model IDs the key may use, without "models/"."""
    return sorted({m.name.removeprefix("models/") for m in client.models.list() if getattr(m, "name", None)})
