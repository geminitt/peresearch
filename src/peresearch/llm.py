"""The model behind the agent, through one gateway: any OpenAI-compatible API — a server the user runs (Ollama,
llama.cpp, LM Studio, vLLM…) or a provider's. How the user gets that API is theirs; adapting to the model behind it
(its profile, tool calls through the prompt, reasoning left in the text) is done here.

Every message passes `guard.outbound` before it leaves the machine. Tool calls normally come back structured;
when a model leaves one as text inside its reasoning or its answer instead (Qwen's XML, Hermes' JSON; e.g.
vllm#39056), it is recovered here. A server that is starting can take minutes to answer the first request, so
dropped connections, timeouts, rate limits and 5xx answers are retried with growing pauses,
and the caller is told why it waits. Replies are streamed (nothing is shown as it streams) so that a caller's
`stop()` interrupts at once, even mid-answer: the reading thread closes the stream at the next chunk, and vLLM
stops generating 0.27 s later (measured), so a GPU is not kept busy for an answer nobody will read.
"""

import datetime
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from peresearch import guard, profiles, settings


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    error: str = ""                # why the arguments could not be read ({} then): not run, the model is told


@dataclass
class Reply:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish: str = ""               # the server's finish_reason: "stop", "tool_calls", "length" (cut at max_tokens)…
    reasoning_in_text: bool = False   # the reasoning came inside the text (<think>) and was moved out
    refusal: str = ""              # structured refusal, preserved separately from content/reasoning


_QWEN_XML = re.compile(r"<tool_call>\s*<function=([\w\-.]+)>(.*?)</function>\s*</tool_call>", re.S)
_QWEN_PARAM = re.compile(r"<parameter=([\w\-.]+)>\n?(.*?)\n?</parameter>", re.S)
_HERMES_JSON = re.compile(r"<tool_call>\s*(\{(?:(?!</tool_call>).)*?\})\s*</tool_call>", re.S)   # within one tag


def _value(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def _arguments(raw, cut: bool = False) -> tuple[dict, str]:
    """A call's arguments as a dict, or ({}, why) when they are not one: JSON cut at max_tokens or malformed, or JSON
    of another kind. Raising here ended the whole question (diagram mark 1); the agent hands the reason to the model
    instead, and the history it sends back holds {} (a server parses the arguments it is sent)."""
    if raw is None or raw == "":
        return {}, "the reply was cut at max_tokens before the call's arguments" if cut else ""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as e:
            return {}, f"the arguments are not valid JSON: {e}" + (" (the reply was cut at max_tokens)" if cut else "")
    if isinstance(raw, dict):
        return raw, ""
    return {}, f"the arguments are a {type(raw).__name__}, not a JSON object"


def _bare_calls(text: str, names) -> list[tuple[int, int, str, dict]]:
    """(start, end, name, arguments) of each call written as bare JSON, the tags left out — {"name", "arguments"},
    or Llama 3.x's {"type": "function", "function": {"name", "parameters"}} (Qwen3-0.6B and Llama-3.2-1B wrote both
    through Ollama, 2026-10-06/07) — taken only when its name is one of the tools offered and its arguments an
    object: other JSON in an answer is the answer's."""
    found, decode, i = [], json.JSONDecoder().raw_decode, (text or "").find("{")
    while i >= 0:
        try:
            d, end = decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(d, dict):
            f = d.get("function") if isinstance(d.get("function"), dict) else d
            args = f.get("arguments", f.get("parameters", {}))
            if f.get("name") in names and isinstance(args, dict):
                found.append((i, end, f["name"], args))
        i = text.find("{", end)
    return found


_EMPTY_FENCE = re.compile(r"```[a-z]*\s*```")


def parse_text_calls(text: str, names=()) -> list[ToolCall]:
    """Tool calls written as text: Qwen's XML (<function=…><parameter=…>) or Hermes' JSON inside <tool_call>; else
    bare JSON naming one of the tools offered (`names`)."""
    calls = []
    for m in _QWEN_XML.finditer(text or ""):
        args = {p.group(1): _value(p.group(2)) for p in _QWEN_PARAM.finditer(m.group(2))}
        calls.append(ToolCall(f"text-{len(calls)}", m.group(1), args))
    if not calls:
        for m in _HERMES_JSON.finditer(text or ""):
            try:
                d = json.loads(m.group(1))
            except ValueError:
                continue
            if isinstance(d, dict) and "name" in d:
                calls.append(ToolCall(f"text-{len(calls)}", d["name"], *_arguments(d.get("arguments"))))
    if not calls and names:
        calls = [ToolCall(f"text-{i}", name, args) for i, (_, _, name, args) in enumerate(_bare_calls(text, names))]
    return calls


def split_reasoning(text: str) -> tuple[str, str]:
    """(reasoning, answer) from a text that carries the model's reasoning itself, as servers do that do not separate
    it: <think>…</think> at the start, only a closing </think> (the chat template opened the tag in the prompt), or
    a <think> never closed (the reply was cut). A text that only mentions the tag further on is left alone."""
    start = text.lstrip()
    if start.startswith("<think>"):
        body = start[len("<think>"):]
        if "</think>" in body:
            thought, rest = body.split("</think>", 1)
            return thought.strip(), rest.strip()
        return body.strip(), ""
    if "</think>" in text and "<think>" not in text:
        thought, rest = text.split("</think>", 1)
        return thought.strip(), rest.strip()
    return "", text


def strip_text_calls(text: str, names=()) -> str:
    text = _HERMES_JSON.sub("", _QWEN_XML.sub("", text or ""))
    if names:
        for start, end, _, _ in reversed(_bare_calls(text, names)):
            text = text[:start] + text[end:]
        text = _EMPTY_FENCE.sub("", text)
    return text.strip()


def executable_text_calls(text: str, names=()) -> list[ToolCall]:
    """Recover protocol-shaped calls, not bare JSON embedded in an explanation.

    Explicit tool-call tags retain their existing protocol. A bare JSON reply
    must consist entirely of calls; fences or surrounding prose are answer data.
    The permissive parsing helpers remain useful for protocol diagnostics.
    """
    if "```" in (text or ""):
        return []
    tagged = bool(_QWEN_XML.search(text or "") or _HERMES_JSON.search(text or ""))
    if not tagged and strip_text_calls(text, names):
        return []
    return parse_text_calls(text, names)


_TOO_LONG = re.compile(r"context length|context window|maximum context|too many tokens|prompt is too long|"
                       r"exceeds the available context size", re.I)


def too_long(error: BaseException) -> tuple[int | None, int | None] | None:
    """(the model's context, the prompt's tokens) when `error` is the server refusing a conversation longer than the
    model's context (each None when the message does not say), else None. vLLM's 400 says both: "maximum context
    length is 16384 tokens. However, you requested 4096 output tokens and your prompt contains 12289 input tokens"."""
    text = str(error)
    if type(error).__name__ != "BadRequestError" or not _TOO_LONG.search(text):
        return None
    # vLLM: "context length is N … prompt contains P"; llama.cpp: "request (P tokens) exceeds the available context
    # size (N tokens)"; OpenAI: "… resulted in P tokens"; Anthropic: "too long: P tokens > N maximum"
    context = re.search(r"context length is (\d+)|> ?(\d+) maximum|context size \((\d+) tokens\)", text)
    prompt = re.search(r"prompt contains (\d+)|resulted in (\d+) tokens|too long: (\d+) tokens|(\d+) input tokens|"
                       r"request \((\d+) tokens\)", text)

    def number(m):
        return int(next(g for g in m.groups() if g)) if m else None
    return number(context), number(prompt)


_TOOLS_REFUSED = re.compile(r"auto.{0,4} tool choice requires --enable-auto-tool-choice and --tool-call-parser", re.I)


def unsupported(error, field: str) -> bool:
    """A field explicitly refused, not an invalid value, model, schema or arbitrary 400."""
    if type(error).__name__ != "BadRequestError":
        return False
    body = getattr(error, "body", None)
    body = body if isinstance(body, dict) else {}
    body = body["error"] if isinstance(body.get("error"), dict) else body
    param, code = body.get("param"), body.get("code")
    if code in ("unsupported_value", "invalid_value"):
        return False
    text = str(body.get("message") or error)
    if re.search(r"must be|at least|invalid value|unsupported value|malformed|minimum|maximum", text, re.I):
        return False
    if param is not None:
        if not isinstance(param, str) or not (param == field or param.startswith(field + ".")):
            return False
        return code in ("unsupported_parameter", "unknown_parameter", "unrecognized_parameter", "extra_forbidden") \
            or bool(re.search(r"not supported|unsupported|unrecognized|unknown|not permitted|not allowed", text, re.I))
    name = re.escape(field)
    return bool(re.search(rf"(?:unsupported|unrecognized|unknown|unexpected)(?: request)?"
                          rf"(?: argument| parameter| field)?\s*[:=]?\s*[\"'`]?{name}\b|"
                          rf"\b{name}[\"'`]?\s+(?:is |are )?(?:not supported|unsupported|not permitted|not allowed|an unknown field)|"
                          rf"(?:does not|doesn't|cannot)\s+support\s+[\"'`]?{name}\b", text, re.I))


# a server's tool-call parser that could not read the model's output ends the stream with an error event (llama.cpp:
# "The model produced output that does not match the expected peg-native format", recorded 2026-10-06)
_CALL_UNREADABLE = re.compile(r"\bpars(e|er|ing)\b|\bformat\b", re.I)


class CallUnreadable(Exception):
    """The server could not read the tool call the model wrote: this request goes through the prompt, where the
    client reads the call itself. Raised inside the client only."""


class ToolsRefused(Exception):
    """The server refused the request's `tools` (no tool-call parser, or a model without tool calls): the calls go
    through the prompt from now on. Raised inside the client only."""


def _hf_name(root: str) -> str:
    """A Hugging Face name from a path in its cache (".../models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/…"), as vLLM
    reports the served model's `root` when it loads from the cache offline; else the root itself."""
    m = re.search(r"models--([^/]+?)--([^/]+)", root or "")
    return f"{m.group(1)}/{m.group(2)}" if m else root


CALL_FORMAT = ("\n\n## Tools\nYou can call the tools below. To call one, write on its own line exactly\n"
               "<tool_call>{\"name\": \"<tool name>\", \"arguments\": {<arguments as JSON>}}</tool_call>\n"
               "You may write several calls; then stop and wait: each result comes back inside <tool_response> tags. "
               "Write no call when you are ready to answer.\n\n")


def prompt_tools(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """The conversation for a model whose tool calls go through the prompt (Continue's "system message tools"): the
    tools described in the system message, earlier calls written back as <tool_call> text, their results as user
    turns in <tool_response> tags, and turns of one role merged so that user and assistant alternate."""
    names, out = {}, []
    listing = "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')}\n  parameters: "
                        + json.dumps(t["function"].get("parameters", {}), ensure_ascii=False, separators=(",", ":"))
                        for t in tools or [])
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if role == "system" and tools:
            m = {"role": "system", "content": content + CALL_FORMAT + listing}
        elif role == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                names[c["id"]] = c["function"]["name"]
                args = c["function"].get("arguments") or "{}"
                calls.append("<tool_call>" + json.dumps({"name": c["function"]["name"], "arguments": json.loads(args)},
                                                        ensure_ascii=False) + "</tool_call>")
            m = {"role": "assistant", "content": "\n".join([content.strip()] * bool(content.strip()) + calls)}
        elif role == "tool":
            m = {"role": "user", "content": f"<tool_response name=\"{names.get(m.get('tool_call_id'), 'tool')}\">\n"
                                            f"{content}\n</tool_response>"}
        else:
            m = {"role": role, "content": content}
        if out and out[-1]["role"] == m["role"] and m["role"] != "system":
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append(m)
    return out


class Interrupted(Exception):
    """The caller's stop() asked to stop; the request in flight was dropped."""


class StreamCut(Exception):
    """A streamed reply ended without a finish reason: the connection was cut mid-answer (the OpenAI client raises
    nothing then, and the partial text would pass for a complete answer). Retried like a dropped connection."""


def _asked_wait(error: BaseException) -> float | None:
    """The wait a server asks for with a 429 or 5xx, in seconds, when it is a sensible one (0 < s ≤ 60, the OpenAI
    SDK's rule): retry-after-ms, else Retry-After in seconds or as an HTTP date. None: the backoff decides."""
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        if headers.get("retry-after-ms") is not None:
            wait = float(headers["retry-after-ms"]) / 1000
        elif headers.get("retry-after") is not None:
            raw = headers["retry-after"]
            try:
                wait = float(raw)
            except ValueError:
                from email.utils import parsedate_to_datetime
                wait = (parsedate_to_datetime(raw) - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        else:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return wait if math.isfinite(wait) and 0 < wait <= 60 else None


def _assembled(text, reasoning, calls, usage, finish, refusal=()) -> dict:
    result = {"text": "".join(text), "reasoning": "".join(reasoning),
            "calls": [dict(c, id=c["id"] or f"call-{i}") for i, c in sorted(calls.items())],
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0, "finish": finish}
    if refusal:
        result["refusal"] = "".join(refusal)
    return result


def _read_stream(stream, stop, on_first=None) -> dict:
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


def _outbound(message: dict) -> dict:
    m = dict(message)
    if isinstance(m.get("content"), str):
        m["content"] = guard.outbound(m["content"], "model")
    if m.get("tool_calls"):
        m["tool_calls"] = [{**c, "function": {**c["function"],
                                              "arguments": guard.outbound(c["function"]["arguments"], "model")}}
                           for c in m["tool_calls"]]
    return m


class LLM:
    """chat(messages, tools) -> Reply. `client` replaces the OpenAI client in tests."""

    RETRYABLE = ("APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError", "StreamCut")

    def __init__(self, url: str | None = None, model: str | None = None, key: str | None = None, client=None,
                 max_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None, retries: int = 5,
                 wait: float = 10.0, on_wait=None, thinking: bool | None = None,
                 slow_start: float = 5.0):
        """`thinking` switches a Qwen model's reasoning on or off (vLLM's chat_template_kwargs); None keeps the
        server's default. PERESEARCH_LLM_THINKING=on/off sets it from the settings. `temperature` and `top_p` given
        here win over the model's profile (peresearch.profiles), which sends them only when it has them."""
        switch = settings.get("PERESEARCH_LLM_THINKING")
        if switch not in ("", "on", "off"):
            raise ValueError("PERESEARCH_LLM_THINKING must be on or off, or unset for the server's default "
                             f"(got {switch!r})")
        if thinking is None and switch:
            thinking = switch == "on"
        context = settings.get("PERESEARCH_LLM_CONTEXT")
        if context and not (context.isdigit() and int(context) >= 1024):
            raise ValueError(f"PERESEARCH_LLM_CONTEXT must be a whole number of tokens, at least 1024 (got {context!r})")
        self._context = int(context) if context else ...      # ... = not asked yet
        self._listed, self._profile, self.tool_mode = ..., None, None    # the server's models; the model's profile
        self._all, self._ran_with = None, ...
        self.thinking, self.template_kwargs, self.usage_option = thinking, True, True
        self.reasoning_option, self._token_field = True, None
        self.on_request = None  # optional observer of submitted, filtered model-visible messages
        self.local = False                             # a server on this machine: refused means not running
        if client is None:
            from openai import OpenAI

            url = url or settings.get("PERESEARCH_LLM_URL")
            if not url:
                raise RuntimeError("no model endpoint: set PERESEARCH_LLM_URL (see peresearch/settings.py)")
            client = OpenAI(base_url=url, api_key=key or settings.get("PERESEARCH_LLM_KEY") or "none",
                            timeout=600.0, max_retries=0)
            from urllib.parse import urlparse
            self.local = urlparse(url).hostname in ("localhost", "127.0.0.1", "::1")
        self.client, self.model = client, model or settings.get("PERESEARCH_LLM_MODEL") or None
        self.max_tokens, self.temperature, self.top_p = max_tokens, temperature, top_p
        self.retries, self.wait, self.on_wait = retries, wait, on_wait or (lambda msg: None)
        assert retries >= 1, "at least one attempt"
        self.slow_start = slow_start                   # seconds without a first chunk before saying so

    def _models(self) -> list:
        """The server's model list (asked once); empty when it gives none."""
        if self._all is None:
            try:
                self._all = list(self.client.models.list().data)
            except Exception:
                self._all = []
        return self._all

    def _name(self) -> str:
        """The model's name: the one given, else the server's only model; else the user is told what to set."""
        if not self.model:
            ids = [m.id for m in self._models()]
            if len(ids) == 1:
                self.model = ids[0]
            else:
                said = (f"the server offers {len(ids)} models ({', '.join(ids[:5])}{', …' if len(ids) > 5 else ''})"
                        if ids else "the server lists no model")
                raise RuntimeError(f"no model name: {said}; set it with /setup (in a terminal: `peresearch setup`)")
        return self.model

    def _served(self):
        """The server's entry for this model, or None: a hosted API may give no list."""
        if self._listed is ...:
            name = self._name()
            self._listed = next((m for m in self._models() if m.id == name), None)
        return self._listed

    def _field(self, name: str):
        m = self._served()
        return getattr(m, name, None) or (getattr(m, "model_extra", None) or {}).get(name) if m is not None else None

    def profile(self) -> "profiles.Profile":
        """The model's profile, looked up by the served name, else by vLLM's `root` for an alias (a path in the
        Hugging Face cache read back as its name); built once."""
        if self._profile is None:
            root = self._field("root")
            names = [n for n in (self._name(), root and _hf_name(root)) if n]
            known = next((n for n in names if profiles.lookup(n)), self.model)
            self._profile = profiles.profile(known)
        return self._profile

    def _server_context(self) -> int | None:
        """The context the server runs the model with, asked once: vLLM's max_model_len, llama.cpp's meta.n_ctx (its
        n_ctx_train is the model's own), Ollama's for the loaded model. The model table's limit is the model's
        maximum, often far above what a server on a small GPU runs (131,072 against 8,192, 2026-10-06)."""
        if self._ran_with is ...:
            meta = self._field("meta")
            found = self._field("max_model_len") or (meta.get("n_ctx") if isinstance(meta, dict) else None)
            self._ran_with = int(found) if found else self._loaded_context()
        return self._ran_with

    def _loaded_context(self) -> int | None:
        """Ollama's context for this model: its /v1/models says none, its /api/ps says it for a loaded model (an empty
        /api/generate loads it). Asked only of a server whose /api/ps answers; anything else is unknown. Ollama cuts
        a longer conversation SILENTLY, its start first (recorded 2026-10-06: 36,015 tokens sent, 4,098 kept, HTTP
        200), so the agent must know this number."""
        http, base = getattr(self.client, "_client", None), str(getattr(self.client, "base_url", "")).rstrip("/")
        if http is None or not base.endswith("/v1"):
            return None
        root, name = base[:-3], self._name()
        names = {name, name + ":latest"}    # untagged "tiny" is listed "tiny:latest" (Ollama 0.32.15, 2026-10-06)

        def loaded():
            r = http.get(root + "/api/ps", timeout=10)
            if r.status_code != 200:
                return None, False
            return next((int(m["context_length"]) for m in r.json().get("models", [])
                         if m.get("name") in names), None), True
        try:
            found, ollama = loaded()
            if found is None and ollama:
                http.post(root + "/api/generate", json={"model": name}, timeout=600)
                found, _ = loaded()
            return found
        except Exception:
            return None

    def context_source(self) -> str:
        """Where context_length() comes from: "setting", "server", "model table", or "" (unknown)."""
        if settings.get("PERESEARCH_LLM_CONTEXT"):
            return "setting"
        if self._server_context():
            return "server"
        return "model table" if self.profile().context else ""

    def context_length(self) -> int | None:
        """The served model's context in tokens, for the agent to keep a conversation within it: the setting
        PERESEARCH_LLM_CONTEXT, else what the server runs it with, else the model's profile (models.dev's limit),
        else None (unknown)."""
        if self._context is ...:
            self._context = self._server_context() or self.profile().context
        return self._context

    def reply_tokens(self) -> int | None:
        """The tokens a reply may use: the smallest of what is known — the model's output limit (its profile), a
        quarter of its context, a cap the caller sets — or None (nothing known: the server's own). No fixed size
        (owner 2026-10-07): a fixed 4,096 left Ollama's default 4,096 context no room for the conversation, every
        tool result cut to its floor (runs/e2e/e2e-silent-cut-2026-10-07)."""
        context = self.context_length()
        known = [n for n in (self.max_tokens, self.profile().output, context // 4 if context else None) if n]
        return min(known) if known else None

    def _create(self, stop, **kw) -> dict:
        """One streamed request, put back together by `_read_stream`. With `stop`, the stream is read in a helper
        thread, so the caller is free the moment stop() is true; the reader closes the stream itself at its next
        chunk (closing it from another thread does not wake a blocked read, and vLLM went on generating)."""
        kw["stream"] = True
        if self.usage_option:
            kw["stream_options"] = {"include_usage": True}
        def create():
            if self.on_request is not None:
                from copy import deepcopy
                self.on_request(deepcopy(kw["messages"]))
            return self.client.chat.completions.create(**kw)

        if stop is None:
            return _read_stream(create(), None)
        done, first, out = threading.Event(), threading.Event(), {}

        def run():
            try:
                out["reply"] = _read_stream(create(), stop, first.set)
            except BaseException as e:            # handed to the caller below
                out["error"] = e
            done.set()
        threading.Thread(target=run, daemon=True).start()
        start, announced = time.time(), False
        while not done.wait(0.1):
            if stop():
                raise Interrupted("stopped by the user")
            if not announced and not first.is_set() and time.time() - start > self.slow_start:
                self.on_wait("the model has not started answering yet (a server that is starting can take minutes)")
                announced = True
            elif announced and first.is_set():
                self.on_wait("Thinking")
                announced = None                       # said once, both ways
        if announced:                                  # the whole answer came before the loop saw its start
            self.on_wait("Thinking")
        assert "reply" in out or "error" in out, "the reader thread ended with neither a reply nor an error"
        if "error" in out:
            raise out["error"]
        return out["reply"]

    def chat(self, messages: list[dict], tools: list[dict] | None = None, thinking: bool | None = None,
             stop=None) -> Reply:
        """`stop()`, polled every 0.1 s, interrupts the request in flight or the wait before a retry."""
        sent = [_outbound(m) for m in messages]
        thinking = self.thinking if thinking is None else thinking
        optional = (self.template_kwargs, self.usage_option, self.reasoning_option, self._token_field)
        prompted = tools and (self.tool_mode or self.profile().tools) == "prompt"
        try:
            try:
                r = self._send(prompt_tools(sent, tools) if prompted else sent, None if prompted else tools,
                               thinking, stop)
            except ToolsRefused:            # the server takes no `tools`: through the prompt, now and from now on
                self.tool_mode = "prompt"
                r = self._send(prompt_tools(sent, tools), None, thinking, stop)
            except CallUnreadable as e:     # the server could not read this output: this request only
                got = getattr(e.__cause__, "partial", None) or {}
                calls = got.get("calls") or []
                if got.get("refusal"):
                    r = got
                elif calls and all(c["name"] and c["args"] and not _arguments(c["args"])[1] for c in calls):
                    r = {**got, "finish": "tool_calls"}     # whole calls before what the parser failed on: used
                else:
                    r = self._send(prompt_tools(sent, tools), None, thinking, stop)
        except BaseException:              # no request succeeded without a dropped field: it was not the cause
            self.template_kwargs, self.usage_option, self.reasoning_option, self._token_field = optional
            raise
        # vLLM reports a call cut at max_tokens as finish "tool_calls" (measured: runs/serve/
        # client-real-check-2026-10-04.log): a reply that used every token allowed may be cut, only its last call
        budget = self.reply_tokens()
        cut = r["finish"] == "length" or bool(budget) and budget <= r["completion_tokens"]
        refusal = r.get("refusal", "")
        calls = [] if refusal else [ToolCall(c["id"], c["name"], *_arguments(c["args"], cut and i == len(r["calls"]) - 1))
                 for i, c in enumerate(r["calls"])]
        thought, text = split_reasoning(r["text"])
        reasoning = "\n".join(part for part in (r["reasoning"], thought) if part)
        if not calls and not refusal:       # recovered from the text or the reasoning (vllm#39056)
            names = {(t.get("function") or {}).get("name") for t in tools or []} - {None}
            calls = executable_text_calls(text, names) or executable_text_calls(reasoning, names)
            if calls:
                text = strip_text_calls(text, names)
            else:
                text = text.strip()
        # the agent reads c.args.get(...) and runs a call unless it has an error: arguments are a dict, {} with one
        assert all(isinstance(c.args, dict) and not (c.error and c.args) for c in calls), \
            f"tool-call arguments not a dict, or kept with an error: {[(c.name, type(c.args).__name__) for c in calls]}"
        if refusal and refusal != text:
            text = "\n".join(part for part in (text.strip(), refusal) if part)
        return Reply(text, calls, reasoning, r["prompt_tokens"], r["completion_tokens"], r["finish"], bool(thought), refusal)

    def _send(self, sent: list[dict], tools, thinking, stop) -> dict:
        """Retry transport failures and precisely refused capabilities. Only an explicitly unsupported field
        falls back without using a transport try. Unrelated 400s propagate unchanged; chat restores unsuccessful
        capability changes. A token field gets one alternate, never an unbounded toggle."""
        attempt, refusals, sends, budget_retried = 0, 0, 0, False
        while True:
            if stop is not None and (attempt or getattr(stop, "access_paths", None)) and stop():
                raise Interrupted("stopped before the model request")
            sends += 1
            assert sends <= self.retries + 4, f"{sends} requests for {self.retries} tries and 4 capability fallbacks"
            prof = self.profile()
            switchable = prof.thinking == "template" or "family" not in prof.source    # unknown: tried, as before
            # the chat template's flag (vLLM, llama.cpp); Ollama ignores it and turns reasoning off only through its
            # own field (checked on Ollama 0.32.15 with qwen3:0.6b, 2026-10-07); each field falls back separately
            payload = {}
            if thinking is not None and switchable:
                if self.template_kwargs:
                    payload["chat_template_kwargs"] = {"enable_thinking": thinking}
                if thinking is False and self.reasoning_option:
                    payload["reasoning_effort"] = "none"
            extra = {"extra_body": payload} if payload else {}
            sampling = {k: v for k, v in (("temperature", self.temperature if self.temperature is not None
                                           else prof.temperature),
                                          ("top_p", self.top_p if self.top_p is not None else prof.top_p))
                        if v is not None}
            tokens = self.reply_tokens()
            field = self._token_field or prof.token_field
            budget = {field: tokens} if tokens else {}                  # nothing known: the server's own
            try:
                return self._create(stop, model=self._name(), messages=sent, tools=tools or None,
                                    **budget, **sampling, **extra)
            except Interrupted:
                raise
            except Exception as e:
                if too_long(e):                     # the conversation's length, not a field: the caller trims it
                    raise
                if tools and (unsupported(e, "tools") or type(e).__name__ == "BadRequestError" and _TOOLS_REFUSED.search(str(e))):
                    raise ToolsRefused(str(e)) from e
                if type(e).__name__ == "APIError" and tools and _CALL_UNREADABLE.search(str(e)):
                    raise CallUnreadable(str(e)) from e
                if budget and not budget_retried and unsupported(e, field):
                    self._token_field = "max_completion_tokens" if field == "max_tokens" else "max_tokens"
                    budget_retried = True
                    continue
                if "chat_template_kwargs" in payload and (unsupported(e, "chat_template_kwargs") or unsupported(e, "enable_thinking")):
                    self.template_kwargs = False
                    continue
                if "reasoning_effort" in payload and unsupported(e, "reasoning_effort"):
                    self.reasoning_option = False
                    continue
                if self.usage_option and (unsupported(e, "stream_options") or unsupported(e, "include_usage")):
                    self.usage_option = False
                    continue
                # Nothing listening on this machine will not start listening by waiting: a refusal gets one more
                # try, not 2.5 minutes of retries (a remote endpoint may be waking up, so it keeps them). The last
                # try always raises.
                refused_here = self.local and type(e).__name__ == "APIConnectionError"
                refusals += refused_here
                if type(e).__name__ not in self.RETRYABLE or attempt == self.retries - 1 or refusals == 2:
                    raise
                asked = None if refused_here else _asked_wait(e)
                pause = 2.0 if refused_here else asked if asked is not None else self.wait * 2 ** attempt
                self.on_wait(f"model not ready ({type(e).__name__}); retrying in {pause:.0f} s")
                end = time.time() + pause
                while time.time() < end:
                    if stop is not None and stop():
                        raise Interrupted("stopped by the user")
                    time.sleep(min(0.1, end - time.time()))
                attempt += 1


CHECK_TOOL = {"type": "function", "function": {"name": "tree", "description": "List what the user's folders contain.",
                                               "parameters": {"type": "object", "properties": {}}}}
CHECK_MESSAGES = [{"role": "system", "content": "You are being checked. When asked to call a tool, call it."},
                  {"role": "user", "content": "Call the tree tool now."}]
REASONING_PROBE = [{"role": "user", "content": "What is 17 * 23? Answer briefly."}]


SMALL_CONTEXT = 16_384     # below it the model check says the context is small for the agent


def check_model(llm: LLM) -> tuple[bool, list[str]]:
    """Whether the model works with the agent, and what was seen: how it calls tools, where its reasoning comes back,
    its context, its speed. One small request with one tool, reasoning off (a second the other way — native or
    through the prompt — when the first brings no call), then one without tools, reasoning on; the /doctor step after
    /setup, not a gate: the user decides."""
    p = llm.profile()
    lines = [f"model: {llm.model} · " + (f"{p.family} family (model table)" if p.family
                                                                      else "not in the model table: defaults")]
    keep, llm.max_tokens = llm.max_tokens, min(llm.max_tokens or 1024, 1024)
    try:
        r = llm.chat(CHECK_MESSAGES, [CHECK_TOOL], thinking=False)
        mode = llm.tool_mode or p.tools
        called = any(c.name == "tree" for c in r.calls)
        advice = ""
        if not called and llm.tool_mode is None:     # the table may be wrong: the other way (not after a refusal)
            other = "prompt" if mode == "native" else "native"
            llm.tool_mode = other
            again = llm.chat(CHECK_MESSAGES, [CHECK_TOOL], thinking=False)
            if any(c.name == "tree" for c in again.calls):
                r, called, mode = again, True, other
                advice = f" (set PERESEARCH_LLM_TOOLS={other} to keep it)"
            elif llm.tool_mode == other:
                llm.tool_mode = None
        t0 = time.monotonic()               # timed last: the first request may wait for the server to load the model
        thought = llm.chat(REASONING_PROBE, None, thinking=True)   # where reasoning comes back, with no tools sent
        took = time.monotonic() - t0
    finally:
        llm.max_tokens = keep
    recovered = any(c.id.startswith("text-") for c in r.calls)
    lines.append("tool calls: " + ("none: the model did not call the test tool; the agent can only answer from "
                                   "memory, without your files or the web" if not called else
                                   ("through the prompt" if mode == "prompt" else
                                    "native, written as text and recovered (the server's tool-call parser may be off)"
                                    if recovered else "native") + advice))
    # a server that drops the closing tag when no tools are sent (vLLM without --reasoning-parser, measured
    # 2026-10-06: the model wrote </think>, the text came back without it) leaves nothing the client can split on
    lines.append("reasoning: " + ("inside the text, moved out of the answer" if thought.reasoning_in_text else
                                  "in its own field" if thought.reasoning.strip() else
                                  "none came back separately; if answers start with the model's thinking, turn on "
                                  "the server's reasoning parser (vLLM: --reasoning-parser)"))
    context = llm.context_length()
    # 16,384: a page of search_notes (22,868 characters measured, ~7,600 tokens), the instructions and tools (~2,000)
    # and a reply (4,096) fit whole; below it, results are cut to fit (Ollama's default is 4,096 on most GPUs)
    lines.append((f"context: {context:,} tokens (from the {llm.context_source()})" + (
        "; small for the agent: long results are shown in parts and earlier turns summarized sooner — run the model "
        "with 16,384 or more "
        "(Ollama: OLLAMA_CONTEXT_LENGTH=16384)" if context < SMALL_CONTEXT else "")) if context else
                 "context: unknown (set PERESEARCH_LLM_CONTEXT, or the conversation is trimmed when refused)")
    n = thought.completion_tokens
    lines.append(f"speed: {n / took:.0f} tokens/s ({n} tokens in {took:.1f} s)" if n and took > 0 else
                 "speed: unknown (the server reports no token counts)")
    return called, lines
