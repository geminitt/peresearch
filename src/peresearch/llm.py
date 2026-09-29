"""The model behind the agent, through any OpenAI-compatible endpoint: vLLM on Modal, or a hosted API.

Every message passes `guard.outbound` before it leaves the machine. Tool calls normally come back structured;
when a Qwen model leaves one as XML inside its reasoning or its text instead (a known vLLM gap with the
qwen3_coder parser, vllm#39056), it is recovered here. A cold Modal container can take minutes to answer the
first request, so dropped connections, timeouts, rate limits and 5xx answers are retried with growing pauses,
and the caller is told why it waits.
"""

import datetime
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from peresearch import guard, settings


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


_QWEN_XML = re.compile(r"<tool_call>\s*<function=([\w\-.]+)>(.*?)</function>\s*</tool_call>", re.S)
_QWEN_PARAM = re.compile(r"<parameter=([\w\-.]+)>\n?(.*?)\n?</parameter>", re.S)
_HERMES_JSON = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _value(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def parse_text_calls(text: str) -> list[ToolCall]:
    """Tool calls written as text: Qwen's XML (<function=…><parameter=…>) or Hermes' JSON inside <tool_call>."""
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
                calls.append(ToolCall(f"text-{len(calls)}", d["name"], d.get("arguments") or {}))
    return calls


def strip_text_calls(text: str) -> str:
    return _HERMES_JSON.sub("", _QWEN_XML.sub("", text or "")).strip()


class BudgetExceeded(Exception):
    """Today's spending cap for the model endpoint is reached."""


class Budget:
    """A daily cap on what a GPU endpoint billed by the second (a Modal container) may cost.

    A container is billed from its start until `scaledown` seconds after its last request. The client cannot see
    the container, so it counts as billed the union of [request start, request end + scaledown] over today's
    requests: an upper bound, since a cold start happens inside the first request. Only request times are stored
    ($PERESEARCH_HOME/usage.jsonl), never content."""

    def __init__(self, path: Path, usd_per_hour: float, cap_usd: float, scaledown: float = 300, clock=time.time):
        self.path, self.rate, self.cap, self.scaledown, self.clock = path, usd_per_hour, cap_usd, scaledown, clock

    @classmethod
    def from_settings(cls, url: str) -> "Budget | None":
        rate = settings.get("PERESEARCH_GPU_USD_PER_HOUR") or ("1.95" if "modal." in (url or "") else "")
        if not rate:
            return None                     # a local server or a provider billed per token: nothing to cap here
        return cls(guard.home() / "usage.jsonl", float(rate), float(settings.get("PERESEARCH_BUDGET_USD_PER_DAY") or 1.0),
                   float(settings.get("PERESEARCH_SCALEDOWN_S") or 300))

    def _today(self) -> list[tuple[float, float]]:
        if not self.path.exists():
            return []
        day = datetime.date.fromtimestamp(self.clock())
        spans = []
        for line in self.path.read_text().splitlines():
            try:
                a, b = json.loads(line)["span"]
            except (ValueError, KeyError):
                continue
            if datetime.date.fromtimestamp(a) == day:
                spans.append((a, b))
        return spans

    def spent(self) -> float:
        """Upper bound on today's spend, in USD."""
        billed, end = 0.0, None
        for a, b in sorted((a, b + self.scaledown) for a, b in self._today()):
            if end is None or a > end:
                billed, end = billed + (b - a), b
            elif b > end:
                billed, end = billed + (b - end), b
        return billed / 3600 * self.rate

    def check(self) -> None:
        spent = self.spent()
        if spent >= self.cap:
            raise BudgetExceeded(f"today's model spending is at ${spent:.2f} of the ${self.cap:.2f} cap "
                                 "(PERESEARCH_BUDGET_USD_PER_DAY); the model is not called again today")

    def record(self, start: float, end: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"span": [start, end]}) + "\n")


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

    RETRYABLE = ("APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError")

    def __init__(self, url: str | None = None, model: str | None = None, key: str | None = None, client=None,
                 max_tokens: int = 4096, temperature: float = 0.6, top_p: float = 0.95, retries: int = 5,
                 wait: float = 10.0, on_wait=None, thinking: bool | None = None, budget: Budget | None = None):
        """`thinking` switches a Qwen model's reasoning on or off (vLLM's chat_template_kwargs); None keeps the
        server's default. PERESEARCH_LLM_THINKING=on/off sets it from the settings."""
        if thinking is None and settings.get("PERESEARCH_LLM_THINKING") in ("on", "off"):
            thinking = settings.get("PERESEARCH_LLM_THINKING") == "on"
        self.thinking, self.template_kwargs = thinking, True
        if client is None:
            from openai import OpenAI

            url = url or settings.get("PERESEARCH_LLM_URL")
            if not url:
                raise RuntimeError("no model endpoint: set PERESEARCH_LLM_URL (see peresearch/settings.py)")
            client = OpenAI(base_url=url, api_key=key or settings.get("PERESEARCH_LLM_KEY") or "none",
                            timeout=600.0, max_retries=0)
            budget = budget or Budget.from_settings(url)
        self.budget = budget
        self.client, self.model = client, model or settings.get("PERESEARCH_LLM_MODEL")
        self.max_tokens, self.temperature, self.top_p = max_tokens, temperature, top_p
        self.retries, self.wait, self.on_wait = retries, wait, on_wait or (lambda msg: None)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, thinking: bool | None = None) -> Reply:
        sent = [_outbound(m) for m in messages]
        thinking = self.thinking if thinking is None else thinking
        for attempt in range(self.retries):
            if self.budget:
                self.budget.check()
            t0 = time.time()
            extra = ({"extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}}}
                     if thinking is not None and self.template_kwargs else {})
            try:
                try:
                    r = self.client.chat.completions.create(model=self.model, messages=sent, tools=tools or None,
                                                            max_tokens=self.max_tokens, temperature=self.temperature,
                                                            top_p=self.top_p, **extra)
                finally:
                    if self.budget:                  # a failed request may still have woken the container
                        self.budget.record(t0, time.time())
                break
            except Exception as e:
                if type(e).__name__ == "BadRequestError" and extra:      # a server without chat_template_kwargs
                    self.template_kwargs = False
                    continue
                if type(e).__name__ not in self.RETRYABLE or attempt == self.retries - 1:
                    raise
                pause = self.wait * 2 ** attempt
                self.on_wait(f"model not ready ({type(e).__name__}); retrying in {pause:.0f} s")
                time.sleep(pause)
        msg = r.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        calls = [ToolCall(c.id, c.function.name, json.loads(c.function.arguments or "{}"))
                 for c in (msg.tool_calls or [])]
        text = msg.content or ""
        if not calls:                       # recovered from the text or the reasoning (vllm#39056)
            calls = parse_text_calls(text) or parse_text_calls(reasoning)
            text = strip_text_calls(text)
        usage = getattr(r, "usage", None)
        return Reply(text, calls, reasoning, getattr(usage, "prompt_tokens", 0) or 0,
                     getattr(usage, "completion_tokens", 0) or 0)
