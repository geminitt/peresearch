"""What every protocol adapter shares: how a request is interrupted, a stream cut short, and a reply put together
from the pieces a stream gave. The adapters, one per wire protocol (model gateway step 2, 2026-10-15): `chat`
(OpenAI Chat Completions, also every compatible server), `responses` (OpenAI Responses API), `anthropic` (Anthropic
Messages), `gemini` (Google's Interactions API). Each translates a request into its protocol's body and reads its
stream back into the same dict (text, reasoning, calls, token counts, finish); `llm.LLM` does the rest."""


class Interrupted(Exception):
    """The caller's stop() asked to stop; the request in flight was dropped."""


class StreamCut(Exception):
    """A streamed reply ended without a finish reason: the connection was cut mid-answer (the OpenAI client raises
    nothing then, and the partial text would pass for a complete answer). Retried like a dropped connection."""


def _assembled(text, reasoning, calls, usage, finish, refusal=()) -> dict:
    result = {"text": "".join(text), "reasoning": "".join(reasoning),
            "calls": [dict(c, id=c["id"] or f"call-{i}") for i, c in sorted(calls.items())],
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0, "finish": finish}
    if refusal:
        result["refusal"] = "".join(refusal)
    return result
