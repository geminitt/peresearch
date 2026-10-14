"""Where a model can run, declared once: each provider's endpoint, protocol, the roles it serves and what it does
differently. The gateway picks its route from here (design decision 2026-10-14) instead of reading it from a refusal.

Endpoints checked on 2026-10-14 against each provider's documentation: platform.openai.com API reference;
platform.claude.com/docs (Messages API; the OpenAI-compatible layer is "not considered … production-ready"); ai.google
.dev/gemini-api/docs/openai; openrouter.ai/docs/quickstart; console.groq.com/docs/openai; ollama.com/blog/openai-
compatibility; lmstudio.ai/docs. A local server's address stays editable: llama-server's default port changed between
versions (8080 installed here, 9931 in its current README) and vLLM takes --port."""

from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    url: str
    protocol: str                  # "chat" (OpenAI Chat Completions) or "anthropic" (Anthropic Messages)
    kind: str                      # "cloud" (a key, a fixed endpoint) or "local" (an address, usually no key)
    roles: tuple = ("chat",)       # the model roles it serves (connections.ROLES)
    quirks: frozenset = field(default_factory=frozenset)
    # "responses-for-tools": models that take tools only through the Responses API (OpenAI)
    # "thought-signatures": tool calls carry a thought signature to send back (Gemini's OpenAI-compatible endpoint)


PROVIDERS = (
    Provider("openai", "OpenAI", "https://api.openai.com/v1", "chat", "cloud", ("chat", "embedding"),
             frozenset({"responses-for-tools"})),
    Provider("anthropic", "Anthropic (Claude)", "https://api.anthropic.com", "anthropic", "cloud", ("chat",)),
    Provider("google", "Google AI Studio (Gemini)", "https://generativelanguage.googleapis.com/v1beta/openai/", "chat",
             "cloud", ("chat", "embedding"), frozenset({"thought-signatures"})),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "chat", "cloud"),
    Provider("groq", "Groq", "https://api.groq.com/openai/v1", "chat", "cloud"),
    Provider("ollama", "Ollama", "http://localhost:11434/v1", "chat", "local", ("chat", "embedding")),
    Provider("lmstudio", "LM Studio", "http://localhost:1234/v1", "chat", "local", ("chat", "embedding")),
    Provider("local", "vLLM, llama.cpp or another server", "http://localhost:8000/v1", "chat", "local",
             ("chat", "embedding", "rerank")),
)
_BY_ID = {p.id: p for p in PROVIDERS}


def get(provider_id: str) -> Provider | None:
    return _BY_ID.get(provider_id or "")


def guess(url: str) -> str:
    """The provider of a cloud endpoint, by its host (a connection saved before providers were recorded); "" for a
    server the registry cannot tell apart (any local or other OpenAI-compatible endpoint)."""
    host = urlsplit(url or "").hostname or ""
    return next((p.id for p in PROVIDERS if p.kind == "cloud" and urlsplit(p.url).hostname == host), "")


def protocol(provider_id: str) -> str:
    found = get(provider_id)
    return found.protocol if found else "chat"
