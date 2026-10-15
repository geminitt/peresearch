"""Where a model can run, declared once: each provider's endpoint, protocol, the roles it serves and what it does
differently (`Compatibility`). The gateway picks its route from here (design decision 2026-10-14) instead of reading it
from a refusal; a server's errors are read only for what cannot be known beforehand. What one model needs (its reply
budget field, tools only through OpenAI's Responses API) is the model profile's (profiles.py).

Endpoints checked on 2026-10-14 against each provider's documentation: platform.openai.com API reference;
platform.claude.com/docs (Messages API; the OpenAI-compatible layer is "not considered … production-ready");
ai.google.dev/gemini-api/docs/interactions (read 2026-10-15: "recommended for all new projects"); openrouter.ai/docs
/quickstart; console.groq.com/docs/openai; ollama.com/blog/openai-compatibility; lmstudio.ai/docs. A local server's
address stays editable: llama-server's default port changed between versions (8080 installed here, 9931 in its current
README) and vLLM takes --port. A provider the user adds names its own protocol (connections.Connection.protocol)."""

from dataclasses import dataclass, field
from urllib.parse import urlsplit

PROTOCOLS = ("chat", "responses", "anthropic", "gemini")


@dataclass(frozen=True)
class Compatibility:
    """What a provider's servers do differently from their protocol, each field from an observed difference."""
    # how a model's reasoning is turned off: "" tries the chat template's enable_thinking and reasoning_effort, each
    # dropped when refused; "reasoning_effort" sends only that (Ollama ignores the template's flag silently: checked
    # on Ollama 0.32.15 with qwen3:0.6b, 2026-10-07)
    thinking_switch: str = ""
    # whether the server may be Ollama, whose /v1/models gives no context: its /api/ps is asked (a cut it makes is
    # silent, so the agent must know the number); a cloud API is never asked
    ollama_probe: bool = False


LOCAL = Compatibility(ollama_probe=True)


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    url: str
    protocol: str                  # "chat" (OpenAI Chat Completions), "anthropic" (Messages), "gemini" (Interactions)
    kind: str                      # "cloud" (a key, a fixed endpoint) or "local" (an address, usually no key)
    roles: tuple = ("chat",)       # the model roles it serves (connections.ROLES)
    compatibility: Compatibility = field(default_factory=Compatibility)


PROVIDERS = (
    Provider("openai", "OpenAI", "https://api.openai.com/v1", "chat", "cloud", ("chat", "embedding")),
    Provider("anthropic", "Anthropic (Claude)", "https://api.anthropic.com", "anthropic", "cloud", ("chat",)),
    # chat through the native Interactions API (protocols.gemini reaches this URL's host); embeddings through the
    # OpenAI-compatible endpoint the URL names
    Provider("google", "Google AI Studio (Gemini)", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini",
             "cloud", ("chat", "embedding")),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "chat", "cloud"),
    Provider("groq", "Groq", "https://api.groq.com/openai/v1", "chat", "cloud"),
    Provider("ollama", "Ollama", "http://localhost:11434/v1", "chat", "local", ("chat", "embedding"),
             Compatibility(thinking_switch="reasoning_effort", ollama_probe=True)),
    Provider("lmstudio", "LM Studio", "http://localhost:1234/v1", "chat", "local", ("chat", "embedding"), LOCAL),
    Provider("local", "vLLM, llama.cpp or another server", "http://localhost:8000/v1", "chat", "local",
             ("chat", "embedding", "rerank"), LOCAL),
)
_BY_ID = {p.id: p for p in PROVIDERS}


def get(provider_id: str) -> Provider | None:
    return _BY_ID.get(provider_id or "")


def guess(url: str) -> str:
    """The provider of a cloud endpoint, by its host (a connection saved before providers were recorded); "" for a
    server the registry cannot tell apart (any local or other endpoint)."""
    host = urlsplit(url or "").hostname or ""
    return next((p.id for p in PROVIDERS if p.kind == "cloud" and urlsplit(p.url).hostname == host), "")


def protocol(provider_id: str, chosen: str = "") -> str:
    """A registered provider's protocol; else the one the user chose for the provider they added; else Chat
    Completions, which every compatible server speaks."""
    found = get(provider_id)
    if found:
        return found.protocol
    return chosen if chosen in PROTOCOLS else "chat"


def compatibility(provider_id: str) -> Compatibility:
    """A registered provider's differences; a server the registry cannot tell apart may be Ollama."""
    found = get(provider_id)
    return found.compatibility if found else LOCAL
