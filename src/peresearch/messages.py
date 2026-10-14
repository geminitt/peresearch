"""peresearch's own message format, and the interface the agent loop uses to reach a model (owner decision,
2026-10-14: the agent knows only this format; each protocol's format stays inside the gateway).

A message is a role and its parts, in order: what was said (Text), what the model thought before it (Reasoning), the
tools it called (ToolCall) and what a tool returned (ToolResult). What a provider attaches to a message or a part and
needs back later travels with it in `provider`, keyed by protocol: Gemini's thought signature on a call (Chat
Completions' `extra_content`), Claude's content blocks with their signed thinking (Anthropic Messages). A message
built again from these parts, through any gateway object, sends them back; none is kept on the side.

The design follows OpenCode's canonical schema (packages/ai/src/schema/messages.ts, read 2026-10-14 at 34d8ded):
messages of typed parts with provider metadata, the system prompt apart from the messages."""

from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Protocol


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Reasoning:
    text: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict
    error: str = ""                # why the arguments could not be read ({} then): not run, the model is told
    provider: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str



@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    parts: tuple = ()
    provider: dict = field(default_factory=dict)

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls("user", (Text(text),))

    @classmethod
    def assistant(cls, text: str) -> "Message":
        return cls("assistant", (Text(text),))

    @classmethod
    def tool(cls, call_id: str, name: str, content: str) -> "Message":
        return cls("tool", (ToolResult(call_id, name, content),))

    @property
    def text(self) -> str:
        """What the message says: its text parts, or a tool result's content."""
        return "".join(p.content if isinstance(p, ToolResult) else p.text for p in self.parts
                       if isinstance(p, (Text, ToolResult)))

    @property
    def calls(self) -> list[ToolCall]:
        return [p for p in self.parts if isinstance(p, ToolCall)]

    @property
    def call_id(self) -> str | None:
        """The call a tool message answers."""
        return next((p.call_id for p in self.parts if isinstance(p, ToolResult)), None)

    def with_text(self, text: str) -> "Message":
        """The same message saying `text` instead: a tool result's content, or the text of a message without calls
        (a trimmed result, a revoked earlier answer)."""
        if self.role == "tool":
            return replace(self, parts=tuple(replace(p, content=text) if isinstance(p, ToolResult) else p
                                             for p in self.parts))
        assert not self.calls, "a message with calls keeps its text"
        return replace(self, parts=tuple(p for p in self.parts if not isinstance(p, Text)) + (Text(text),))


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict               # JSON Schema of the arguments


@dataclass(frozen=True)
class Request:
    system: str
    messages: tuple = ()
    tools: tuple = ()

    def __post_init__(self):
        assert all(isinstance(m, Message) for m in self.messages), "a request holds Messages"
        assert all(isinstance(t, Tool) for t in self.tools), "a request offers Tools"


class ModelGateway(Protocol):
    """What the agent loop needs from a model: one reply to a request, and what is known of the model's limits.
    None says unknown."""

    on_text: Callable[[str], None] | None       # the answer as it streams, when the caller sets it

    def chat(self, request: Request, thinking: bool | None = None, stop=None): ...

    def context_length(self) -> int | None: ...

    def reply_tokens(self) -> int | None: ...
