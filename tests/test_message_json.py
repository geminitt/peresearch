"""A message kept in a conversation file comes back as it was (PLAN.md, memory and session state, part B): every part
— text, reasoning, calls with their arguments and provider data (Gemini's signatures, Claude's thinking blocks), tool
results — survives to_json and from_json, through JSON text."""

import json

from hypothesis import given, strategies as st

from peresearch.messages import Message, Reasoning, Text, ToolCall, ToolResult, from_json, to_json

scalars = st.one_of(st.none(), st.booleans(), st.integers(-10**9, 10**9), st.text(max_size=20))
values = st.recursive(scalars, lambda inner: st.one_of(st.lists(inner, max_size=3),
                                                       st.dictionaries(st.text(max_size=8), inner, max_size=3)),
                      max_leaves=10)
provider = st.dictionaries(st.sampled_from(["chat", "anthropic", "gemini", "responses"]),
                           st.dictionaries(st.text(max_size=8), values, max_size=2), max_size=2)
parts = st.one_of(st.builds(Text, st.text(max_size=40)), st.builds(Reasoning, st.text(max_size=40)),
                  st.builds(ToolCall, st.text(max_size=8), st.text(max_size=8),
                            st.dictionaries(st.text(max_size=8), values, max_size=3), st.text(max_size=20), provider),
                  st.builds(ToolResult, st.text(max_size=8), st.text(max_size=8), st.text(max_size=40)))
messages = st.builds(Message, st.sampled_from(["user", "assistant", "tool"]), st.lists(parts, max_size=4).map(tuple),
                     provider)


@given(messages)
def test_a_message_survives_json(m):
    assert from_json(json.loads(json.dumps(to_json(m), ensure_ascii=False))) == m


def test_the_shape_kept_on_disk_names_each_part():
    result = Message.tool("c1", "read", "[R1: …]")
    assert to_json(result) == {"role": "tool", "provider": {},
                               "parts": [{"type": "result", "call_id": "c1", "name": "read", "content": "[R1: …]"}]}
    m = Message("assistant", (Reasoning("think"), Text("hi"), ToolCall("c1", "read", {"path": "a.md"},
                                                                     provider={"chat": {"extra_content": {"g": 1}}})),
                {"anthropic": {"blocks": [{"type": "thinking", "signature": "s"}]}})
    assert to_json(m) == {"role": "assistant", "provider": {"anthropic": {"blocks": [{"type": "thinking",
                                                                                     "signature": "s"}]}},
                          "parts": [{"type": "reasoning", "text": "think"}, {"type": "text", "text": "hi"},
                                    {"type": "call", "id": "c1", "name": "read", "args": {"path": "a.md"},
                                     "error": "", "provider": {"chat": {"extra_content": {"g": 1}}}}]}
