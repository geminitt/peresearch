"""The agent's own message format (PLAN.md, model gateway step 1, 2026-10-14): what a provider attaches to a turn
travels in the message, so a conversation continued through another gateway object still sends it back; the agent
loop holds no protocol's format.

Red before the change (runs/design-revision-2026-10-14/gateway-step1/red-new-llm.log): Gemini's thought signature
and Claude's thinking block were kept on the LLM object, and a new one sent neither."""

import json
import re
from pathlib import Path

from peresearch import agent as agent_module
from peresearch.llm import CHECK_REQUEST, LLM, REASONING_PROBE, Reply
from peresearch.protocols.chat import chat_messages, chat_tools, request_size
from peresearch.messages import Message, Reasoning, Request, Text, Tool, ToolCall, ToolResult
from tests.test_llm_anthropic import Block, THINK, CALL, claude, reply
from tests.test_llm_gemini import SIGNATURE, chunks_with_call, gemini

TREE = Tool("tree", "folders", {"type": "object", "properties": {}})
SEARCH = Tool("search_notes", "search the notes", {"type": "object", "properties": {"query": {"type": "string"}}})


def continued(question: Request, first) -> Request:
    """The question, the model's turn as its message, and the tool's result."""
    call = first.calls[0]
    return Request(question.system, question.messages + (first.message, Message.tool(call.id, call.name, "result")),
                   question.tools)


def test_a_thought_signature_goes_back_through_a_new_gateway(home, monkeypatch):
    monkeypatch.delenv("PERESEARCH_LLM_TOOLS", raising=False)
    sent = []
    question = Request("", (Message.user("how many projects?"),), (TREE,))
    first = LLM(client=gemini(sent, chunks_with_call()), model="gemini-3.5-flash-lite", retries=1).chat(question)
    assert first.calls[0].provider == {"chat": {"extra_content": SIGNATURE}}
    LLM(client=gemini(sent, chunks_with_call()), model="gemini-3.5-flash-lite", retries=1).chat(continued(question, first))
    replayed = sent[-1][1]["tool_calls"][0]
    assert replayed["extra_content"] == SIGNATURE and replayed["id"] == "function-call-1"


def test_a_signed_thinking_block_goes_back_through_a_new_gateway(home):
    sent = []
    question = Request("rules", (Message.user("what is bpe?"),), (SEARCH,))
    first = LLM(client=claude(sent, [reply([THINK, CALL], stop="tool_use")]), model="claude-opus-5-5", retries=1,
                provider="anthropic").chat(question)
    assert [b["type"] for b in first.message.provider["anthropic"]["content"]] == ["thinking", "tool_use"]
    LLM(client=claude(sent, [reply([Block(type="text", text="BPE merges pairs.")])]), model="claude-opus-5-5",
        retries=1, provider="anthropic").chat(continued(question, first))
    turn = sent[-1]["messages"][1]
    assert turn["role"] == "assistant" and turn["content"][0] == {"type": "thinking", "thinking": "search first",
                                                                  "signature": "sig-1"}


def test_a_reply_is_the_assistant_message_in_order():
    call = ToolCall("c1", "tree", {}, provider={"chat": {"extra_content": SIGNATURE}})
    m = Reply("looking", [call], reasoning="first the folders").message
    assert m.role == "assistant" and m.parts == (Reasoning("first the folders"), Text("looking"), call)
    assert m.calls[0].provider == {"chat": {"extra_content": SIGNATURE}} and m.text == "looking"
    assert Reply("done").message.parts == (Text("done"),)                    # no reasoning part when there was none


def test_trimming_a_message_keeps_its_calls_and_ids():
    result = Message.tool("c1", "read", "long text")
    assert result.with_text("[trimmed]").call_id == "c1" and result.with_text("[trimmed]").text == "[trimmed]"
    assert Message.assistant("old answer").with_text("revoked").parts == (Text("revoked"),)


def test_the_agent_loop_holds_no_protocol_format():
    source = Path(agent_module.__file__).read_text()
    for word in ("tool_calls", "tool_call_id", '"type": "function"', '["content"]', '["role"]'):
        assert word not in source, f"agent.py still uses {word}"
    assert not re.search(r"getattr\(self\.llm|hasattr\(self\.llm", source)     # the gateway's interface is declared


def test_a_reply_built_without_provider_data_is_rebuilt_for_the_protocol(home):
    """A reply this gateway did not return (a test's, or a turn kept from before) carries no provider data: its
    message goes back rebuilt as the protocol's own blocks."""
    hand = Reply("", [ToolCall("toolu_9", "search_notes", {"query": "bpe"})])
    assert hand.provider == {} and hand.message.provider == {}
    sent = []
    llm = LLM(client=claude(sent, [reply([Block(type="text", text="ok")])]), model="claude-opus-5-5", retries=1,
              provider="anthropic")
    question = Request("rules", (Message.user("what is bpe?"),), (SEARCH,))
    llm.chat(Request(question.system, question.messages + (hand.message, Message.tool("toolu_9", "search_notes", "r")),
                     question.tools))
    assert sent[-1]["messages"][1] == {"role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_9", "name": "search_notes", "input": {"query": "bpe"}}]}


def test_call_arguments_go_back_with_their_characters_unescaped():
    turn = Message("assistant", (Text(""), ToolCall("c1", "search_notes", {"query": "phở bò"})))
    sent = chat_messages(Request("", (Message.user("q"), turn)))
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == '{"query": "phở bò"}'


def test_the_size_estimate_counts_text_calls_and_tools_but_not_provider_data():
    """Characters as the agent's ratio was fitted on (agent.CHARS_PER_TOKEN): each message's text and its calls as
    Chat Completions writes them, then the tools; what a provider attached to a call is not counted."""
    call = ToolCall("c1", "tree", {}, provider={"chat": {"extra_content": SIGNATURE}})
    req = Request("sys", (Message.user("q"), Message("assistant", (Text("t"), call)), Message.tool("c1", "tree", "res")),
                  (TREE,))
    calls = [{"id": "c1", "type": "function", "function": {"name": "tree", "arguments": "{}"}}]
    nothing = len(json.dumps(""))                                  # a message without calls counts its text and ""
    expected = (3 + nothing) + (1 + nothing) + (1 + len(json.dumps(calls))) + (3 + nothing) + \
        len(json.dumps(chat_tools((TREE,))))
    assert request_size(req) == expected


def test_the_model_check_asks_what_it_asked_before_the_internal_format():
    """The /doctor probes, byte for byte as they were sent before 2026-10-14 (llm.CHECK_MESSAGES, CHECK_TOOL)."""
    assert chat_messages(CHECK_REQUEST) == [
        {"role": "system", "content": "You are being checked. When asked to call a tool, call it."},
        {"role": "user", "content": "Call the tree tool now."}]
    assert chat_tools(CHECK_REQUEST.tools) == [{"type": "function", "function": {
        "name": "tree", "description": "List what the user's folders contain.",
        "parameters": {"type": "object", "properties": {}}}}]
    assert chat_messages(REASONING_PROBE) == [{"role": "user", "content": "What is 17 * 23? Answer briefly."}]


def test_a_signature_retry_drops_only_thinking_and_keeps_the_other_blocks_as_returned(home):
    class BadRequestError(Exception):
        pass
    call = Block(type="tool_use", id="toolu_1", name="search_notes", input={"query": "bpe"}, caller={"type": "direct"})
    sent = []
    replies = [reply([THINK, call], stop="tool_use"), BadRequestError("Invalid `signature` in `thinking` block"),
               reply([Block(type="text", text="ok")])]
    llm = LLM(client=claude(sent, replies), model="claude-opus-5-5", retries=1, provider="anthropic")
    question = Request("rules", (Message.user("what is bpe?"),), (SEARCH,))
    llm.chat(continued(question, llm.chat(question)))
    assert sent[2]["messages"][1]["content"] == [{"type": "tool_use", "id": "toolu_1", "name": "search_notes",
                                                  "input": {"query": "bpe"}, "caller": {"type": "direct"}}]


def test_a_message_says_its_parts_in_order_and_names_what_it_answers():
    assert Message("assistant", (Text("a"), Reasoning("r"), Text("b"))).text == "ab"      # its text parts, joined
    result = Message.tool("c1", "read", "x")
    assert result.parts == (ToolResult("c1", "read", "x"),) and result.call_id == "c1"
    assert Message.user("q").call_id is None and Message.assistant("a").call_id is None
