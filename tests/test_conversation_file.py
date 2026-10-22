"""One file per conversation (PLAN.md, memory and session state, part B): a header, each turn's line followed by its
messages whole — the question, the model's calls with their provider data, each result as its note with its ref, the
answer — and the conversation's summaries and name; earlier turns go back to the model whole, a resumed conversation
sends what it would have sent uninterrupted, and summarizing or withholding works on whole turns."""

import json

from peresearch.agent import (REVOKED_RESULT, Agent, Limits, conversation_file, conversations, find_conversation,
                              kept_note, read_messages, read_turns, usage)
from peresearch.llm import Reply, ToolCall
from peresearch.messages import Message
from tests.test_agent import FakeModel, FakeToolbox


def records(a):
    return [json.loads(line) for line in a.history_path.read_text().splitlines()]


def tool_round():
    return [Reply("", [ToolCall("c1", "tree", {"q": 1})]), Reply("It holds notes [N1].")]


def test_a_turn_keeps_its_messages_whole_after_its_line(home):
    a = Agent(FakeModel(tool_round()), FakeToolbox())
    a.ask("What is there?")
    lines = records(a)
    assert lines[0]["schema"] == 1 and lines[0]["conversation"] == a.conversation          # a header first
    turn = next(i for i, x in enumerate(lines) if x.get("question") == "What is there?")
    kept = [x for x in lines[turn + 1:] if "message" in x]
    assert all(x["turn"] == 0 for x in kept)
    messages = read_messages(a.history_path)[0]
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[0].text == "What is there?" and messages[1].calls[0].name == "tree"
    assert messages[2].text == kept_note("R1", "tree", {"q": 1}, len("[N1] result 1 of tree"))   # the result: its note
    assert messages[3].text == "It holds notes [N1]."
    assert a.history_path == conversation_file("default", a.conversation)


def test_the_next_question_sends_the_earlier_turn_whole(home):
    model = FakeModel(tool_round() + [Reply("Again [N1].")])
    a = Agent(model, FakeToolbox())
    a.ask("What is there?")
    a.ask("And now?")
    sent = model.seen[-1]["messages"]
    assert [m["role"] for m in sent[1:]] == ["user", "assistant", "tool", "assistant", "user"]
    assert sent[2]["tool_calls"][0]["function"]["name"] == "tree" and sent[3]["content"].startswith("[R1: tree(")


def test_a_resumed_conversation_sends_what_it_would_have_sent(home):
    uninterrupted = FakeModel(tool_round() + [Reply("Again [N1].")])
    a = Agent(uninterrupted, FakeToolbox())
    a.ask("What is there?")
    a.ask("And now?")
    first = Agent(FakeModel(tool_round()), FakeToolbox())
    first.ask("What is there?")
    resumed = Agent(FakeModel([Reply("Again [N1].")]), FakeToolbox())    # another agent, as after a restart
    resumed.resume(first.conversation)
    resumed.ask("And now?")
    assert resumed.llm.seen[-1]["messages"] == uninterrupted.seen[-1]["messages"]


def test_the_harness_prompts_are_not_kept_and_an_interrupted_call_is_answered(home):
    model = FakeModel([Reply("", [ToolCall("c1", "tree", {"q": 1})]), Reply("", [ToolCall("c2", "read", {"q": 2})]),
                       Reply("done")])
    a = Agent(model, FakeToolbox(), limits=Limits(steps=1))
    a.ask("q")
    kept = read_messages(a.history_path)[0]
    assert not any("Limit reached" in m.text for m in kept)
    b = Agent(FakeModel([Reply("", [ToolCall("d1", "tree", {"q": 1}), ToolCall("d2", "read", {"q": 2})])]),
              FakeToolbox())
    real = b.toolbox.call

    def cancel_after_first(name, args):
        b.cancel()
        return real(name, args)
    b.toolbox.call = cancel_after_first
    b.ask("q")
    kept = read_messages(b.history_path)[0]
    results = {m.call_id: m.text for m in kept if m.role == "tool"}
    assert set(results) == {"d1", "d2"} and results["d2"] == "not run: the question was interrupted"
    at = next(i for i, m in enumerate(kept) if m.calls)
    assert [m.role for m in kept[at:]] == ["assistant", "tool", "tool", "assistant"]   # results right after the calls


def test_a_turn_kept_without_its_messages_goes_back_as_question_and_answer(home):
    a = Agent(FakeModel([Reply("ok")]), FakeToolbox())
    a.history_path.parent.mkdir(parents=True, exist_ok=True)
    a.history_path.write_text(json.dumps({"time": 1, "question": "old q", "answer": "old a"}) + "\n")
    assert a.history() == [Message.user("old q"), Message.assistant("old a")]


def test_a_withheld_turn_shows_none_of_its_calls(home):
    a = Agent(FakeModel(tool_round()), FakeToolbox())
    a.ask("What is there?")
    a._turn_allowed = lambda turn: False
    assert a.history() == [Message.user("What is there?"), Message.assistant(REVOKED_RESULT)]


def test_summarizing_frees_whole_turns(home):
    from tests.test_context_fit import Wire, ready, summarize
    a = Agent(FakeModel(tool_round() + [Reply("plain")]), FakeToolbox())
    a.ask("first, with a tool")
    a.ask("second, plain")
    wire = Wire(summarize(), reserve=100)
    b = Agent(wire, FakeToolbox())
    b.resume(a.conversation)
    turns = b._history()
    ready(b, past=len(turns))
    b._sizes = [len(t) for t in turns]
    messages = [m for t in turns for m in t] + [Message.user("now")]
    assert b._fold(messages, 1) is True and b._done == 1
    assert messages[0] == Message.user("second, plain") and b._sizes == [2]
    assert "first, with a tool" in wire.requests[0].messages[0].text


def test_conversations_names_and_usage_come_from_the_conversation_files(home):
    a = Agent(FakeModel([Reply("one")]), FakeToolbox())
    a.ask("first question")
    a.rename("my topic")
    b = Agent(FakeModel([Reply("two"), Reply("three")]), FakeToolbox())
    b.ask("other")
    b.ask("other again")
    listed = {c.id: (c.first, c.questions, c.name) for c in conversations("default")}
    assert listed == {a.conversation: ("first question", 1, "my topic"), b.conversation: ("other", 2, "")}
    assert find_conversation("default", "My Topic").id == a.conversation
    assert usage("default", b.conversation)["requests"] == 2 and usage("default")["requests"] == 3
    assert [t["question"] for t in read_turns(b.history_path)] == ["other", "other again"]


# --- edges the part B mutation run found untested ---

def write(a, *lines):
    a.history_path.parent.mkdir(parents=True, exist_ok=True)
    a.history_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))


def test_the_file_is_the_conversation_and_stray_lines_are_ignored(home):
    a = Agent(FakeModel([]), FakeToolbox())
    ok = Message.user("q")
    write(a, {"time": 1, "conversation": "elsewhere", "question": "q", "answer": "a"},
          {"turn": 0},                                                            # no message: not a broken turn
          {"turn": 0, "message": {"role": "user", "parts": [{"type": "text", "text": "q"}]}},
          {"summary": "no count"},                                                # no turns: not a summary
          {"question": "no time", "answer": "b", "usage": {"requests": 5}})
    assert [t["conversation"] for t in read_turns(a.history_path)] == [a.conversation] * 2
    assert read_messages(a.history_path) == {0: [ok]}
    from peresearch.agent import read_summary
    assert read_summary(a.history_path) == ("", 0)
    listed = conversations("default")[0]
    assert (listed.started, listed.last) == (0.0, 1)                              # a turn without a time: 0
    assert usage("default", since=0.5)["requests"] == 0 and usage("default")["requests"] == 5   # no time: never since


def test_a_turn_with_a_message_that_cannot_be_read_goes_back_as_question_and_answer(home):
    a = Agent(FakeModel([]), FakeToolbox())
    good = {"role": "user", "parts": [{"type": "text", "text": "q"}]}
    write(a, {"time": 1, "question": "q", "answer": "a"}, {"turn": 0, "message": good},
          {"turn": 0, "message": {"role": "user", "parts": [{"type": "unknown"}]}}, {"turn": 0, "message": good})
    assert a.history() == [Message.user("q"), Message.assistant("a")]


def test_a_summarized_conversation_keeps_each_later_turn_with_its_own_messages(home):
    a = Agent(FakeModel(tool_round() + [Reply("plain")]), FakeToolbox())
    a.ask("first, with a tool")
    a.ask("second, plain")
    with a.history_path.open("a") as f:
        f.write(json.dumps({"summary": "the first turn", "turns": 1}) + "\n")
    assert a.history() == [Message.user("second, plain"), Message.assistant("plain")]


def test_a_later_turn_withheld_replaces_its_own_messages_only(home):
    a = Agent(FakeModel(tool_round() + [Reply("plain")] + tool_round()), FakeToolbox())
    for q in ("one", "two", "three"):
        a.ask(q)
    turns = a._history()
    a._past, a._sizes, a._done, a._summary = len(turns), [len(t) for t in turns], 0, ""
    messages = [m for t in turns for m in t] + [Message.user("now")]
    a._turn_allowed = lambda turn: turn["question"] != "three"
    a._filter_access(messages)
    assert messages[:6] == turns[0] + turns[1] and messages[6:8] == [Message.user("three"),
                                                                    Message.assistant(REVOKED_RESULT)]
    assert a._sizes == [4, 2, 2]


def test_message_lines_keep_their_language_and_the_header_is_written_once_with_its_fields(home):
    a = Agent(FakeModel([Reply("Chào bạn"), Reply("lần nữa")]), FakeToolbox())
    a.rename("tên")                                                               # a name before any question
    a.ask("Xin chào")
    a.ask("Tiếng Việt")
    lines = a.history_path.read_text().splitlines()
    assert any('"message"' in x and "Tiếng Việt" in x for x in lines)             # not \u-escaped
    headers = [json.loads(x) for x in lines if '"schema"' in x]
    assert len(headers) == 1 and set(headers[0]) == {"schema", "project", "conversation", "started"}
    assert headers[0]["project"] == "default" and headers[0]["conversation"] == a.conversation


def test_calls_cut_off_before_any_ran_are_answered_in_order(home):
    model = FakeModel([Reply("", [ToolCall("e1", "tree", {"q": 1}), ToolCall("e2", "read", {"q": 2})])])
    a = Agent(model, FakeToolbox())
    model.on_chat = lambda n: a.cancel()                                          # Esc while the model answers
    a.ask("q")
    kept = read_messages(a.history_path)[0]
    results = [(m.call_id, m.parts[0].name, m.text) for m in kept if m.role == "tool"]
    assert results == [("e1", "tree", "not run: the question was interrupted"),
                       ("e2", "read", "not run: the question was interrupted")]


def test_a_folded_turn_is_read_as_its_question_and_its_final_answer(home):
    from tests.test_context_fit import Wire, ready, summarize
    a = Agent(FakeModel(tool_round() + [Reply("plain")]), FakeToolbox())
    a.ask("first, with a tool")
    a.ask("second, plain")
    wire = Wire(summarize(), reserve=100)
    b = Agent(wire, FakeToolbox())
    b.resume(a.conversation)
    turns = b._history()
    ready(b, past=len(turns))
    b._sizes = [len(t) for t in turns]
    b._fold([m for t in turns for m in t] + [Message.user("now")], 1)
    assert wire.requests[0].messages[0].text == "User: first, with a tool\nAssistant: It holds notes [N1].\n\n"
