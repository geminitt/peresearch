"""The context-fitting paths one by one (PLAN.md step 3.5: step 1's mutation review deferred them; step 3's mutation
found them untested): what a cut result says it left out (_shown), the note an earlier result becomes (_note_of),
fitting a request (_fit) — earlier rounds trimmed oldest first, the latest round cut longest first, a kept result
shown as its head and where to read on — and where one page ends (_page_end). Every number is derived from the rule
the code states, so a changed rule fails here."""

import pytest

from peresearch import agent as A
from peresearch import registry
from peresearch.agent import (CUT, FULL, MORE, TRIMMED, _fit, _note_of, _page_end, _shown, _size, _wrap,
                              kept_note)
from peresearch.messages import Message, Text, ToolCall
from peresearch.registry import Registry, ToolSpec, fitting, head


def call(cid):
    return Message("assistant", (Text(""), ToolCall(cid, "read", {"q": cid})))


def tool(cid, text):
    return Message.tool(cid, "read", text)


# --- _shown and _note_of ---

def test_a_cut_result_says_its_whole_length():
    assert _shown("abc") == ("abc", 3)
    assert _shown("x" * 10 + CUT.format(n=5)) == ("x" * 10, 15)


def test_an_earlier_result_becomes_its_note_when_that_is_shorter():
    assert _note_of(tool("c", "y" * 500), None) == TRIMMED.format(n=500)
    assert _note_of(tool("c", "y" * 500 + CUT.format(n=1_000)), {}) == TRIMMED.format(n=1_500)   # counts what was cut
    held = {"c": ("R4", "read", {"path": "a.md"}, 9_000)}
    assert _note_of(tool("c", "y" * 500), held) == kept_note("R4", "read", {"path": "a.md"}, 9_000)
    equal = next(n for n in range(50, 400) if len(TRIMMED.format(n=n)) == n)
    assert _note_of(tool("c", "y" * equal), None) == "y" * equal                 # no shorter: left as it is
    assert _note_of(tool("c", "y"), None) == "y"


# --- _fit: earlier rounds ---

def earlier_round(big=5_000):
    return [Message.user("q"), call("c0"), tool("c0", "a" * big), call("c1"), tool("c1", "small")]


def test_a_request_that_fits_exactly_is_left_whole():
    messages = earlier_round()
    before = list(messages)
    _fit("sys", messages, None, _size("sys", messages, None))
    assert messages == before


def test_one_character_over_trims_the_oldest_result_to_its_note():
    messages = earlier_round()
    _fit("sys", messages, None, _size("sys", messages, None) - 1)
    assert messages[2].text == TRIMMED.format(n=5_000)
    assert messages[4].text == "small" and messages[0].text == "q"


def test_the_users_messages_and_notes_already_made_are_never_trimmed():
    messages = [Message.user("u" * 9_000), call("c0"), tool("c0", TRIMMED.format(n=70_000) + " " * 300), call("c1"),
                tool("c1", "x")]
    kept_already = [Message.user("q"), call("c0"), tool("c0", kept_note("R1", "read", {}, 9) + ""), call("c1"),
                    tool("c1", "x")]
    _fit("sys", messages, None, 10)
    assert messages[0].text == "u" * 9_000 and messages[2].text.startswith(TRIMMED.format(n=70_000))
    before = kept_already[2].text
    _fit("sys", kept_already, None, 10)
    assert kept_already[2].text == before


# --- _fit: the latest round, results not kept (a refusal, an error) ---

def latest_round(a=3_000, b=8_000):
    return [Message.user("q"), Message("assistant", (Text(""), ToolCall("c0", "read", {}), ToolCall("c1", "read", {}))),
            tool("c0", "a" * a), tool("c1", "b" * b)]


def test_the_longest_of_the_latest_round_is_cut_first_and_says_what_it_left():
    messages = latest_round()
    over = 2_000
    _fit("sys", messages, None, _size("sys", messages, None) - over)
    keep = 8_000 - over - 80
    assert messages[3].text == "b" * keep + CUT.format(n=8_000 - keep)
    assert messages[2].text == "a" * 3_000                                     # enough freed: the other whole


def test_a_cut_keeps_at_least_500_characters_and_then_cuts_the_next():
    messages = latest_round()
    _fit("sys", messages, None, _size("sys", messages, None) - 7_900)
    assert messages[3].text == "b" * 500 + CUT.format(n=7_500)
    assert messages[2].text.startswith("a" * 500) and messages[2].text.endswith(" characters left out]")
    body, whole = _shown(messages[2].text)
    assert whole == 3_000 and len(body) < 3_000


def test_a_result_cut_again_counts_all_it_left_out():
    messages = latest_round()
    _fit("sys", messages, None, _size("sys", messages, None) - 2_000)
    first = messages[3].text
    _fit("sys", messages, None, _size("sys", messages, None) - 1_000)
    body, whole = _shown(messages[3].text)
    assert whole == 8_000 and len(body) < len(_shown(first)[0])                # all it left out, not the last cut
    assert len(body) == len(_shown(first)[0]) - 1_000 - 80                    # cut from what it shows, by the need


# --- _fit: the latest round, a kept result ---

def kept_round(length=8_000):
    messages = [Message.user("q"), call("c1"), tool("c1", _wrap("read", "z" * length))]
    return messages, {"c1": ("R2", "read", {"q": 1}, length)}


def test_a_kept_result_becomes_its_head_and_where_to_read_on_filling_the_room():
    messages, kept = kept_round()
    over = 2_000
    limit = _size("sys", messages, None) - over
    _fit("sys", messages, None, limit, kept)
    note = len(MORE.format(ref="R2", total=8_000, upto=8_000))
    keep = 8_000 - over - note - 1
    assert messages[2].text == _wrap("read", "z" * keep + MORE.format(ref="R2", total=8_000, upto=keep))
    assert _size("sys", messages, None) <= limit


def test_a_kept_result_with_no_room_says_so():
    messages, kept = kept_round()
    _fit("sys", messages, None, 10, kept)
    assert messages[2].text == _wrap("read", FULL.format(ref="R2", total=8_000))


# --- where a page ends ---

@pytest.mark.parametrize("trailing", [True, False])
def test_2000_lines_are_one_page_and_2001_are_not(trailing):
    lines = [f"{i}" for i in range(2_000)]
    text = "\n".join(lines) + ("\n" if trailing else "")
    assert _page_end(text) == len(text)
    longer = text + ("" if trailing else "\n") + "2000" + ("\n" if trailing else "")
    assert _page_end(longer) == len("".join(f"{i}\n" for i in range(2_000)))


def test_a_page_holds_exactly_its_bytes():
    line = "y" * 99 + "\n"
    lines = line * (A.PAGE_BYTES // 100 - 1)
    text = lines + "z" * (A.PAGE_BYTES - len(lines))                        # exactly PAGE_BYTES, no line end last
    assert len(text.encode()) == A.PAGE_BYTES and _page_end(text) == len(text)
    assert _page_end(text + "z") == len(lines)                               # one byte more: after the last line end


def test_a_page_of_exactly_2000_line_pieces_ends_after_its_last_line_end():
    lines = "x" * 24 + "\n"
    text = lines * 1_999 + "y" * 5_000 + "\n" + lines                       # past the bytes, 2,000 pieces in the head
    assert _page_end(text) == len(lines) * 1_999


def test_a_page_ends_after_a_line_end_found_from_the_end_even_the_first_character():
    text = "a\nb\nc\n" + "y" * 60_000
    assert _page_end(text) == 6
    assert _page_end("\n" + "y" * 60_000) == 1
    assert _page_end("y" * 60_000) == A.PAGE_BYTES                           # no line end: mid-line


# --- the registry's byte helpers and its backstop ---

def test_fitting_counts_the_separators():
    assert fitting(["a" * 10] * 3, room=32) == 3 and fitting(["a" * 10] * 3, room=31) == 2
    assert fitting(["a" * 10] * 3, sep="--", room=34) == 3 and fitting(["a" * 10] * 3, sep="--", room=33) == 2


def test_head_keeps_what_fits_whole_and_never_cuts_inside_a_character():
    assert head("ab\ncd", 5) == "ab\ncd"
    assert head("é" * 10, 5) == "éé"
    assert head("abcdef\ngh" + "x" * 20, 10) == "abcdef\n"                  # a line end in the second half
    assert head("ab\n" + "x" * 30, 10) == "ab\nxxxxxxx"                      # in the first half: mid-line


def test_the_backstop_leaves_a_tool_within_the_notes_room(monkeypatch):
    monkeypatch.setattr(registry, "COLLECT_BYTES", 1_000)
    exact = "r" * (1_000 + registry.NOTES_BYTES)
    r = Registry([ToolSpec("t", "", {}, "web", lambda **kw: exact)])
    assert r.run("t", {}).text == exact


def test_a_refusal_repeats_60_characters_of_an_argument_at_most():
    r = Registry([ToolSpec("count", "", {"type": "object", "properties": {"n": {"type": "integer"}}}, "files",
                           lambda **kw: "ok")])
    said = r.run("count", {"n": "x" * 500}).text
    assert said == "bad arguments for count: n must be a whole number (got '" + "x" * 58 + "…)"


# --- the model's gateway as _chat and _fold see it ---

import json  # noqa: E402

from peresearch.agent import FRAMING, REFUSALS, SILENT_FLOOR, SILENT_KEPT, SILENT_SHARE, SUMMARIZE, Agent  # noqa: E402
from peresearch.llm import Reply  # noqa: E402
from peresearch.protocols.chat import request_size  # noqa: E402
from tests.test_agent import FakeToolbox, write_turns  # noqa: E402


class Wire:
    """A gateway answering request n with `answer(request, n)`: a Reply, or an error it raises."""

    def __init__(self, answer, context=100_000, reserve=0):
        self.answer, self.context, self.reserve, self.requests, self.on_text = answer, context, reserve, [], None

    def context_length(self):
        return self.context

    def reply_tokens(self):
        return self.reserve

    def chat(self, request, thinking=None, stop=None):
        self.requests.append(request)
        # a loop that never ends must fail here, not fill the machine's memory (2026-10-21: a mutant's endless retry
        # of a refused summary did, with no cap on the rerun)
        assert len(self.requests) <= 50, "more than 50 requests: a loop that does not end"
        got = self.answer(request, len(self.requests))
        if isinstance(got, BaseException):
            raise got
        return got


def too_long(told=None, counted=None):
    said = (f"This model's maximum context length is {told} tokens. However, you requested 0 output tokens and your "
            f"prompt contains {counted} input tokens") if told else "the input exceeds the model's context window"
    return type("BadRequestError", (Exception,), {})(said)


def ready(a, ratio=1.0, past=0):
    """The state Agent.ask sets before its first request."""
    a._ratio, a._fold_blocked, a.cancelled, a._told = ratio, False, False, None
    a._base = a._prompt = "sys"
    a._summary, a._done, a._past, a._side = "", 0, past, [0, 0]
    a._sizes = [2] * past                                                 # each earlier turn: question, answer
    a._kept, a._scopes, a._replayed_scope = {}, {}, set()
    return a


# --- _chat: a server that cut the conversation without saying so ---

def chatted(tokens, size=3_000):
    """(the agent, the wire) after one _chat of a user message making a `size`-character request, the server
    counting `tokens(expected)` prompt tokens where expected is the agent's own estimate."""
    messages = [Message.user("x" * size)]
    expected = _size("sys", messages, None) / 1.0 + FRAMING * 2
    wire = Wire(lambda request, n: Reply("ok", prompt_tokens=tokens(expected), completion_tokens=1))
    a = ready(Agent(wire, FakeToolbox(), record=False))
    a._fold_blocked = True
    a._chat(messages, None)
    return a, wire, expected


def test_a_count_just_under_half_the_estimate_is_a_silent_cut_and_nothing_trimmable_is_not_sent_again():
    import math
    a, wire, expected = chatted(lambda e: math.floor(e * SILENT_SHARE - 1e-9))
    assert a._told == math.floor(expected * SILENT_SHARE - 1e-9) and len(wire.requests) == 1


def test_a_count_of_half_the_estimate_is_not_a_cut():
    size = next(n for n in (3_000, 3_001) if (_size("sys", [Message.user("x" * n)], None) + FRAMING * 2) % 2 == 0)
    a, _, expected = chatted(lambda e: int(e * SILENT_SHARE), size=size)
    assert expected * SILENT_SHARE == int(expected * SILENT_SHARE) and a._told is None


def test_the_smallest_count_judged_and_the_shortest_request_judged():
    a, _, _ = chatted(lambda e: SILENT_KEPT)
    assert a._told == SILENT_KEPT
    size = SILENT_FLOOR - FRAMING * 2                                       # an estimate of exactly SILENT_FLOOR
    a, _, expected = chatted(lambda e: SILENT_KEPT + 10, size=size - len("sys") - (_size("sys", [Message.user("")],
                                                                                         None) - len("sys")))
    assert expected == SILENT_FLOOR and a._told is None                    # not judged: too short to tell


def test_characters_per_token_are_learnt_only_past_the_framing():
    for tokens in (FRAMING * 2, FRAMING * 2 - 1):
        a, _, _ = chatted(lambda e: tokens)
        assert a._ratio == 1.0
    a, wire, _ = chatted(lambda e: 1_000)
    assert a._ratio == request_size(wire.requests[0]) / (1_000 - FRAMING * 2)


# --- _fold: the oldest turns summarized by the model ---

def folding(home, turns, answer, context=100_000, reserve=100, summary=""):
    wire = Wire(answer, context, reserve)
    a = Agent(wire, FakeToolbox())
    write_turns(a, turns)
    ready(a, past=len(turns))
    a._summary = summary
    return a, wire, a.history() + [Message.user("now")]


def summarize(text="tóm tắt tiếng Việt", tokens=10):
    return lambda request, n: Reply(text, prompt_tokens=tokens, completion_tokens=3)


def test_as_few_turns_as_free_the_need_are_summarized(home):
    turns = [(f"question {i}", "z" * 10_000) for i in range(3)]
    gain = _size("", [Message.user("question 0"), Message.assistant("z" * 10_000)], None) - _size("", [], None) \
        + 2 * FRAMING * 1.0
    a, wire, messages = folding(home, turns, summarize())
    assert a._fold(messages, gain - 100) is True and a._done == 1         # freed exactly: one turn
    assert "question 1" not in wire.requests[0].messages[0].text
    a, wire, messages = folding(home, turns, summarize(), summary="p" * 500)
    assert a._fold(messages, gain + 500 - 100) is True and a._done == 1   # the summary it replaces counts too


def test_nothing_to_fold_or_nothing_needed_is_said_and_written_nowhere(home):
    turns = [("question 0", "z" * 10_000)]
    a, wire, messages = folding(home, turns, summarize())
    assert a._fold(messages, 0) is False and not wire.requests
    a._past = 0
    assert a._fold(messages, 5_000) is False and not wire.requests
    a._past = 1
    assert a._fold(messages, 10**9, enough=True) is False and not wire.requests     # even all would not free it
    lines = a.history_path.read_text().splitlines()
    assert not any("summary" in json.loads(x) for x in lines)
    a, wire, messages = folding(home, turns, summarize())
    assert a._fold(messages, 1) is True                                    # one character is a need


def test_a_summary_in_parts_counts_every_requests_tokens_and_keeps_its_language(home):
    a, wire, messages = folding(home, [("question 0", "z" * 5_000)], summarize(), context=2_000)
    assert a._fold(messages, 1) is True and len(wire.requests) > 1
    n = len(wire.requests)
    assert a._side == [10 * n, 3 * n]
    first, second = (r.messages[0].text for r in wire.requests[:2])
    free = int(a._room(2_000, 2) - len(SUMMARIZE))
    assert first.startswith("User: question 0") and len(first) == free     # no summary before it; a part
    assert second.startswith("Your summary of what came before:\ntóm tắt tiếng Việt\n\n")
    assert "tiếng Việt" in a.history_path.read_text()                      # kept as written, not escaped


def test_whole_turns_are_packed_while_they_fit_and_a_turn_just_past_is_cut_with_no_empty_rest(home):
    a, _, _ = folding(home, [("q0", "z")], summarize())
    free = int(a._room(100_000, 2) - len(SUMMARIZE))
    t0, t1 = "z" * 100, None
    l0 = len(f"User: q0\nAssistant: {t0}")
    t1 = "y" * (free - 4 - l0 - len("User: q1\nAssistant: "))               # the two exactly fill the room
    a, wire, messages = folding(home, [("q0", t0), ("q1", t1)], summarize())
    assert a._fold(messages, 10**9) is True                                  # no number of turns frees it: all
    assert wire.requests[0].messages[0].text == f"User: q0\nAssistant: {t0}\n\nUser: q1\nAssistant: {t1}\n\n"
    near = "w" * (free - 1 - len("User: q0\nAssistant: "))                 # one turn of free - 1: not whole with
    a, wire, messages = folding(home, [("q0", near)], summarize())         # its separator, so cut: no rest left
    assert a._fold(messages, 10**9) is True and len(wire.requests) == 1


def test_a_refused_summary_learns_the_server_and_is_asked_again_in_order(home):
    turns = [(f"question {i}", "z" * 3_000) for i in range(3)]

    def answer(request, n):
        return too_long(50_000, 500) if n == 1 else Reply("summary", prompt_tokens=10, completion_tokens=3)
    a, wire, messages = folding(home, turns, answer)
    a._fold(messages, 10**4)
    first = request_size(wire.requests[0])
    assert a._ratio == first / (500 - FRAMING * 2) and a._told == 50_000
    assert wire.requests[1].messages[0].text.startswith("User: question 0")
    assert not any("\n\n\n" in r.messages[0].text for r in wire.requests)


def test_a_refusal_that_says_no_numbers_halves_the_characters_a_token(home):
    a, wire, messages = folding(home, [("question 0", "z" * 3_000)],
                                lambda request, n: too_long() if n == 1 else Reply("s", completion_tokens=1))
    a._fold(messages, 1_000)
    assert a._ratio == 0.5


def test_a_summary_refused_every_time_gives_up_after_its_retries(home):
    a, wire, messages = folding(home, [("question 0", "z" * 3_000)], lambda request, n: too_long(50_000, 500))
    with pytest.raises(Exception, match="maximum context length"):
        a._fold(messages, 1_000)
    assert len(wire.requests) == REFUSALS + 1


# --- edges step 3's final mutation rerun found untested ---

def test_one_character_over_in_the_latest_round_is_cut():
    messages = latest_round()
    _fit("sys", messages, None, _size("sys", messages, None) - 1)
    assert messages[3].text == "b" * (8_000 - 1 - 80) + CUT.format(n=81)


def test_a_wrapped_result_not_kept_is_cut_plainly():
    messages = [Message.user("q"), call("c1"), tool("c1", _wrap("read", "b" * 8_000))]
    _fit("sys", messages, None, _size("sys", messages, None) - 2_000)
    assert messages[2].text.endswith(CUT.format(n=_shown(messages[2].text)[1] - len(_shown(messages[2].text)[0])))


def test_a_kept_result_already_in_part_is_cut_from_what_it_shows():
    inner = "z" * 5_000 + MORE.format(ref="R2", total=8_000, upto=5_000)
    messages = [Message.user("q"), call("c1"), tool("c1", _wrap("read", inner))]
    kept = {"c1": ("R2", "read", {"q": 1}, 8_000)}
    _fit("sys", messages, None, _size("sys", messages, None) - 1_000, kept)
    note = len(MORE.format(ref="R2", total=8_000, upto=8_000))
    keep = 5_000 - 1_000 - note + (len(inner) - 5_000) - 1
    assert messages[2].text == _wrap("read", "z" * keep + MORE.format(ref="R2", total=8_000, upto=keep))


def test_two_kept_results_with_no_room_both_say_so():
    messages = [Message.user("q"), Message("assistant", (Text(""), ToolCall("c1", "read", {}), ToolCall("c2", "read", {}))),
                tool("c1", _wrap("read", "z" * 8_000)), tool("c2", _wrap("read", "y" * 6_000))]
    kept = {"c1": ("R2", "read", {}, 8_000), "c2": ("R3", "read", {}, 6_000)}
    _fit("sys", messages, None, 10, kept)
    assert messages[2].text == _wrap("read", FULL.format(ref="R2", total=8_000))
    assert messages[3].text == _wrap("read", FULL.format(ref="R3", total=6_000))


def test_the_floor_notes_earlier_rounds_and_keeps_the_latest():
    a = ready(Agent(Wire(summarize()), FakeToolbox(), record=False))
    messages = [Message.user("q"), call("c0"), tool("c0", "a" * 5_000), call("c1"), tool("c1", "b" * 5_000)]
    floor = a._floor(messages)
    assert floor[2].text == TRIMMED.format(n=5_000) and floor[4].text == "b" * 5_000


def test_a_page_counts_lines_not_words():
    text = "".join(f"w{i} a b c d e f g h i j k\n" for i in range(3_000))  # past the bytes, words in each line
    head = text.encode()[:A.PAGE_BYTES].decode()
    assert _page_end(text) == head.rfind("\n") + 1


def test_a_part_prefers_a_line_end_in_its_second_half_and_is_never_longer_than_the_result():
    _part_of = Agent._part
    note = len(MORE.format(ref="R1", total=1_131, upto=1_131))
    text = "a" * 120 + "\n" + "b" * 1_010                                  # a line end at 120 of a 300 head
    shown = _part_of("R1", text, 0, 300 + note)                              # not in the second half: mid-line
    assert shown == text[:300] + MORE.format(ref="R1", total=1_131, upto=300)
    assert _part_of("R1", "x" * 50, 0, 10) == "x" * 50                       # nothing fits, yet no longer note


def test_folding_frees_the_need_with_its_frame_by_the_characters_a_token(home):
    turns = [(f"question {i}", "z" * 10_000) for i in range(3)]
    first = [Message.user("question 0"), Message.assistant("z" * 10_000)]
    gain = _size("", first, None) - _size("", [], None) + 2 * FRAMING * 2.0     # as a request measures it
    longest = 100 * 2.0
    a, wire, messages = folding(home, turns, summarize())
    a._ratio = 2.0
    assert a._fold(messages, gain - longest) is True and a._done == 1       # ratio 2: the frame is 32, not 8
    a, wire, messages = folding(home, turns, summarize())
    a._ratio = 2.0
    assert a._fold(messages, gain - longest + 1) is True and a._done == 2   # one more: two turns


def test_a_long_summary_alone_never_counts_as_folding_a_turn(home):
    a, wire, messages = folding(home, [("question 0", "z" * 10_000)], summarize(), summary="p" * 5_000)
    assert a._fold(messages, 1) is True and a._done == 1


def test_no_room_for_a_summary_request_is_said_false(home):
    a, wire, messages = folding(home, [("question 0", "z" * 10_000)], summarize(), context=150, reserve=0)
    assert a._fold(messages, 1) is False and not wire.requests


def test_a_turn_one_short_of_the_room_is_sent_without_a_separator(home):
    a, _, _ = folding(home, [("q0", "z")], summarize())
    free = int(a._room(100_000, 2) - len(SUMMARIZE))
    text = "w" * (free - 1 - len("User: q0\nAssistant: "))
    a, wire, messages = folding(home, [("q0", text)], summarize())
    a._fold(messages, 10**9)
    assert wire.requests[0].messages[0].text == f"User: q0\nAssistant: {text}"


def test_a_refused_summary_asks_again_from_its_first_turn(home):
    turns = [(f"question {i}", "z" * 1_500) for i in range(3)]

    def answer(request, n):
        return too_long(50_000, 500) if n == 1 else Reply("summary", prompt_tokens=10, completion_tokens=3)
    a, wire, messages = folding(home, turns, answer, context=4_000)
    assert "question 2" not in wire.requests[0].messages[0].text if a._fold(messages, 10**9) else False
    assert wire.requests[1].messages[0].text.startswith("User: question 0")


def test_no_trim_follows_the_last_resend_to_a_server_that_always_cuts(home):
    messages = [Message.user("q"), call("c0"), tool("c0", "a" * 5_000), call("c1"), tool("c1", "b" * 9_000)]
    wire = Wire(lambda request, n: Reply("ok", prompt_tokens=SILENT_KEPT, completion_tokens=1))
    a = ready(Agent(wire, FakeToolbox(), record=False))
    a._fold_blocked = True
    a._chat(messages, None)
    assert request_size(wire.requests[-1]) == _size("sys", messages, None)  # the messages are what was last sent


def test_an_earlier_result_as_long_as_its_note_is_left_as_it_is_while_fitting():
    equal = next(n for n in range(50, 400) if len(TRIMMED.format(n=n)) == n)
    messages = [Message.user("q"), call("c0"), tool("c0", "y" * equal), call("c1"), tool("c1", "small")]
    _fit("sys", messages, None, _size("sys", messages, None) - 1)
    assert messages[2].text == "y" * equal


def test_what_a_kept_result_frees_is_counted_before_the_next_is_cut():
    messages = [Message.user("q"), Message("assistant", (Text(""), ToolCall("c1", "read", {}), ToolCall("c2", "read", {}))),
                tool("c1", _wrap("read", "z" * 8_000)), tool("c2", _wrap("read", "y" * 6_000))]
    kept = {"c1": ("R2", "read", {}, 8_000), "c2": ("R3", "read", {}, 6_000)}
    _fit("sys", messages, None, _size("sys", messages, None) - 8_100, kept)   # more than the first can free
    assert messages[2].text == _wrap("read", FULL.format(ref="R2", total=8_000))
    assert messages[3].text != _wrap("read", "y" * 6_000) and "recall(\"R3\", from=" in messages[3].text
