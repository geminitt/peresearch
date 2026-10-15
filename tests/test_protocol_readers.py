"""Details of the three older stream readers and the shared reply assembly, pinned when they moved into protocols/
(model gateway step 2, 2026-10-15) and mutation showed them loosely held."""

from types import SimpleNamespace as NS

import pytest

from peresearch.protocols import Interrupted, StreamCut, _assembled
from peresearch.protocols.anthropic import _read_anthropic
from peresearch.protocols.chat import _read_stream
from peresearch.protocols.responses import _read_responses
from tests.test_llm_anthropic import Block, reply


def test_a_reply_joins_its_pieces_and_names_a_call_that_has_no_id():
    got = _assembled(["a", "b"], ["x", "y"], {0: {"id": None, "name": "t", "args": "{}"}, 1: {"id": "c", "name": "u",
                                                                                           "args": ""}},
                     NS(prompt_tokens=3, completion_tokens=4), "stop", ["no", "pe"])
    assert (got["text"], got["reasoning"], got["refusal"]) == ("ab", "xy", "nope")
    assert [c["id"] for c in got["calls"]] == ["call-0", "c"] and (got["prompt_tokens"], got["completion_tokens"]) == (3, 4)
    assert "refusal" not in _assembled([], [], {}, None, "stop") and _assembled([], [], {}, None, "stop")["prompt_tokens"] == 0


def chunk(**delta):
    return NS(choices=[NS(delta=NS(**delta), finish_reason=None)], usage=None)


def test_the_chat_reader_says_why_it_stopped_and_keeps_a_call_given_without_a_function():
    with pytest.raises(Interrupted, match="^stopped by the user$"):
        _read_stream(iter([chunk(content="x")]), lambda: True)
    got = _read_stream(iter([NS(choices=[NS(delta=NS(tool_calls=[NS(index=0, id="c1", function=None)]),
                                            finish_reason="tool_calls")], usage=NS(prompt_tokens=2, completion_tokens=1))]),
                       None)
    assert got["calls"] == [{"id": "c1", "name": "", "args": ""}] and got["prompt_tokens"] == 2
    got = _read_stream(iter([NS(choices=[NS(delta=NS(content="a"), finish_reason="stop")], usage=NS(
        prompt_tokens=5, completion_tokens=6)), NS(choices=[], usage=None)]), None)
    assert (got["prompt_tokens"], got["completion_tokens"]) == (5, 6)              # a last chunk without usage keeps it


def test_an_interrupted_or_failed_stream_carries_its_usage_and_its_end():
    class Failing:
        def __iter__(self):
            yield NS(type="response.output_text.delta", delta="half")
            yield NS(type="response.completed", response=NS(usage=NS(input_tokens=8, output_tokens=2),
                                                            incomplete_details=None))
            raise StreamCut("cut")
    with pytest.raises(StreamCut) as raised:
        _read_responses(Failing(), None)
    assert (raised.value.partial["text"], raised.value.partial["prompt_tokens"], raised.value.partial["finish"]) == \
        ("half", 8, "stop")


def test_the_messages_reader_counts_cached_input_and_keeps_what_came_before_a_failure():
    final = reply([Block(type="text", text="hi")], usage=(100, 20, 30, 7))
    got = _read_anthropic(final, None)
    assert (got["prompt_tokens"], got["completion_tokens"]) == (137, 20)           # input, cache read, cache write

    class Failing:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            yield NS(type="text", text="half")
            raise StreamCut("cut")
    with pytest.raises(StreamCut) as raised:
        _read_anthropic(Failing(), None)
    assert raised.value.partial["text"] == "half" and raised.value.partial["finish"] == ""
    refused = reply([], stop="refusal", details=NS(category="x", explanation="declined"))
    assert _read_anthropic(refused, None)["refusal"] == "declined"
    assert _read_anthropic(reply([], stop="refusal", details=None), None)["refusal"] == "the model declined to answer"
    assert _read_anthropic(reply([Block(type="text", text="a")], stop="pause_turn"), None)["finish"] == "stop"


def test_cache_counts_the_stream_does_not_give_are_zero_not_missing():
    class Stream:
        def __init__(self, usage):
            self.usage = usage

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return iter([])

        def get_final_message(self):
            return NS(content=[], stop_reason="end_turn", usage=self.usage, stop_details=None)
    bare = _read_anthropic(Stream(NS(input_tokens=5, output_tokens=2)), None)
    assert (bare["prompt_tokens"], bare["completion_tokens"]) == (5, 2)             # no cache fields at all
    nones = _read_anthropic(Stream(NS(input_tokens=None, output_tokens=None, cache_read_input_tokens=None,
                                      cache_creation_input_tokens=None)), None)
    assert (nones["prompt_tokens"], nones["completion_tokens"]) == (0, 0)


def test_an_error_event_with_no_message_and_no_failed_response_says_so():
    from peresearch.protocols.responses import ResponsesFailed
    for event in (NS(type="error", message=None), NS(type="response.failed", response=None),
                  NS(type="response.failed", response=NS(error=None))):
        with pytest.raises(ResponsesFailed, match="^the Responses API reported a failure$"):
            _read_responses(iter([event]), None)
    with pytest.raises(ResponsesFailed, match="^boom$"):
        _read_responses(iter([NS(type="response.failed", response=NS(error="boom"))]), None)
