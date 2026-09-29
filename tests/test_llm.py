"""The model client: tool calls left as text, a cold or limited server, and the daily spending cap."""

import threading
import time
from types import SimpleNamespace

import pytest

from peresearch.agent import check
from peresearch.llm import LLM, parse_text_calls
from tests.helpers import streamed


def client_of(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_a_streamed_answer_is_put_back_together():
    """Text, reasoning, tool calls whose arguments arrive in pieces, and the token counts of the last chunk."""
    sent = []

    def create(**kw):
        sent.append(kw)
        return streamed(content="Looking.", reasoning="first the notes",
                        tool_calls=[("grep", '{"pattern": "SuperBPE"}'), ("read", '{"path": "a.md", "start": 3}')],
                        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=34))
    r = LLM(client=client_of(create), model="m").chat([{"role": "user", "content": "x"}])
    assert (r.text, r.reasoning) == ("Looking.", "first the notes")
    assert [(c.name, c.args) for c in r.calls] == [("grep", {"pattern": "SuperBPE"}), ("read", {"path": "a.md", "start": 3})]
    assert (r.prompt_tokens, r.completion_tokens) == (120, 34)
    assert sent[0]["stream"] is True and sent[0]["stream_options"] == {"include_usage": True}


def test_tool_calls_left_as_text_are_recovered():
    xml = "<tool_call>\n<function=web_search>\n<parameter=query>\nbpe history\n</parameter>\n</function>\n</tool_call>"
    assert parse_text_calls(xml)[0].name == "web_search" and parse_text_calls(xml)[0].args == {"query": "bpe history"}
    hermes = '<tool_call>{"name": "read", "arguments": {"path": "a.md", "start": 3}}</tool_call>'
    assert parse_text_calls(hermes)[0].args == {"path": "a.md", "start": 3}
    llm = LLM(client=client_of(lambda **kw: streamed(reasoning="thinking… " + xml)), model="m")
    assert llm.chat([{"role": "user", "content": "x"}]).calls[0].name == "web_search"


def test_a_cold_model_is_waited_for_then_other_errors_surface():
    class APIConnectionError(Exception):
        pass
    calls, waits = [], []

    def create(**kw):
        calls.append(1)
        if len(calls) < 3:
            raise APIConnectionError("starting")
        return streamed(content="ok")
    llm = LLM(client=client_of(create), model="m", wait=0, on_wait=waits.append)
    assert llm.chat([{"role": "user", "content": "x"}]).text == "ok" and len(waits) == 2

    def broken(**kw):
        raise ValueError("bad request")
    llm.client.chat.completions.create = broken
    with pytest.raises(ValueError):
        llm.chat([{"role": "user", "content": "x"}])


def test_a_server_without_template_switches_is_asked_without_them():
    class BadRequestError(Exception):
        pass
    seen = []

    def create(**kw):
        seen.append("extra_body" in kw)
        if "extra_body" in kw:
            raise BadRequestError("unknown field chat_template_kwargs")
        return streamed(content="ok")
    llm = LLM(client=client_of(create), model="m", thinking=False)
    assert llm.chat([{"role": "user", "content": "x"}]).text == "ok" and seen == [True, False]
    llm.chat([{"role": "user", "content": "y"}])
    assert seen[-1] is False                                        # remembered for the session


def test_the_daily_budget_counts_container_time_and_stops_the_model(home):
    from peresearch.llm import Budget, BudgetExceeded
    now = [1_800_000_000.0]
    b = Budget(home / "usage.jsonl", usd_per_hour=3.6, cap_usd=1.0, scaledown=300, clock=lambda: now[0])
    t0 = now[0]
    b.record(t0, t0 + 60)             # container up 60 s + 300 s idle = 360 s
    b.record(t0 + 100, t0 + 160)      # overlaps: only extends to 460 s
    b.record(t0 + 2000, t0 + 2040)    # a second wake-up: 340 s more
    assert abs(b.spent() - 800 / 3600 * 3.6) < 1e-9 and b.spent() < 1.0
    b.check()
    b.record(t0 + 3000, t0 + 3300)    # 600 s more: 1400 s × $3.6/h = $1.40
    with pytest.raises(BudgetExceeded):
        b.check()
    now[0] += 86400                   # a new day starts from zero
    b.check()


def test_a_capped_client_does_not_call_the_model(home):
    from peresearch.llm import Budget, BudgetExceeded
    calls = []
    b = Budget(home / "usage.jsonl", usd_per_hour=3600, cap_usd=0.5, scaledown=0)

    def create(**kw):
        calls.append(1)
        return streamed(content="ok")
    llm = LLM(client=client_of(create), model="m", budget=b)
    b.record(0, 0)
    import time as _t
    b.record(_t.time(), _t.time() + 1)                # one second at $3600/h = $1 > $0.5
    with pytest.raises(BudgetExceeded):
        llm.chat([{"role": "user", "content": "x"}])
    assert calls == []


def test_the_budget_applies_to_a_modal_endpoint_only(monkeypatch):
    from peresearch.llm import Budget
    monkeypatch.delenv("PERESEARCH_GPU_USD_PER_HOUR", raising=False)
    assert Budget.from_settings("http://localhost:8000/v1") is None
    b = Budget.from_settings("https://ws--peresearch-llm-server.modal.run/v1")
    assert b and b.rate == 1.95 and b.cap == 1.0 and b.scaledown == 300


class BlockingClient:
    """An OpenAI-like client whose answer streams a chunk every 50 ms for 30 s, like a model writing a long answer;
    `closed` is set when the stream is closed (what makes vLLM stop generating)."""

    def __init__(self, error=None):
        self.closed, self.closed_by, self.calls, self.error = threading.Event(), None, 0, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls += 1
        if self.error:
            raise self.error
        client = self

        class Stream:
            def __iter__(self):
                end = time.time() + 30
                while time.time() < end and not client.closed.is_set():
                    time.sleep(0.05)
                    yield streamed(content="word ")[0]

            def close(self):
                client.closed_by = threading.current_thread()
                client.closed.set()
        return Stream()


def test_an_interrupt_stops_a_model_call_in_flight_and_closes_its_stream():
    """Closing the stream is what makes vLLM stop generating (0.27 s later, measured), and a Modal GPU stop being
    billed for it; it must be closed by the thread reading it (a close from another thread left vLLM generating)."""
    from peresearch.llm import Interrupted

    client = BlockingClient()
    llm = LLM(client=client, model="m")
    t0 = time.time()
    with pytest.raises(Interrupted):
        llm.chat([{"role": "user", "content": "x"}], stop=lambda: time.time() - t0 > 0.3)
    assert time.time() - t0 < 1.5
    assert client.closed.wait(1) and client.closed_by is not threading.main_thread()


def test_an_interrupt_also_ends_the_wait_between_retries():
    from peresearch.llm import Interrupted

    class APIConnectionError(Exception):
        pass
    llm = LLM(client=BlockingClient(error=APIConnectionError("down")), model="m", wait=10)
    t0 = time.time()
    with pytest.raises(Interrupted):
        llm.chat([{"role": "user", "content": "x"}], stop=lambda: time.time() - t0 > 0.3)
    assert time.time() - t0 < 1.5
