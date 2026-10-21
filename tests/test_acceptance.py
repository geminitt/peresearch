"""eval/acceptance.py: the 10 acceptance prompts through the real agent with a scripted model — asked in order, the
follow-up in its prompt's conversation, /local for its prompt, every call that reached the tools recorded; a run
stopped midway goes on where it stopped (also from another home, where the follow-up's conversation is gone); the
harness's guarantees and the owner's rules checked as written."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

from peresearch.agent import Agent
from peresearch.llm import Reply, ToolCall
from peresearch.web import Web
from tests.helpers import ScriptedLLM, notes, toolbox  # noqa: F401  (notes is a fixture)

EVAL = Path(__file__).parent.parent / "eval"
LIMITS = {"steps": 12, "web_searches": 4}


@pytest.fixture
def acceptance(tmp_path, monkeypatch, home):
    monkeypatch.setenv("PERESEARCH_RUNS", str(tmp_path / "runs"))
    spec = importlib.util.spec_from_file_location("acceptance", EVAL / "acceptance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def call(name, **args):
    return Reply("", [ToolCall(f"c-{name}-{len(args)}", name, args)])


# One scripted model for the whole run, prompt by prompt in PROMPTS' order.
SCRIPT = {
    "greeting-vi": [Reply("Xin chào! Tôi giúp bạn tìm trong ghi chú của bạn.")],
    "greeting-en": [Reply(""), Reply("Hello! I search your files and the web.")],     # empty: asked again
    "bpe-notes": [call("search_notes", query="SuperBPE"), Reply("SuperBPE adds a second stage [N1].")],
    "where-follow-up": [Reply("Trong file bpe.md, dòng 5 [N1].")],
    "tokenization-files": [call("overview", query="tokenization"), Reply("Your file bpe.md covers it [N1].")],
    "superbpe-exact": [call("grep", pattern="SuperBPE"), call("grep", pattern="SuperBPE"),     # the repeat: refused
                       Reply("bpe.md dòng 5 có cụm SuperBPE [N1].")],
    "nanovitok-folders": [call("tree"), Reply("Project có thư mục notes.")],
    "superbpe-paper": [call("web_search", query="SuperBPE"),                                     # the web first: runs
                       call("search_notes", query="SuperBPE"), call("web_search", query="SuperBPE paper"),
                       Reply("SuperBPE adds a stage [N1]; the paper [W1].")],
    "vllm-latest": [call("web_search", query="vLLM latest release"), Reply("vLLM's latest [W1].")],
    "vllm-latest-local": [call("web_search", query="vLLM"), Reply("Ghi chú của bạn không nói về phiên bản vLLM.")],
}


def scripted(acceptance, ids):
    return [r for pid, _, _ in acceptance.PROMPTS if pid in ids for r in SCRIPT[pid]]


class Interrupted(ScriptedLLM):
    """Ctrl+C on the model call after the script runs out (a BaseException, as the real one)."""

    def answer(self, *a, **kw):
        if not self.replies:
            raise KeyboardInterrupt
        return super().answer(*a, **kw)


def factory(home, notes, llm, made):
    def make(on_event):
        a = Agent(llm, toolbox(home, notes), on_event=on_event)
        made.append(a)
        return a
    return make


def searches(monkeypatch):
    count = [0]
    real = Web.search

    def search(self, *a, **kw):
        count[0] += 1
        return real(self, *a, **kw)
    monkeypatch.setattr(Web, "search", search)
    return lambda: count[0]


def test_the_prompts_run_in_order_and_record_what_reached_the_tools(acceptance, home, notes, monkeypatch):
    made, outside = [], searches(monkeypatch)
    llm = ScriptedLLM(*scripted(acceptance, SCRIPT))
    path = acceptance.run("t", factory(home, notes, llm, made), {"model": "m"}, outside=outside, say=lambda *a: None)
    recs = acceptance.done(path)
    assert [r["id"] for r in map(json.loads, path.read_text().splitlines())] == [p for p, _, _ in acceptance.PROMPTS]
    assert len(made) == 1 and not llm.replies                              # one agent; every reply used
    assert recs["where-follow-up"]["conversation"] == recs["bpe-notes"]["conversation"]
    assert recs["where-follow-up"]["earlier_messages"] == 2                 # bpe-notes' question and answer
    assert len({r["conversation"] for r in recs.values()}) == 9           # a fresh one for every other prompt
    assert [r["earlier_messages"] for k, r in recs.items() if k != "where-follow-up"] == [0] * 9
    assert {k: [c["tool"] for c in r["calls"]] for k, r in recs.items()} == {
        "greeting-vi": [], "greeting-en": [], "bpe-notes": ["search_notes"], "where-follow-up": [],
        "tokenization-files": ["overview"], "superbpe-exact": ["grep"], "nanovitok-folders": ["tree"],
        "superbpe-paper": ["web_search", "search_notes", "web_search"],
        "vllm-latest": ["web_search"], "vllm-latest-local": ["web_search"]}
    assert not recs["superbpe-paper"]["calls"][0]["result"].startswith("refused:")     # no gate before the web
    assert recs["superbpe-paper"]["calls"][0]["args"] == {"query": "SuperBPE"}
    assert recs["vllm-latest-local"]["calls"][0]["result"] == ("refused: the user turned the web off for this "
                                                                "session (local only)")
    assert not made[0].toolbox.local_only                                  # the last prompt's /local is its own
    assert recs["superbpe-exact"]["loop_refused"] == {"grep": 1}
    assert [k for k, r in recs.items() if r["loop_refused"]] == ["superbpe-exact"]
    assert [r["asked_again"] for r in recs.values()] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert {k: r["outside"] for k, r in recs.items() if r["outside"]} == {"superbpe-paper": 2, "vllm-latest": 1}
    assert [r["language"] for r in recs.values()] == ["vi", "en", "en", "vi", "en", "vi", "vi", "en", "en", "vi"]
    assert {k: Path(v).name for k, v in recs["bpe-notes"]["sources"].items()} == {
        "N1": "bpe.md:L1-5", "N2": "food.md:L1-3", "N3": "trap.md:L1-3"}            # where each source is
    assert recs["where-follow-up"]["check"] == {"unknown_ids": [], "unsupported_quotes": 0, "uncited": False}
    assert recs["where-follow-up"]["sources"]["N1"] == recs["bpe-notes"]["sources"]["N1"]
    assert recs["greeting-en"]["answer"] == "Hello! I search your files and the web."
    assert all(r["stopped"] == "" and r["steps"] >= 1 and r["seconds"] >= 0 for r in recs.values())
    assert {k: acceptance.harness(r, LIMITS) for k, r in recs.items()} == {k: [] for k in recs}
    assert {k: acceptance.rules(r) for k, r in recs.items()} == {
        "greeting-vi": {"no tool": True, "in Vietnamese": True},
        "greeting-en": {"no tool": True, "in English": True},
        "bpe-notes": {"search_notes ran": True, "cites [N…]": True, "no web": True},
        "where-follow-up": {"names a file and a line": True, "no web": True},
        "tokenization-files": {"overview ran": True, "in English": True},
        "superbpe-exact": {"grep ran": True, "names a file and a line": True},
        "nanovitok-folders": {"tree ran": True},
        "superbpe-paper": {"files first": False, "web or papers ran": True, "cites": True},
        "vllm-latest": {"files first": False, "web ran": True, "cites [W…]": True},
        "vllm-latest-local": {"no web tool called": False, "no web cited": True}}
    table = re.sub(r" \| [\d.]+ \|$", " |", acceptance.report(path, LIMITS), flags=re.M)
    assert table.splitlines()[2:] == [
        "| greeting-vi | ok | 2/2 | - | 1 |", "| greeting-en | ok | 2/2 | - | 2 |",
        "| bpe-notes | ok | 3/3 | search_notes | 2 |", "| where-follow-up | ok | 2/2 | - | 1 |",
        "| tokenization-files | ok | 2/2 | overview | 2 |", "| superbpe-exact | ok | 2/2 | grep | 3 |",
        "| nanovitok-folders | ok | 1/1 | tree | 2 |",
        "| superbpe-paper | ok | 2/3 (missed: files first) | web_search, search_notes, web_search | 4 |",
        "| vllm-latest | ok | 2/3 (missed: files first) | web_search | 2 |",
        "| vllm-latest-local | ok | 1/2 (missed: no web tool called) | - | 2 |"]


def test_a_run_stopped_midway_goes_on_where_it_stopped(acceptance, home, notes, tmp_path, monkeypatch):
    """Stopped on the follow-up's model call; a torn line left; the rerun is in another home, where the follow-up's
    conversation is gone: its prompt is asked again first, then the rest, each once."""
    first = scripted(acceptance, ("greeting-vi", "greeting-en", "bpe-notes"))
    with pytest.raises(KeyboardInterrupt):
        acceptance.run("t", factory(home, notes, Interrupted(*first), []), {"model": "m"}, say=lambda *a: None)
    path = acceptance.checkpoint("t", {"model": "m"})
    assert list(acceptance.done(path)) == ["greeting-vi", "greeting-en", "bpe-notes"]
    with open(path, "a") as f:
        f.write('{"id": "where-follow-up", "ans')                          # a power cut mid-line
    other = tmp_path / "other-home"
    monkeypatch.setenv("PERESEARCH_HOME", str(other))
    rest = scripted(acceptance, ("bpe-notes", "where-follow-up", "tokenization-files", "superbpe-exact",
                                 "nanovitok-folders", "superbpe-paper", "vllm-latest", "vllm-latest-local"))
    llm, made = ScriptedLLM(*rest), []
    acceptance.run("t", factory(other, notes, llm, made), {"model": "m"}, say=lambda *a: None)
    lines = path.read_text().splitlines()
    ids = [json.loads(line)["id"] for line in lines if line.endswith("}")]
    assert ids == ["greeting-vi", "greeting-en", "bpe-notes", "bpe-notes", "where-follow-up", "tokenization-files",
                   "superbpe-exact", "nanovitok-folders", "superbpe-paper", "vllm-latest", "vllm-latest-local"]
    recs = acceptance.done(path)
    assert recs["where-follow-up"]["conversation"] == recs["bpe-notes"]["conversation"]
    assert recs["where-follow-up"]["earlier_messages"] == 2 and not llm.replies
    assert acceptance.checkpoint("t", {"model": "other"}) != path              # another model: a run of its own


def record(**kw):
    base = {"id": "bpe-notes", "mode": "new", "answer": "An answer [N1].", "stopped": "", "steps": 2, "calls": [],
            "earlier_messages": 0, "outside": 0, "language": "en"}
    return {**base, **kw}


def ran_call(tool, result="ok"):
    return {"tool": tool, "args": {}, "result": result}


@pytest.mark.parametrize("rec, said", [
    (record(), []),
    (record(error="APIConnectionError: refused", calls=[]), ["the question failed: APIConnectionError: refused"]),
    (record(stopped="cancelled"), ["no answer came back (stopped 'cancelled')"]),
    (record(answer="  "), ["no answer came back (stopped '')"]),
    (record(stopped="steps"), []),
    (record(calls=[ran_call("web_search")]), []),                       # the web first: the model's rule, not a gate
    (record(mode="local", calls=[ran_call("fetch")]), ["a web tool ran under /local"]),
    (record(mode="local", calls=[ran_call("web_search", "refused: local only")]), []),
    (record(mode="local", outside=2), ["2 requests left the machine under /local"]),
    (record(mode="new", outside=2), []),
    (record(calls=[ran_call("web_search")] * 5),
     ["5 web searches for a limit of 4"]),
    (record(calls=[ran_call("web_search")] * 4 + [ran_call("fetch")] * 3), []),
    (record(steps=14), []),
    (record(steps=15), ["15 model calls for a limit of 12"]),
    (record(mode="after:bpe-notes", earlier_messages=0), ["the follow-up was sent without its earlier turn"]),
    (record(mode="after:bpe-notes", earlier_messages=1), ["the follow-up was sent without its earlier turn"]),
    (record(mode="after:bpe-notes", earlier_messages=2), []),
])
def test_the_harness_checks_say_each_broken_guarantee(acceptance, rec, said):
    assert acceptance.harness(rec, LIMITS) == said


def test_no_limit_set_is_never_broken(acceptance):
    rec = record(steps=500, calls=[ran_call("web_search")] * 40)
    assert acceptance.harness(rec, {"steps": None, "web_searches": None}) == []


def test_a_failed_question_misses_every_rule_and_is_failed_in_the_table(acceptance, tmp_path):
    rec = {"id": "bpe-notes", "prompt": "…", "mode": "new", "conversation": "x", "earlier_messages": 0,
           "error": "RuntimeError: boom", "calls": [], "seconds": 1.0, "outside": None}
    assert acceptance.rules(rec) == {"answered": False}
    path = tmp_path / "r.jsonl"
    acceptance.append(path, rec)
    rows = acceptance.report(path, LIMITS).splitlines()
    assert rows[2] == "| greeting-vi | not run | | | | |"
    assert rows[4] == "| bpe-notes | FAILED: the question failed: RuntimeError: boom | 0/1 (missed: answered) | - | - | 1.0 |"


@pytest.mark.parametrize("text, lang", [
    ("Xin chào, tôi có thể giúp gì?", "vi"),
    ("Hello, I can help.", "en"),
    ("BPE là thuật toán gộp các cặp ký tự thường gặp [N1].", "vi"),
    ("See `src/peresearch/agent.py` and ~/notes/bpe.md [N1, N2]: tokens.", "en"),
    ("Theo `tokenizer.json` và file notes/bpe.md, dòng 5 [N1].", "vi"),
    ("这是一个测试。", "other"),
    ("12 34 [N1]", "other"),
    ("", "other"),
])
def test_the_language_of_an_answer_leaves_code_paths_and_citations_out(acceptance, text, lang):
    assert acceptance.language(text) == lang


@pytest.mark.parametrize("text, named", [
    ("Trong file bpe.md, dòng 5 [N1].", True),
    ("notes/bpe.md:5", True),
    ("`~/projects/nanovitok/README.md` L12–L20", True),
    ("It is in bpe.md, lines 4-6.", True),
    ("Trong file bpe.md [N1].", False),
    ("dòng 5 của ghi chú", False),
])
def test_a_place_names_a_file_and_a_line(acceptance, text, named):
    assert bool(acceptance._PLACE.search(text)) == named


@pytest.mark.parametrize("rec, met", [
    (record(id="greeting-vi", language="vi", calls=[ran_call("tree", "refused: outside the declared folders")]),
     {"no tool": False, "in Vietnamese": True}),                          # a refused call is still a call made
    (record(id="superbpe-paper", answer="It [N1][W1].",
            calls=[ran_call("read", "refused: outside the declared folders"), ran_call("web_search")]),
     {"files first": True, "web or papers ran": True, "cites": True}),
    (record(id="superbpe-paper", answer="From memory.", calls=[ran_call("search_notes")]),
     {"files first": True, "web or papers ran": False, "cites": False}),
])
def test_the_owners_rules_count_the_calls_made_refused_or_not(acceptance, rec, met):
    assert acceptance.rules(rec) == met


class FailsOnce(ScriptedLLM):
    """The first model call fails as a server can (a broken connection): the other prompts still run."""

    def answer(self, *a, **kw):
        if not getattr(self, "failed", False):
            self.failed = True
            raise RuntimeError("connection reset")
        return super().answer(*a, **kw)


def test_a_question_that_fails_is_recorded_and_the_run_goes_on(acceptance, home, notes):
    llm = FailsOnce(*scripted(acceptance, [p for p in SCRIPT if p != "greeting-vi"]))
    path = acceptance.run("t", factory(home, notes, llm, []), {"model": "m"}, say=lambda *a: None)
    recs = acceptance.done(path)
    assert len(recs) == 10 and not llm.replies
    assert {k: v for k, v in recs["greeting-vi"].items() if k not in ("conversation", "seconds")} == {
        "id": "greeting-vi", "prompt": "Xin chào", "mode": "new", "earlier_messages": 0,
        "error": "RuntimeError: connection reset", "calls": [], "outside": None}
    assert acceptance.harness(recs["greeting-vi"], LIMITS) == ["the question failed: RuntimeError: connection reset"]
    assert [k for k, r in recs.items() if acceptance.harness(r, LIMITS)] == ["greeting-vi"]
