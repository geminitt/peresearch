"""The owner's decisions of 2026-10-14 on what the agent may do in a question: no limit on steps, web searches or
pages unless the user sets one; the user's files before the web is the model's instruction, not a gate (`gaps` is
gone); fetch opens a URL the user wrote in the question as it opens one a search returned."""

from peresearch.agent import SYSTEM, Agent, Limits, _wrap
from peresearch.llm import Reply, ToolCall
from peresearch.tools import Toolbox
from tests.test_agent import FakeModel, FakeToolbox
from tests.test_tools import FakePapers, FakeWeb, FakeWiki


def test_no_limit_is_set_by_default_on_steps_searches_or_pages(home):
    script = [Reply("", [ToolCall(f"c{i}", "web_search", {"q": i})]) for i in range(15)]
    script += [Reply("", [ToolCall(f"f{i}", "fetch", {"url": f"https://u/{i}"})]) for i in range(8)]
    tb = FakeToolbox()
    ans = Agent(FakeModel(script + [Reply("done")]), tb, record=False).ask("q")
    assert ans.stopped == "" and ans.text == "done" and ans.steps == 24
    assert [n for n, _ in tb.ran] == ["web_search"] * 15 + ["fetch"] * 8
    assert (Limits().steps, Limits().web_searches, Limits().fetches) == (None, None, None)


def test_a_limit_the_user_sets_still_holds(home):
    script = [Reply("", [ToolCall(f"c{i}", "tree", {"q": i})]) for i in range(5)]
    tb = FakeToolbox()
    ans = Agent(FakeModel(script), tb, limits=Limits(steps=3), record=False).ask("q")
    assert ans.stopped == "steps" and len(tb.ran) == 3                  # three rounds ran, then it must answer


def test_the_web_is_searched_without_recording_anything_first():
    web = FakeWeb()
    tb = Toolbox(None, web, roots=[], papers=FakePapers([]), wiki=FakeWiki([]))
    assert "gaps" not in [t.name for t in tb.schemas()] and "gaps" not in Toolbox.SPECS
    assert tb.call("web_search", {"query": "x", "n": 1}).startswith("(web search: fake)") and web.searched == ["x"]
    assert "gaps" not in SYSTEM
    assert "look at the user's files first, then the web" in SYSTEM


def test_fetch_opens_a_url_the_user_wrote_in_the_question_and_no_other():
    web = FakeWeb()
    web.pages["https://example.org/a"] = "page a"
    tb = Toolbox(None, web, roots=[])
    tb.begin("Đọc giúp tôi https://example.org/a, rồi tóm tắt.")
    assert tb.call("fetch", {"url": "https://example.org/a"}) == "[W1] https://example.org/a\npage a"
    assert tb.sources.items["W1"].title == "https://example.org/a" and tb.sources.items["W1"].text == "page a"
    assert tb.call("fetch", {"url": "https://example.org/b"}).startswith("refused: fetch only opens URLs")
    tb.sources.begin_question()
    tb.begin("another question")                                        # a URL of an earlier question: no more
    assert tb.call("fetch", {"url": "https://example.org/a"}).startswith("refused: fetch only opens URLs")


def test_the_agent_lets_fetch_open_the_urls_of_its_question(home):
    web = FakeWeb()
    web.pages["https://example.org/a"] = "page a"
    tb = Toolbox(None, web, roots=[])
    model = FakeModel([Reply("", [ToolCall("c1", "fetch", {"url": "https://example.org/a"})]), Reply("It says [W1].")])
    ans = Agent(model, tb, record=False).ask("What does https://example.org/a say?")
    assert ans.sources["W1"].where == "https://example.org/a" and ans.check.ok


def test_fetch_is_offered_with_the_page_reader_even_without_a_search_server():
    web = FakeWeb()
    web.searcher = None                                                 # no search server set
    names = [t.name for t in Toolbox(None, web, roots=[]).schemas()]
    assert "fetch" in names and "web_search" not in names
    assert "fetch" not in [t.name for t in Toolbox(None, None, roots=[], papers=FakePapers([])).schemas()]


def test_a_search_limit_the_user_sets_is_shared_by_the_three_searches_and_renewed_each_question(home):
    from tests.test_tools import paper
    web = FakeWeb()
    tb = Toolbox(None, web, roots=[], papers=FakePapers([paper()]), wiki=FakeWiki([]))
    script = [Reply("", [ToolCall("a", "web_search", {"query": "x"})]),
              Reply("", [ToolCall("b", "search_papers", {"query": "x"})]), Reply("done")]
    a = Agent(FakeModel(script), tb, limits=Limits(web_searches=1), record=False)
    first = a.ask("q1")
    assert web.searched == ["x"] and tb.papers.asked == []               # one search in all: the papers refused
    a.llm.replies = [Reply("", [ToolCall("c", "search_papers", {"query": "y"})]), Reply("done")]
    a.ask("q2")
    assert len(tb.papers.asked) == 1 and first.stopped == ""             # a new question: its own search


def test_a_rate_limit_is_waited_within_the_questions_time_and_not_past_it(home, monkeypatch):
    from peresearch import registry
    from peresearch.registry import RateLimited
    slept = []
    monkeypatch.setattr(registry.time, "sleep", slept.append)
    tb = FakeToolbox()
    told = iter([RateLimited("Semantic Scholar", 2.0), None, RateLimited("Semantic Scholar", 50.0)])

    def call(name, args):
        error = next(told)
        if error:
            raise error
        return "found"
    tb.call = call
    script = [Reply("", [ToolCall("a", "tree", {})]), Reply("done")]
    ans = Agent(FakeModel(script), tb, limits=Limits(seconds=30), record=False).ask("q")
    assert slept == [2.0] and ans.stopped == ""                           # 2 s fit the question's 30 s: waited
    a = Agent(FakeModel([Reply("", [ToolCall("b", "tree", {})]), Reply("done")]), tb, limits=Limits(seconds=30),
              record=False)
    model = a.llm
    a.ask("q2")
    assert slept == [2.0]                                                 # 50 s: past the question's time
    told_model = [m["content"] for m in model.seen[-1]["messages"] if m["role"] == "tool"][-1]
    assert "rate limited by Semantic Scholar" in told_model


def test_an_empty_recall_ref_is_said_and_a_long_call_is_shown_in_120_characters(home):
    events = []
    script = [Reply("", [ToolCall("r", "recall", {"ref": ""}), ToolCall("t", "tree", {"path": "p" * 300})]),
              Reply("done")]
    model = FakeModel(script)
    Agent(model, FakeToolbox(), record=False, on_event=lambda k, d: events.append((k, d))).ask("q")
    told = [m["content"] for m in model.seen[-1]["messages"] if m["role"] == "tool"]
    assert told[0] == _wrap("recall", "recall needs a ref such as R1: each trimmed result's note gives it")
    assert ("tool", "tree(" + ('"' + "p" * 300 + '"')[:120] + ")") in events


def test_a_page_the_user_named_that_cannot_be_read_is_said_without_a_source_and_urls_end_before_punctuation():
    web = FakeWeb()
    web.pages["https://example.org/BOX"] = ""
    tb = Toolbox(None, web, roots=[])
    tb.begin("See https://example.org/BOX. Thanks!")
    assert tb.asked_urls == {"https://example.org/BOX"}
    assert tb.call("fetch", {"url": "https://example.org/BOX"}) == "https://example.org/BOX\nthe page could not be read"
    assert tb.sources.items == {}
