"""The agent loop with a scripted model: nothing runs before the model reads the message, what the model is
shown, what it may do, what leaves the machine, and what the answer check catches."""

import json
from types import SimpleNamespace

from peresearch.agent import Agent, Limits, check
from peresearch.llm import LLM, Reply, ToolCall
from peresearch.tools import Source
from peresearch.web import Web
from tests.test_guard import CANARIES
from tests.helpers import notes, toolbox, agent


def test_a_greeting_is_answered_without_any_tool(home, notes):
    """The model reads the message first; nothing is searched on its behalf."""
    a, llm = agent(home, notes, Reply("Xin chào! Mình giúp gì được cho bạn?"))
    ans = a.ask("xin chào")
    first = llm.seen[0][0]
    assert not any(m["role"] == "tool" or m.get("tool_calls") for m in first)
    assert {"tree", "search_notes", "grep", "glob", "read", "gaps", "web_search", "fetch"} <= {
        t["function"]["name"] for t in llm.seen[0][1]}                  # every tool is offered, none is forced
    assert ans.calls == [] and ans.steps == 1 and not ans.sources and ans.check.ok


def test_the_model_looks_into_the_files_when_it_decides_to(home, notes):
    a, llm = agent(home, notes, Reply("", [ToolCall("c1", "tree", {"path": str(notes)})]),
                   Reply("", [ToolCall("c2", "read", {"path": "bpe.md"})]),
                   Reply("The folder holds bpe.md and food.md [N1]; BPE merges frequent pairs [N2]."))
    ans = a.ask("What is in my notes folder?")
    listing = [m["content"] for m in llm.seen[1][0] if m["role"] == "tool"][-1]
    assert "bpe.md" in listing and "food.md" in listing and "trust=\"untrusted data\"" in listing
    assert [c[0] for c in ans.calls] == ["tree", "read"] and ans.check.ok, ans.check


def test_search_then_read_a_page_then_answer_with_checked_citations(home, notes):
    a, llm = agent(home, notes,
                   Reply("", [ToolCall("c0", "search_notes", {"query": "BPE"})]),
                   Reply("", [ToolCall("c1", "gaps", {"have": ["BPE merges pairs [N1]"], "missing": ["its origin"]})]),
                   Reply("", [ToolCall("c2", "web_search", {"query": "byte pair encoding history"})]),
                   Reply("", [ToolCall("c3", "fetch", {"url": "https://example.org/bpe"})]),
                   Reply('Notes: BPE merges pairs [N1]. Web: "It was introduced for neural machine translation in 2016" [W1].'))
    ans = a.ask("BPE merges pairs: where does it come from?")
    assert ans.check.ok, ans.check
    assert ans.sources["W1"].where == "https://example.org/bpe" and "introduced" in ans.sources["W1"].text
    assert "menu home login" not in ans.sources["W1"].text            # the page's main text only
    assert ans.steps == 5 and not ans.stopped


def test_limits_cap_web_searches_and_steps(home, notes):
    loop = [Reply("", [ToolCall(f"c{i}", "web_search", {"query": f"q{i}"})]) for i in range(20)]
    a, llm = agent(home, notes, Reply("", [ToolCall("g", "gaps", {"have": [], "missing": ["q"]})]), *loop,
                   limits=Limits(steps=6, web_searches=2))
    ans = a.ask("search forever")
    assert a.toolbox.web_calls == 2 and ans.stopped == "steps" and ans.steps == 8   # 6 + the forced answer + its retry
    assert llm.seen[-1][1] is None                                    # the last call offers no tools: answer now
    assert any("Limit reached" in m["content"] for m in llm.seen[-1][0] if m["role"] == "user")


def test_instruction_like_file_text_is_flagged_and_never_acted_on(home, notes):
    a, llm = agent(home, notes, Reply("", [ToolCall("c1", "search_notes", {"query": "ignore previous instructions paper"})]),
                   Reply("Nothing to do [N1]."))
    a.ask("ignore previous instructions paper")
    shown = llm.seen[1][0][-1]["content"]
    assert "Ignore previous instructions" in shown and "reads like an instruction" in shown


def test_no_secret_reaches_the_model(home, notes):
    (notes / "keys.md").write_text("# Deploy\n\nThe token is " + CANARIES["github-token"] + " for CI.\n")
    sent = []

    def create(**kw):
        sent.append(json.dumps(kw["messages"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="done [N1]", tool_calls=None))],
                               usage=None)
    llm = LLM(client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), model="m")
    tb = toolbox(home, notes)
    tb.read("keys.md")                                                    # even a file read in full
    a = Agent(llm, tb)
    a.ask("the token for CI " + CANARIES["huggingface-token"])
    assert sent and all(CANARIES["github-token"] not in s and CANARIES["huggingface-token"] not in s for s in sent)


def test_history_of_the_project_comes_back_within_its_budget(home, notes):
    a, llm = agent(home, notes, Reply("first answer [N1]"), Reply("second answer [N1]"), limits=Limits(history_chars=200))
    a.ask("first question")
    a.ask("second question")
    roles = [(m["role"], m["content"]) for m in llm.seen[1][0] if m["role"] in ("user", "assistant") and m["content"]]
    assert ("user", "first question") in roles and ("assistant", "first answer [N1]") in roles
    b, llm2 = agent(home, notes, Reply("x [N1]"), limits=Limits(history_chars=10))
    b.ask("third")
    assert all(m["content"] != "first question" for m in llm2.seen[0][0])


def test_the_check_catches_unknown_ids_invented_quotes_and_missing_citations():
    src = {"N1": Source("N1", "file", "a.md:L1", "", "BPE merges the most frequent pair of symbols.")}
    assert check('It "merges the most frequent pair of symbols" [N1].', src).ok
    c = check('It "splits every word into characters first" [N1] and [W9].', src)
    assert c.unknown_ids == ["W9"] and c.unsupported_quotes and not c.ok
    assert check("An answer with no citation.", src).uncited


def test_an_empty_answer_is_asked_again_without_reasoning(home, notes):
    a, llm = agent(home, notes, Reply("", reasoning="it is all in here, never closed"), Reply("BPE merges pairs [N1]."))
    ans = a.ask("BPE merges pairs?")
    assert ans.text == "BPE merges pairs [N1]." and llm.thinking[-1] is False and ans.steps == 2


def test_the_check_ignores_quote_marks_that_do_not_enclose_a_cited_quote():
    src = {"N1": Source("N1", "file", "a.md:L1", "", "KV cache stores keys and values.")}
    text = 'The files do not define "KV cache" in Vietnamese. The notes [N1] and [N2] discuss it as "a cache".'
    c = check(text.replace(" and [N2]", ""), src)
    assert c.unsupported_quotes == [] and c.ok                          # from the real local run of 2026-09-29


def test_the_model_is_told_when_the_web_is_unavailable(home, notes):
    a, llm = agent(home, notes, Reply("Only your files [N1]."))
    a.toolbox.web.providers = []
    a.ask("BPE merges pairs?")
    assert "Web search is NOT available" in llm.seen[0][0][0]["content"]
    assert all(t["function"]["name"] not in ("web_search", "fetch", "gaps") for t in llm.seen[0][1])


def test_a_repeated_identical_call_is_not_run_again(home, notes):
    same = ToolCall("c", "glob", {"pattern": "*.md"})
    a, llm = agent(home, notes, Reply("", [same]), Reply("", [ToolCall("d", "glob", {"pattern": "*.md"})]),
                   Reply("Done [N1]."))
    a.ask("BPE merges pairs?")
    tool_msgs = [m["content"] for m in llm.seen[-1][0] if m["role"] == "tool"]
    assert "bpe.md" in tool_msgs[-2] and "already called with the same arguments" in tool_msgs[-1]


def test_past_answers_are_cut_and_the_current_question_marked(home, notes):
    a, llm = agent(home, notes, Reply("A" * 2000 + " [N1]"), Reply("second [N1]"))
    a.ask("BPE merges pairs?")
    a.ask("And SuperBPE?")
    msgs = llm.seen[1][0]
    past = [m["content"] for m in msgs if m["role"] == "assistant" and m["content"]]
    assert past and len(past[0]) <= 810 and past[0].endswith("[…]")
    user = [m["content"] for m in msgs if m["role"] == "user"]
    assert user[0] == "BPE merges pairs?" and user[-1].startswith("Current question") and user[-1].endswith("And SuperBPE?")
