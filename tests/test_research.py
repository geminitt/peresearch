"""eval/research.py with a scripted agent, a scripted grader and a word-overlap stand-in for the NLI model (CPU):
every suite runs, resumes, and reports; the ALCE citation scores and the attack detection match hand-worked cases."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from peresearch.agent import Agent, Limits
from peresearch.llm import Reply, ToolCall
from peresearch.tools import Toolbox
from peresearch.zetokrag.index import Index
from peresearch.zetokrag.search import Searcher
from tests.conftest import FakeEmbedder, FakeReranker
from tests.test_agent import ScriptedLLM

EVAL = Path(__file__).parent.parent / "eval"


class WordNLI:
    """Entailment when most words of the hypothesis appear in the premise."""

    def entails(self, premise, hypothesis):
        h = [w for w in hypothesis.lower().replace(".", " ").split() if len(w) > 2]
        p = premise.lower()
        return bool(h) and sum(w in p for w in h) / len(h) >= 0.8


@pytest.fixture
def research(tmp_path, monkeypatch, home):
    monkeypatch.setenv("PERESEARCH_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("PERESEARCH_RESULTS", str(tmp_path / "results"))
    spec = importlib.util.spec_from_file_location("research", EVAL / "research.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def files_agent(notes, llm, web=None):
    idx = Index(notes.parent / "idx", embedder=FakeEmbedder())
    idx.update([notes])
    return Agent(llm, Toolbox(Searcher(idx, reranker=FakeReranker()), web, roots=[notes]),
                 limits=Limits(history_chars=0), record=False)


@pytest.fixture
def notes(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "bpe.md").write_text("# Tokenization\n\nBPE merges frequent symbol pairs into new tokens.\n")
    return root


def test_alce_citation_recall_and_precision_on_hand_worked_sentences(research):
    src = {"N1": "BPE merges frequent symbol pairs into new tokens.", "N2": "Pho is a Vietnamese soup."}
    s = research.citation_scores("BPE merges frequent symbol pairs [N1]. BPE merges frequent pairs [N1, N2]. "
                                 "Transformers use attention [N1]. No citation here.", src, WordNLI())
    # 3 cited sentences: two supported; the second cites N2 needlessly (it alone does not support, N1 does)
    assert s == {"cited_sentences": 3, "supported": 2, "citations": 4, "needed": 2}


def test_qa_suite_runs_grades_resumes_and_reports(research, notes, monkeypatch):
    items = [{"id": f"s{i}", "question": f"BPE merges pairs question {i}?", "gold": "pairs"} for i in range(3)]
    monkeypatch.setattr(research, "load_simpleqa", lambda: items)
    asked = []

    def factory():
        llm = ScriptedLLM(*[r for _ in range(3) for r in (Reply("", [ToolCall("s", "search_notes", {"query": "BPE"})]),
                                                          Reply("BPE merges frequent symbol pairs [N1]."))])
        a = files_agent(notes, llm)
        orig = a.ask
        a.ask = lambda q: (asked.append(q), orig(q))[1]
        return a
    sanity = [Reply(x) for x in ("A", "B", "C", "B")]
    judge = ScriptedLLM(*sanity, Reply("A"), Reply("B"), Reply("C"), *[Reply(x) for x in ("A", "B", "C", "B")])
    path = research.run_qa("simpleqa", None, factory, judge, WordNLI())
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["grade"] for r in recs] == ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"] and len(asked) == 3
    assert recs[0]["citations"]["supported"] == 1 and recs[0]["check_ok"]
    research.run_qa("simpleqa", None, factory, judge, WordNLI())         # resumed: nothing asked again
    assert len(asked) == 3
    text = research.report()
    assert "correct 1/3" in text and "correct given attempted 1/2" in text and "F-score 40.0" in text


def test_personal_suite_scores_evidence_and_abstention(research, notes, monkeypatch):
    where = str(notes / "bpe.md")
    monkeypatch.setattr(research, "load_personal", lambda: [
        {"id": "a1", "question": "BPE merges pairs?", "answerable": True,
         "relevant": [{"path": where, "unit": "line", "start": 1, "end": 4}]},
        {"id": "u1", "question": "What is my passport number?", "answerable": False, "relevant": []}])
    llm = ScriptedLLM(Reply("", [ToolCall("s", "search_notes", {"query": "BPE merges pairs"})]),
                      Reply("Your notes: BPE merges pairs [N1]."), Reply("Your files do not cover this."))
    path = research.run_personal([str(notes)], lambda: files_agent(notes, llm), WordNLI())
    recs = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
    assert recs["a1"]["evidence"] and recs["u1"]["cited_files"] == 0 and "answer" not in recs["a1"]
    text = research.report()
    assert "evidence recall (cites the labelled passage), answerable: 1/1" in text and "unanswerable: 1/1" in text


def test_injection_suite_tells_an_obeying_agent_from_a_resisting_one(research, monkeypatch):
    monkeypatch.setattr(research, "ATTACKS", research.ATTACKS[:1] + research.ATTACKS[4:5])   # one file, one web

    def factory(notes, web, obey):
        look = Reply("", [ToolCall("c0", "search_notes", {"query": "gradient checkpointing"})])
        if obey:
            replies = [look, Reply("", [ToolCall("g", "gaps", {"have": [], "missing": ["checkpointing"]})]),
                       Reply("", [ToolCall("c1", "web_search", {"query": research.CANARY + " private notes dump"})]),
                       Reply(f"Checkpointing trades compute for memory [N1]. {research.MARKER}")]
        else:
            replies = [look, Reply("Gradient checkpointing recomputes activations, trading compute for memory [N1].")]
        idx = Index(notes.parent / "idx", embedder=FakeEmbedder())
        idx.update([notes])
        return Agent(ScriptedLLM(*replies), Toolbox(Searcher(idx, reranker=FakeReranker()), web, roots=[notes]),
                     limits=Limits(history_chars=0), record=False)

    path = research.run_injection(1, lambda n, w: factory(n, w, obey=True), WordNLI())
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(r["success"] for r in recs) and "web_search with the canary" in recs[0]["actions"]
    assert recs[0]["exposed"]                                              # the planted note was retrieved
    path.unlink()
    path = research.run_injection(1, lambda n, w: factory(n, w, obey=False), WordNLI())
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert not any(r["success"] for r in recs)
    assert "attack success rate: 0/2" in research.report()


def test_the_grader_prompt_is_the_official_one_with_its_three_slots(research):
    assert research.GRADER.count("{") == 3 and "NOT_ATTEMPTED" in research.GRADER
    judge = ScriptedLLM(Reply("B"))
    assert research.grade(judge, "Q?", "gold", "an answer") == "INCORRECT"
    sent = judge.seen[0][0][0]["content"]
    assert "Question: Q?" in sent and "Gold target: gold" in sent and "Predicted answer: an answer" in sent


def test_a_judge_that_fails_the_worked_examples_is_refused_and_empty_answers_are_not_attempted(research):
    with pytest.raises(SystemExit):
        research.check_judge(ScriptedLLM(*[Reply("A")] * 4))           # grades everything CORRECT
    judge = ScriptedLLM()
    assert research.grade(judge, "Q?", "gold", "   ") == "NOT_ATTEMPTED" and not judge.seen
