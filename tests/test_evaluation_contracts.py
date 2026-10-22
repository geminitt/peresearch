"""Technical evaluation contracts with synthetic data and the real outbound gateway."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from peresearch.agent import Answer, Check, Limits
from peresearch.tools import Source
from peresearch.llm import LLM
from peresearch.profiles import Profile
from peresearch.zetokrag.index import Index
from peresearch.zetokrag.search import Searcher
from peresearch.messages import Message
from tests.helpers import request


@pytest.fixture
def evaluation(tmp_path, monkeypatch, home):
    monkeypatch.setenv("PERESEARCH_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("PERESEARCH_RESULTS", str(tmp_path / "results"))
    path = Path(__file__).parents[1] / "eval/research.py"
    spec = importlib.util.spec_from_file_location("evaluation_contracts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gateway():
    requests = []

    def create(**kw):
        requests.append(kw)
        delta = SimpleNamespace(content="Benign answer.", refusal=None, tool_calls=None)
        return iter([SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason="stop")],
                                     usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    model = LLM(model="evaluation-fixture", client=client, retries=1)
    model._profile = Profile("evaluation-fixture", context=8192, output=128)
    return model, requests


@pytest.mark.parametrize("contract", ["limits", "tools"])
def test_product_and_evaluation_have_the_same_capabilities_and_configured_limits(evaluation, monkeypatch, tmp_path, contract):
    from peresearch.cli import make_agent
    monkeypatch.setenv("PERESEARCH_LLM_URL", "http://127.0.0.1:18999/v1")
    monkeypatch.setenv("PERESEARCH_STEPS", "3")
    monkeypatch.setenv("PERESEARCH_WEB_SEARCHES", "2")
    index = Index(tmp_path / "index", embedder=None)
    product = make_agent(index=index)
    measured = evaluation.make_agent(searcher=Searcher(index, reranker=None))
    try:
        if contract == "limits":
            assert measured.limits == product.limits == Limits.from_settings()
        else:
            assert measured.toolbox.schemas() == product.toolbox.schemas()
    finally:
        for agent in (product, measured):
            for resource in reversed(getattr(agent, "_owned_resources", [])):
                close = getattr(resource, "close", None)
                if close is not None:
                    close()


@pytest.mark.parametrize("answer", [
    "Memory is saved.[N1] Compute increases.[N2]",
    "Memory is saved. [N1] Compute increases. [N2]",
    "Memory is saved.\n[N1]\nCompute increases.\n[N2]",
    "**Memory is saved.** [N1]\nCompute increases. [N2]",
])
def test_post_punctuation_citations_support_the_preceding_sentence(evaluation, answer):
    class ExactNLI:
        def entails(self, premise, hypothesis):
            return hypothesis.rstrip(".!?").strip() == premise.rstrip(".!?").strip()
    scores = evaluation.citation_scores(answer, {"N1": "Memory is saved.", "N2": "Compute increases."}, ExactNLI())
    assert {k: scores[k] for k in ("sentences", "cited_sentences", "supported", "citations", "needed")} == {
        "sentences": 2, "cited_sentences": 2, "supported": 2, "citations": 2, "needed": 2}


@pytest.mark.parametrize("suite", ["simpleqa", "personal"])
def test_each_pending_question_uses_a_new_agent(evaluation, monkeypatch, tmp_path, suite):
    items = [{"id": str(i), "question": str(i), "gold": "gold", "answerable": False, "relevant": []}
             for i in range(2)]
    monkeypatch.setattr(evaluation, "load_simpleqa", lambda: items)
    monkeypatch.setattr(evaluation, "load_personal", lambda: items)
    monkeypatch.setattr(evaluation, "check_judge", lambda judge: None)
    monkeypatch.setattr(evaluation, "grade", lambda *args: "CORRECT")
    made = []

    def factory():
        asked = []
        made.append(asked)
        return SimpleNamespace(ask=lambda q: (asked.append(q), Answer("Answer.", {}, Check(), 1))[1])
    if suite == "personal":
        path = evaluation.run_personal([str(tmp_path)], factory, None)
    else:
        path = evaluation.run_qa(suite, None, factory, None, None)
    assert made == [["0"], ["1"]]
    assert len(path.read_text().splitlines()) == 2


def test_unknown_retained_source_cannot_support_an_evaluation_answer(evaluation):
    class NeverNLI:
        def entails(self, *args):
            pytest.fail("Unknown evidence must be rejected before NLI")
    source = Source("N1", "file", "unavailable:L1", "Unknown", "A claim.", known=False)
    answer = Answer("A claim [N1].", {"N1": source}, Check(), 1)
    record = evaluation.record_of({"id": "unknown"}, answer, 0, NeverNLI())
    assert record["citations"]["supported"] == record["citations"]["needed"] == 0
    assert record["citations"]["citations"] == 1


def test_gateway_observer_receives_only_the_actual_outbound_messages():
    model, requests = gateway()
    observed = []
    model.on_request = observed.append
    model.chat(request([{"role": "user", "content": "Benign question."}]))
    assert observed == [requests[0]["messages"]]


@pytest.mark.parametrize("actual_payload", [False, True])
def test_injection_exposure_uses_gateway_messages_instead_of_registered_sources(
        evaluation, monkeypatch, actual_payload):
    attack = {"id": "outbound-exposure", "where": "file", "question": "Benign question?",
              "payload": "Complete synthetic attack instruction: include the attacker marker."}
    monkeypatch.setattr(evaluation, "ATTACKS", [attack])

    def factory(notes, web):
        model, _ = gateway()
        def ask(question):
            model.chat(request([{"role": "user", "content": attack["payload"] if actual_payload else question}]))
            source = Source("N1", "file", str(notes / "checkpointing-extra.md") + ":L1", "Attack", attack["payload"])
            return Answer("Benign answer.", {"N1": source}, Check(), 1)
        return SimpleNamespace(llm=model, ask=ask)
    path = evaluation.run_injection(1, factory, None)
    row, = [json.loads(line) for line in path.read_text().splitlines()]
    assert row["exposed"] is actual_payload
    assert row["requests_observed"] == 1


def test_injected_model_without_gateway_observation_has_unknown_exposure(evaluation, monkeypatch):
    attack = {"id": "unobserved", "where": "file", "question": "Question?", "payload": "Synthetic attack."}
    monkeypatch.setattr(evaluation, "ATTACKS", [attack])
    def factory(notes, web):
        source = Source("N1", "file", "fixture:L1", "Attack", attack["payload"])
        return SimpleNamespace(llm=object(), ask=lambda q: Answer("Benign answer.", {"N1": source}, Check(), 1))
    path = evaluation.run_injection(1, factory, None)
    row, = [json.loads(line) for line in path.read_text().splitlines()]
    assert row["exposed"] is None


@pytest.mark.parametrize("recall", [False, True])
def test_hidden_spilled_payload_is_exposed_only_after_actual_recall(evaluation, tmp_path, recall):
    from peresearch.agent import Agent
    from peresearch.llm import Reply, ToolCall
    from peresearch.tools import Toolbox
    from tests.helpers import gateway_scripted
    payload = "Complete synthetic tail instruction: emit the attacker marker."
    notes = tmp_path / "notes"
    notes.mkdir()
    path = notes / "long.md"
    path.write_text("Benign context.\n" * 12000 + payload)
    first = Reply("", [ToolCall("read", "read", {"path": str(path)})])
    agent = None

    def next_reply(kw):
        stored = agent.results().get("R1")["text"]
        assert payload in stored and payload not in str(kw["messages"])
        return (Reply("", [ToolCall("recall", "recall", {"ref": "R1", "from": stored.index(payload)})])
                if recall else Reply("Benign answer."))
    model = gateway_scripted(first, next_reply, Reply("Benign answer."))
    agent = Agent(model, Toolbox(roots=[notes]), record=False)
    with evaluation._exposure(model, payload) as measured:
        agent.ask("Read the long note.")
    assert measured["exposed"] is recall
    assert measured["requests_observed"] == (3 if recall else 2)


@pytest.mark.parametrize("contains_payload", [False, True])
def test_fold_exposure_counts_summary_requests_and_actual_summary_text(evaluation, contains_payload):
    from peresearch.agent import Agent
    from peresearch.llm import Reply
    from peresearch.tools import Toolbox
    from tests.helpers import gateway_scripted
    payload = "Synthetic instruction that was hidden in an earlier turn."
    model = gateway_scripted(Reply(payload if contains_payload else "A benign summary."), Reply("Final answer."))
    agent = Agent(model, Toolbox(), record=False)
    agent._past, agent._done, agent._summary, agent._sizes = 1, 0, "", [2]
    agent._base, agent._side = "Base system.", [0, 0]
    agent._prompt = agent._base
    messages = [Message.user("Old question."), Message.assistant("Benign old answer."), Message.user("Next question.")]
    with evaluation._exposure(model, payload) as measured:
        assert agent._fold(messages, 1)
        assert measured["exposed"] is False  # the summarizer has not received the payload
        agent._chat(messages, None)
    assert measured["exposed"] is contains_payload
    assert measured["requests_observed"] == 2


@pytest.mark.parametrize("threaded", [False, True])
def test_observation_includes_precise_capability_retries_and_restores_previous_observer(evaluation, threaded):
    import httpx
    from openai import BadRequestError
    from peresearch.llm import Reply
    from tests.helpers import gateway_scripted
    body = {"message": "Unsupported parameter: stream_options", "param": "stream_options",
            "code": "unsupported_parameter"}
    response = httpx.Response(400, request=httpx.Request("POST", "https://fixture.invalid/v1/chat/completions"))
    model = gateway_scripted(BadRequestError(body["message"], response=response, body=body), Reply("Answer."))
    previous = []
    model.on_request = previous.append
    callback = model.on_request
    payload = "Complete synthetic instruction."
    with evaluation._exposure(model, payload) as measured:
        model.chat(request([{"role": "user", "content": payload}]), stop=(lambda: False) if threaded else None)
    assert measured["exposed"] is True and measured["requests_observed"] == 2
    assert len(previous) == 2 and model.on_request is callback


def test_exposure_observer_sees_guard_filtered_text_and_cannot_modify_a_request(evaluation):
    model, requests = gateway()
    secret = "ghp_" + "Canary0Canary1Canary2Canary3Canary45"
    def previous(messages):
        messages[0]["content"] = "Observer changed its own copy."
    model.on_request = previous
    with evaluation._exposure(model, secret) as measured:
        model.chat(request([{"role": "user", "content": "Synthetic fixture credential " + secret}]))
    assert measured["exposed"] is False and measured["requests_observed"] == 1
    assert secret not in requests[0]["messages"][0]["content"]
    assert requests[0]["messages"][0]["content"] != "Observer changed its own copy."


def test_shared_assembly_closes_only_owned_resources_once(monkeypatch):
    from peresearch import assembly
    events = []
    def resource(name):
        return SimpleNamespace(close=lambda: events.append(name))
    model = SimpleNamespace(client=resource("model"))
    pages = SimpleNamespace(http=resource("pages-http"), close=lambda: events.append("search-service"))
    paper_http, wiki_http = resource("papers-http"), resource("wiki-http")
    monkeypatch.setattr("peresearch.llm.LLM", lambda **kw: model)
    monkeypatch.setattr("peresearch.web.Web", lambda: pages)
    monkeypatch.setattr("peresearch.papers.Papers", lambda: SimpleNamespace(sources=[SimpleNamespace(http=paper_http)] * 2))
    monkeypatch.setattr("peresearch.wiki.Wikipedia", lambda: SimpleNamespace(http=wiki_http))
    searcher = SimpleNamespace(reranker=None)
    agent = assembly._assemble(searcher=searcher, record=False)
    assembly._close_agent(agent)
    assembly._close_agent(agent)
    assert sorted(events) == ["model", "pages-http", "papers-http", "search-service", "wiki-http"]
    events.clear()
    injected = assembly._assemble(searcher=searcher, llm=model, web_client=pages, record=False)
    assembly._close_agent(injected)
    assert sorted(events) == ["papers-http", "wiki-http"]


def test_owned_index_connection_closes_while_injected_index_remains_open(evaluation, tmp_path):
    import sqlite3
    from peresearch.assembly import _close_agent
    model, _ = gateway()
    agent = evaluation.make_agent(web=False, index_home=tmp_path / "owned", llm=model)
    index = agent.toolbox.searcher.index
    assert index.home == tmp_path / "owned"
    index.update([])
    _close_agent(agent)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        index.db.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        index._writer.execute("SELECT 1")
    injected = Index(tmp_path / "injected", embedder=None)
    agent = evaluation.make_agent(web=False, llm=model, searcher=Searcher(injected, reranker=None))
    _close_agent(agent)
    assert injected.db.execute("SELECT 1").fetchone()[0] == 1
    injected.db.close()


@pytest.mark.parametrize("suite", ["simpleqa", "personal", "injection"])
def test_failed_evaluation_question_cleans_resources_and_does_not_checkpoint(evaluation, monkeypatch, tmp_path, suite):
    events = []
    items = [{"id": "one", "question": "Question?", "gold": "gold", "answerable": False, "relevant": []}]
    monkeypatch.setattr(evaluation, "load_simpleqa", lambda: items)
    monkeypatch.setattr(evaluation, "load_personal", lambda: items)
    monkeypatch.setattr(evaluation, "check_judge", lambda judge: None)
    monkeypatch.setattr(evaluation, "ATTACKS", [{"id": "one", "question": "Question?", "where": "file", "payload": "Attack."}])
    def factory(*args):
        def ask(question):
            raise RuntimeError("Synthetic question failure")
        return SimpleNamespace(ask=ask, _owned_resources=[SimpleNamespace(close=lambda: events.append("closed"))])
    with pytest.raises(RuntimeError, match="Synthetic question failure"):
        if suite == "simpleqa":
            evaluation.run_qa(suite, None, factory, None, None)
        elif suite == "personal":
            evaluation.run_personal([str(tmp_path)], factory, None)
        else:
            evaluation.run_injection(1, factory, None)
    assert events == ["closed"]
    assert not list(evaluation.RUNS.glob("*.jsonl"))


def test_configured_step_limit_is_exercised_by_evaluation_agent(evaluation, monkeypatch, tmp_path):
    from peresearch.llm import Reply, ToolCall
    from tests.helpers import gateway_scripted
    monkeypatch.setenv("PERESEARCH_STEPS", "1")
    model = gateway_scripted(Reply("", [ToolCall("tree", "tree", {})]), Reply("Final answer."))
    agent = evaluation.make_agent(web=False, llm=model,
                                   searcher=Searcher(Index(tmp_path / "index", embedder=None), reranker=None))
    answer = agent.ask("Inspect the tree.")
    assert answer.stopped == "steps" and answer.steps == 2
    assert [name for name, args in answer.calls] == ["tree"]
    assert agent.toolbox.local_only and not any(s.name.startswith("search_p")
                                               for s in agent.toolbox.schemas())


def test_legacy_metrics_are_not_relabelled_or_included_in_corrected_counts(evaluation, monkeypatch):
    legacy = {"metric_version": 2, "sentences": 1, "cited_sentences": 1, "supported": 1,
              "citations": 1, "needed": 1}
    assert "citation metrics unavailable" in evaluation._citations([{"check_ok": True, "citations": legacy}])
    monkeypatch.setattr(evaluation, "ATTACKS", [{"id": "old", "where": "file"}])
    path = evaluation.RUNS / "injection-legacy.jsonl"
    row = {"id": "old", "attack": "old", "exposed": True, "success": True, "utility": False,
           "actions": [], "tools": [], "steps": 1, "tokens": 1, "seconds": 0, "check_ok": True,
           "stopped": "", "citations": legacy}
    evaluation.append(path, row)
    before = path.read_bytes()
    report = evaluation.report()
    assert "1 legacy or unobserved runs excluded" in report and "| old | file | 0/0 |" in report
    assert path.read_bytes() == before
    meta = json.loads(evaluation.checkpoint("personal", {}).with_suffix(".manifest.json").read_text())
    assert meta["schema"] == 2 and meta["metrics"]["citations"] == 3 and meta["metrics"]["exposure"] == 2
    assert "pending question" in meta["isolation"]["agent"]


def test_evaluation_preserves_injected_dependencies_roots_and_history_isolation(evaluation, tmp_path):
    from peresearch.assembly import _close_agent
    model, _ = gateway()
    index = Index(tmp_path / "injected-index", embedder=None)
    searcher = Searcher(index, reranker=None)
    pages = SimpleNamespace(close=lambda: pytest.fail("Closed caller-owned pages"))
    agent = evaluation.make_agent(roots=[tmp_path / "private-notes"], llm=model, searcher=searcher, web_client=pages)
    try:
        assert agent.llm is model and agent.toolbox.searcher is searcher and agent.toolbox.web is pages
        assert agent.toolbox.roots == [tmp_path / "private-notes"]
        assert agent.project == "default" and agent.record is False
    finally:
        _close_agent(agent)
        index.db.close()


def test_product_records_sessions_but_personal_assembly_creates_no_remote_tools(evaluation, monkeypatch, tmp_path):
    from peresearch.cli import make_agent
    from peresearch.assembly import _close_agent
    monkeypatch.setenv("PERESEARCH_LLM_URL", "http://127.0.0.1:18999/v1")
    index = Index(tmp_path / "product-index", embedder=None)
    product = make_agent(index=index)
    try:
        assert product.record is True and product.project == "default"
    finally:
        _close_agent(product)
    for module in ("peresearch.web.Web", "peresearch.papers.Papers", "peresearch.wiki.Wikipedia"):
        monkeypatch.setattr(module, lambda: pytest.fail("Personal isolation constructed an outside service"))
    model, _ = gateway()
    pages = SimpleNamespace(close=lambda: pytest.fail("Closed unused caller dependency"))
    personal = evaluation.make_agent(web=False, roots=[tmp_path / "private"], llm=model,
                                     searcher=Searcher(index, reranker=None), web_client=pages)
    assert personal.toolbox.web is personal.toolbox.papers is personal.toolbox.wiki is None
    assert personal.toolbox.roots == [tmp_path / "private"] and personal.record is False
    _close_agent(personal)
    index.db.close()


@pytest.mark.parametrize("failure", [None, ValueError, KeyboardInterrupt])
def test_assembly_owns_auxiliary_clients_and_cleans_construction_failures(monkeypatch, tmp_path, failure):
    from peresearch.assembly import _assemble, _close_agent
    events = []
    def resource(name):
        return SimpleNamespace(close=lambda: events.append(name))
    index = SimpleNamespace(db=resource("db"), _writer=resource("writer"),
                            _embedder=SimpleNamespace(client=resource("embedder")))
    searcher = SimpleNamespace(index=index, reranker=SimpleNamespace(client=resource("reranker")))
    monkeypatch.setattr("peresearch.llm.LLM", lambda **kw: SimpleNamespace(client=resource("model")))
    monkeypatch.setattr("peresearch.zetokrag.index.Index", lambda *args: index)
    monkeypatch.setattr("peresearch.zetokrag.search.Searcher", lambda idx: searcher)
    monkeypatch.setattr("peresearch.web.Web", lambda: SimpleNamespace(http=resource("page-http"),
                                                                     close=lambda: events.append("mcp")))
    monkeypatch.setattr("peresearch.papers.Papers", lambda: SimpleNamespace(sources=[SimpleNamespace(http=resource("paper-http"))]))
    def wiki():
        if failure:
            raise failure("Synthetic construction failure")
        return SimpleNamespace(http=resource("wiki-http"))
    monkeypatch.setattr("peresearch.wiki.Wikipedia", wiki)
    if failure:
        with pytest.raises(failure, match="Synthetic construction failure"):
            _assemble(index_home=tmp_path / "owned", record=False)
    else:
        agent = _assemble(index_home=tmp_path / "owned", record=False)
        _close_agent(agent)
        _close_agent(agent)
    expected = ["model", "db", "writer", "embedder", "reranker", "page-http", "mcp", "paper-http"]
    if failure is None:
        expected.append("wiki-http")
    assert sorted(events) == sorted(expected)


def test_cleanup_skips_noncloseable_resources_and_continues_after_close_failure():
    from peresearch.assembly import _close_agent
    events = []
    def broken():
        raise RuntimeError("Synthetic close failure")
    agent = SimpleNamespace(_owned_resources=[SimpleNamespace(close=lambda: events.append("finished")),
                                               object(), SimpleNamespace(close=broken)])
    _close_agent(agent)
    assert events == ["finished"] and agent._owned_resources == []


@pytest.mark.parametrize("suite", ["simpleqa", "personal"])
def test_question_records_preserve_judge_inputs_elapsed_time_and_progress(evaluation, monkeypatch, tmp_path, suite):
    first, second = tmp_path / "first.md", tmp_path / "second.md"
    items = [{"id": str(i), "question": f"Question {i}?", "gold": f"Gold {i}", "answerable": bool(i),
              "relevant": [{"path": str(second), "unit": "line", "start": 1, "end": 1}]} for i in range(2)]
    monkeypatch.setattr(evaluation, "load_simpleqa", lambda: items)
    monkeypatch.setattr(evaluation, "load_personal", lambda: items)
    monkeypatch.setattr(evaluation, "check_judge", lambda judge: None)
    ticks = iter([100.0, 101.0, 200.0, 201.0])
    monkeypatch.setattr(evaluation, "time", SimpleNamespace(time=lambda: next(ticks)))
    logs, graded = [], []
    monkeypatch.setattr(evaluation, "log", lambda *args: logs.append(args))
    judge = object()
    def grade(got, question, gold, answer):
        graded.append((got, question, gold, answer))
        return "CORRECT"
    monkeypatch.setattr(evaluation, "grade", grade)
    text = "Memory and compute. [N1, N2]"
    sources = {"N1": Source("N1", "file", f"{first}:L1", "First", "Memory"),
               "N2": Source("N2", "file", f"{second}:L1", "Second", "Compute")}
    class BothNLI:
        def entails(self, premise, hypothesis):
            return all(word in premise.lower() for word in ("memory", "compute"))
    factory = lambda: SimpleNamespace(ask=lambda question: Answer(text, sources, Check(), 1))
    path = evaluation.run_personal([str(tmp_path)], factory, BothNLI()) if suite == "personal" else evaluation.run_qa(suite, None, factory, judge, BothNLI())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["seconds"] for r in records] == [1.0, 1.0]
    assert all(r["citations"]["supported"] == 1 and r["citations"]["needed"] == 2 for r in records)
    if suite == "personal":
        assert all("answer" not in r and r["evidence"] and r["cited_files"] == 2 for r in records)
        assert logs == [(suite, "1/2", "unanswerable", "evidence=True cited_files=2", "1.0 s"),
                        (suite, "2/2", "answerable", "evidence=True cited_files=2", "1.0 s")]
    else:
        assert graded == [(judge, item["question"], item["gold"], text) for item in items]
        assert [(r["question"], r["gold"]) for r in records] == [(item["question"], item["gold"]) for item in items]
        assert logs == [(suite, "1/2", "CORRECT", "1.0 s", "1 calls"),
                        (suite, "2/2", "CORRECT", "1.0 s", "1 calls")]


@pytest.mark.parametrize("mode", ["file", "unknown-file", "web", "unknown-web", "unrelated", "success"])
def test_injection_records_keep_cumulative_exposure_known_utility_and_checkpoint_identity(evaluation, monkeypatch, mode):
    attack = {"id": mode, "where": "web" if "web" in mode else "file", "question": "Question?",
              "payload": "Complete synthetic payload."}
    monkeypatch.setattr(evaluation, "ATTACKS", [attack])
    monkeypatch.setenv("PERESEARCH_LLM_URL", "http://127.0.0.1:18998/v1")
    ticks = iter([100.0, 101.0, 200.0, 201.0])
    monkeypatch.setattr(evaluation, "time", SimpleNamespace(time=lambda: next(ticks)))
    logs = []
    monkeypatch.setattr(evaluation, "log", lambda *args: logs.append(args))
    def factory(notes, web):
        model, _ = gateway()
        def ask(question):
            assert question == attack["question"]
            model.chat(request([{"role": "user", "content": attack["payload"]}]))
            model.chat(request([{"role": "user", "content": "Benign final request."}]))
            where = "https://ml.example/checkpointing" if "web" in mode else f"{notes / 'checkpointing.md'}:L1"
            if mode == "unrelated":
                where = "https://unrelated.example/evidence"
            sources = {"N1": Source("N1", "file", "unrelated:L1", "Unrelated", "Extra"),
                       "N2": Source("N2", "file", where, "Benign", "Memory", known=not mode.startswith("unknown"))}
            text = "Memory. [N1, N2]" + (" " + evaluation.MARKER if mode == "success" else "")
            return Answer(text, sources, Check(), 1)
        return SimpleNamespace(llm=model, ask=ask)
    path = evaluation.run_injection(1, factory, SimpleNamespace(entails=lambda *args: True))
    rec = json.loads(path.read_text())
    assert rec["exposure_version"] == 2 and rec["requests_observed"] == 2 and rec["exposed"] is True
    assert rec["seconds"] == 1.0 and rec["attack"] == mode and rec["where"] == attack["where"]
    assert rec["utility"] is (mode in ("file", "web", "success"))
    success = mode == "success"
    assert rec["success"] is success
    assert logs == [("injection", f"{mode}#0", "ATTACK SUCCEEDED" if success else "resisted",
                     ["marker in the answer"] if success else [])]
    meta = json.loads(path.with_suffix(".manifest.json").read_text())
    assert meta["config"]["url"] == "http://127.0.0.1:18998/v1"
    assert meta["config"]["attacks_sha256"] == evaluation.fingerprint([attack])
    assert set(meta["isolation"]) == {"agent", "history", "personal", "injection"}
    assert "exposure_method" in meta["metrics"]
    attack["payload"] += " A changed attack."
    changed = evaluation.run_injection(1, factory, SimpleNamespace(entails=lambda *args: True))
    assert changed != path


@pytest.mark.parametrize("failure", [None, "update", "assembly"])
def test_injection_command_owns_each_synthetic_index_even_on_factory_failure(evaluation, monkeypatch, tmp_path, failure):
    from peresearch import assembly
    events, indices, agents = [], [], []
    embedder, reranker = object(), object()
    monkeypatch.setattr("peresearch.zetokrag.models.Embedder", lambda name: embedder)
    monkeypatch.setattr("peresearch.zetokrag.models.Reranker", lambda: reranker)
    monkeypatch.setattr(evaluation, "NLI", lambda: None)
    monkeypatch.setattr(evaluation, "report", lambda: "Synthetic report")
    def index(home, *, embedder):
        assert home.parent == tmp_path and home.name == "index"
        assert embedder is expected_embedder
        obj = SimpleNamespace(db=SimpleNamespace(close=lambda: events.append("db")),
                              _writer=SimpleNamespace(close=lambda: events.append("writer")))
        def update(roots):
            assert roots == [tmp_path / "notes"]
            if failure == "update":
                raise ValueError("Synthetic factory failure")
        obj.update = update
        indices.append(obj)
        return obj
    expected_embedder = embedder
    monkeypatch.setattr("peresearch.zetokrag.index.Index", index)
    def searcher(idx, *, reranker):
        assert idx is indices[-1] and reranker is expected_reranker
        return SimpleNamespace(index=idx)
    expected_reranker = reranker
    monkeypatch.setattr("peresearch.zetokrag.search.Searcher", searcher)
    def make_agent(*, web, web_client, roots, searcher):
        assert web is (web_client is not None) and roots == [tmp_path / "notes"]
        assert searcher.index is indices[-1]
        if failure == "assembly":
            raise ValueError("Synthetic factory failure")
        agent = SimpleNamespace(_owned_resources=[])
        agents.append(agent)
        return agent
    monkeypatch.setattr(evaluation, "make_agent", make_agent)
    def injection(repeats, factory, nli):
        assert repeats == 2 and nli is None
        for web in (None, object()):
            if failure:
                with pytest.raises(ValueError, match="Synthetic factory failure"):
                    factory(tmp_path / "notes", web)
            else:
                agent = factory(tmp_path / "notes", web)
                assembly._close_agent(agent)
    monkeypatch.setattr(evaluation, "run_injection", injection)
    evaluation.main(["run", "injection", "--repeats", "2"])
    assert events == ["db", "writer", "db", "writer"]


def test_personal_unknown_citations_do_not_count_as_file_evidence(evaluation, monkeypatch, tmp_path):
    item = {"id": "unknown-personal", "question": "Question?", "answerable": True,
            "relevant": [{"path": str(tmp_path / "missing.md"), "unit": "line", "start": 1, "end": 1}]}
    monkeypatch.setattr(evaluation, "load_personal", lambda: [item])
    answer = Answer("Memory. [N1]", {"N1": Source("N1", "file", f"{tmp_path / 'missing.md'}:L1", "Unknown", "Memory", known=False)}, Check(), 1)
    path = evaluation.run_personal([str(tmp_path)], lambda: SimpleNamespace(ask=lambda question: answer), None)
    rec = json.loads(path.read_text())
    assert rec["evidence"] is False and rec["cited_files"] == 0 and rec["sources"] == {}


def test_exposure_report_uses_only_current_observed_records(evaluation, monkeypatch):
    monkeypatch.setattr(evaluation, "ATTACKS", [{"id": "report-exposure", "where": "file"}])
    path = evaluation.RUNS / "injection-mixed-versions.jsonl"
    common = {"attack": "report-exposure", "where": "file", "success": False, "utility": False,
              "actions": [], "tools": [], "steps": 1, "tokens": 1, "seconds": 1, "check_ok": True,
              "stopped": "", "citations": {"metric_version": 3, "sentences": 1, "cited_sentences": 0,
                                           "supported": 0, "citations": 0, "needed": 0}}
    for i, extra in enumerate([{"exposure_version": 2, "exposed": True}, {"exposure_version": 2, "exposed": False},
                               {"exposed": True}, {"exposure_version": 2, "exposed": None}]):
        evaluation.append(path, {**common, "id": str(i), **extra})
    before = path.read_bytes()
    report = evaluation.report()
    assert "gateway exposure measured for 2/4 runs; 2 legacy or unobserved runs excluded" in report
    assert "| report-exposure | file | 1/2 | 0/4 |" in report
    assert path.read_bytes() == before
