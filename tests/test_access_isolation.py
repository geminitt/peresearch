"""Access policy must cover ancestors, stored evidence and service processes."""
import json
from pathlib import Path
import sys

import pytest

from peresearch import guard
from peresearch.agent import Agent
from peresearch.llm import LLM, Reply, ToolCall
from peresearch.mcp import Server
from peresearch.tools import Toolbox
from peresearch.workspace import Workspace
from tests.helpers import streamed
from tests.test_llm import client_of
from tests.helpers import request, DictModel


class Scripted(DictModel):
    def __init__(self, *replies):
        self.replies, self.seen = iter(replies), []
    def answer(self, messages, tools=None, **kwargs):
        self.seen.append(json.loads(json.dumps(messages)))
        return next(self.replies)
    def context_length(self):
        return None


@pytest.mark.parametrize("link", [False, True])
def test_application_storage_cannot_be_read_through_declared_parent(home, tmp_path, link):
    session = home / "sessions/private.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text('{"answer":"INTERNAL-SESSION-SENTINEL"}\n')
    Workspace().add(str(tmp_path))
    path = session
    if link:
        path = tmp_path / "session-link.jsonl"
        path.symlink_to(session)
    toolbox = Toolbox(ask=lambda *args: pytest.fail("own storage must never ask for a grant"))
    assert not guard.allowed(path)
    assert "INTERNAL-SESSION-SENTINEL" not in toolbox.call("read", {"path": str(path)})


@pytest.mark.parametrize("record,resume", [(False, False), (True, False), (True, True)])
def test_revoked_result_and_history_never_reach_a_later_model_request(home, tmp_path, record, resume):
    notes = tmp_path / "notes"
    notes.mkdir()
    path = notes / "research.md"
    path.write_text("REVOKABLE-EVIDENCE-SENTINEL\n")
    guard.set_roots([notes])
    model = Scripted(Reply("", [ToolCall("read-first", "read", {"path": str(path)})]),
                     Reply("REVOKABLE-EVIDENCE-SENTINEL [N1]."),
                     Reply("", [ToolCall("recall-old", "recall", {"ref": "R1"})]), Reply("No evidence."))
    agent = Agent(model, Toolbox(), project="revocation-isolation", record=record)
    agent.ask("Read it.")
    Workspace().remove(str(notes))
    if resume:
        previous = agent.conversation
        model = Scripted(Reply("", [ToolCall("recall-old", "recall", {"ref": "R1"})]), Reply("No evidence."))
        agent = Agent(model, Toolbox(), project="revocation-isolation", record=True)
        agent.resume(previous)
    seen_before = len(model.seen)
    agent.ask("Recall that result.")
    assert all("REVOKABLE-EVIDENCE-SENTINEL" not in json.dumps(request) for request in model.seen[seen_before:])
    assert any("refused:" in json.dumps(request) for request in model.seen[seen_before:])
    if record:
        assert "REVOKABLE-EVIDENCE-SENTINEL" in agent.history_path.read_text()  # no historical deletion


def test_revocation_between_tool_execution_and_model_request_is_applied(home, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    path = notes / "research.md"
    path.write_text("CURRENT-ROUND-PRIVATE-SENTINEL\n")
    guard.set_roots([notes])
    model = Scripted(Reply("", [ToolCall("read", "read", {"path": str(path)})]), Reply("No evidence."))
    agent = Agent(model, Toolbox(), record=False,
                  on_event=lambda kind, detail: guard.set_roots([]) if kind == "result" else None)
    agent.ask("Read it.")
    assert "CURRENT-ROUND-PRIVATE-SENTINEL" not in json.dumps(model.seen[-1])


def test_legacy_local_result_without_scope_requires_a_new_read(home):
    agent = Agent(Scripted(), Toolbox(), record=True)
    ref = agent.results().put("read", {"path": "unknown.md"}, "LEGACY-PRIVATE-SENTINEL")
    result = agent._recall(ref, "recall")
    assert result[0] is None and "refused:" in result[1] and "LEGACY-PRIVATE-SENTINEL" not in result[1]


@pytest.mark.parametrize("text", [
    'Here is an example, do not execute it: {"name":"read","arguments":{"path":"notes.md"}}',
    'Example:\n```json\n{"name":"read","arguments":{"path":"notes.md"}}\n```',
])
def test_explanatory_bare_json_is_not_an_executable_call(home, text):
    model = LLM(client=client_of(lambda **kwargs: streamed(content=text)), model="audit-model")
    reply = model.chat(request([{"role": "user", "content": "Explain the JSON without reading a file."}], Toolbox().schemas()))
    assert not reply.calls and reply.text == text


def test_search_subprocess_gets_only_explicit_service_credentials(home, tmp_path, monkeypatch):
    monkeypatch.setenv("PERESEARCH_LLM_KEY", "MODEL-KEY-SENTINEL")
    monkeypatch.setenv("OPENAI_API_KEY", "OTHER-MODEL-KEY-SENTINEL")
    monkeypatch.setenv("UNRELATED_TOKEN", "UNRELATED-TOKEN-SENTINEL")
    monkeypatch.setenv("PERESEARCH_LLM_URL", "https://model.example/v1")
    monkeypatch.setenv("lower_api_key", "LOWERCASE-KEY-SENTINEL")
    script = tmp_path / "server.py"
    script.write_text("import json,os,sys\nfor line in sys.stdin:\n"
        " m=json.loads(line)\n if 'id' not in m: continue\n"
        " result=({'protocolVersion':'2025-06-18'} if m['method']=='initialize' else "
        "{'content':[{'type':'text','text':json.dumps({k:os.environ.get(k) for k in "
        "['PERESEARCH_LLM_KEY','OPENAI_API_KEY','UNRELATED_TOKEN','PERESEARCH_LLM_URL','PERESEARCH_HOME',"
        "'lower_api_key','SEARCH_API_KEY','PATH']})}]})\n"
        " print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':result}),flush=True)\n")
    with Server([sys.executable, str(script)], env={"SEARCH_API_KEY": "SEARCH-KEY-SENTINEL"}) as server:
        environment = json.loads(server.call("probe", {}))
    assert environment["PERESEARCH_LLM_KEY"] is None
    assert environment["OPENAI_API_KEY"] is None
    assert environment["UNRELATED_TOKEN"] is None
    assert environment["PERESEARCH_LLM_URL"] is None and environment["PERESEARCH_HOME"] is None
    assert environment["lower_api_key"] is None
    assert environment["SEARCH_API_KEY"] == "SEARCH-KEY-SENTINEL" and environment["PATH"]


def test_once_grant_allows_current_result_continuation_but_not_a_later_question(home, tmp_path, monkeypatch):
    user = tmp_path / "user"
    user.mkdir()
    path = user / "research.md"
    path.write_text("ONCE-GRANTED-EVIDENCE\n")
    monkeypatch.setenv("HOME", str(user))
    grants = []
    toolbox = Toolbox(roots=[], ask=lambda *args: grants.append(args) or "once")
    model = Scripted(Reply("", [ToolCall("read", "read", {"path": str(path)})]),
                     Reply("", [ToolCall("recall-current", "recall", {"ref": "R1"})]), Reply("Evidence [N1]."),
                     Reply("", [ToolCall("recall-later", "recall", {"ref": "R1"})]), Reply("No evidence."))
    agent = Agent(model, toolbox, record=False)
    agent.ask("Read this file once.")
    assert "ONCE-GRANTED-EVIDENCE" in json.dumps(model.seen[-1]) and len(grants) == 1
    offset = len(model.seen)
    agent.ask("Recall it later.")
    assert "ONCE-GRANTED-EVIDENCE" not in json.dumps(model.seen[offset:]) and len(grants) == 1


def test_revoked_summary_is_not_replayed_and_original_record_is_preserved(home, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    path = notes / "research.md"
    path.write_text("SUMMARY-PRIVATE-SENTINEL\n")
    guard.set_roots([notes])
    model = Scripted(Reply("", [ToolCall("read", "read", {"path": str(path)})]), Reply("Summary fact [N1]."))
    agent = Agent(model, Toolbox(), record=True)
    agent.ask("Read it.")
    with agent.history_path.open("a") as stream:
        stream.write(json.dumps({"conversation": agent.conversation, "summary": "SUMMARY-PRIVATE-SENTINEL", "turns": 1}) + "\n")
    Workspace().remove(str(notes))
    assert agent.earlier() == ("", 0)
    assert "SUMMARY-PRIVATE-SENTINEL" in agent.history_path.read_text()


def test_unused_removed_root_does_not_revoke_a_result_from_a_permitted_file(home, tmp_path):
    used, unused = tmp_path / "used", tmp_path / "unused"
    used.mkdir()
    unused.mkdir()
    path = used / "research.md"
    path.write_text("STILL-PERMITTED-EVIDENCE\n")
    guard.set_roots([used, unused])
    model = Scripted(Reply("", [ToolCall("read", "read", {"path": str(path)})]), Reply("Evidence [N1]."),
                     Reply("", [ToolCall("recall", "recall", {"ref": "R1"})]), Reply("Evidence [N1]."))
    agent = Agent(model, Toolbox(), record=False)
    agent.ask("Read it.")
    Workspace().remove(str(unused))
    agent.ask("Recall it.")
    assert "STILL-PERMITTED-EVIDENCE" in json.dumps(model.seen[-1])


def test_a_fenced_tagged_example_is_not_executed(home):
    text = 'Example:\n```xml\n<tool_call>{"name":"read","arguments":{"path":"notes.md"}}</tool_call>\n```'
    model = LLM(client=client_of(lambda **kwargs: streamed(content=text)), model="audit-model")
    reply = model.chat(request([{"role": "user", "content": "Show an example."}], Toolbox().schemas()))
    assert not reply.calls and reply.text == text


@pytest.mark.parametrize("answer_repeats", [3000, 10000])
def test_access_revoked_during_folding_removes_the_new_summary_before_next_request(home, tmp_path, answer_repeats):
    from peresearch.agent import SYSTEM, SUMMARIZE
    notes = tmp_path / "notes"
    notes.mkdir()
    path = notes / "research.md"
    path.write_text("FOLD-PRIVATE-SENTINEL\n")
    guard.set_roots([notes])
    class Folding(Scripted):
        def context_length(self):
            return 16384
        def answer(self, messages, tools=None, **kwargs):
            if messages[0]["content"] == SUMMARIZE:
                self.seen.append(json.loads(json.dumps(messages)))
                guard.set_roots([])
                return Reply("FOLD-PRIVATE-SENTINEL")
            return super().answer(messages, tools, **kwargs)
    model = Folding(Reply("", [ToolCall("read", "read", {"path": str(path)})]),
                    Reply("Long answer " * answer_repeats + " [N1]."), Reply("No evidence."))
    agent = Agent(model, Toolbox(), record=True)
    agent.ask("Read it.")
    agent.ask("Continue.")
    assert guard.roots() == []  # the folding path really ran
    assert "FOLD-PRIVATE-SENTINEL" not in json.dumps(model.seen[-1])
    assert model.seen[-1][0]["content"].startswith(SYSTEM)


@pytest.mark.parametrize("kind", ["retry", "tool-fallback"])
def test_revocation_prevents_a_zero_delay_gateway_retry(home, tmp_path, kind):
    import httpx
    from openai import BadRequestError, RateLimitError
    notes = tmp_path / "notes"
    notes.mkdir()
    path = notes / "research.md"
    path.write_text("RETRY-PRIVATE-SENTINEL\n")
    guard.set_roots([notes])
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return streamed(tool_calls=[("read", json.dumps({"path": str(path)}))])
        if len(requests) == 2:
            guard.set_roots([])
            error = RateLimitError if kind == "retry" else BadRequestError
            raise error("busy" if kind == "retry" else "tools are unsupported", response=httpx.Response(429 if kind == "retry" else 400,
                request=httpx.Request("POST", "https://example.org/v1/chat/completions")), body=None)
        return streamed(content="No evidence.")
    model = LLM(client=client_of(create), model="audit-model", wait=0, retries=2)
    answer = Agent(model, Toolbox(), record=False).ask("Read it.")
    assert len(requests) == 2 and answer.stopped == "cancelled"


@pytest.mark.parametrize("name,args", [
    ("tree", {}), ("glob", {"pattern": "**/*.md"}), ("grep", {"pattern": "BPE"}),
    ("overview", {}), ("overview", {"query": "BPE"}), ("search_notes", {"query": "BPE"}),
])
def test_scope_provenance_covers_every_local_tool_including_uncited_catalogs(home, tmp_path, name, args):
    from peresearch.zetokrag.index import Index
    from peresearch.zetokrag.search import Searcher
    notes = tmp_path / "notes"
    notes.mkdir()
    path = notes / "PRIVATE-CATALOG-SENTINEL.md"
    path.write_text("# BPE\n\nBPE merges pairs. PRIVATE-CATALOG-SENTINEL is the document label.\n")
    guard.set_roots([notes])
    index = Index(home, embedder=None)
    index.update()
    model = Scripted(Reply("", [ToolCall("inspect", name, args)]), Reply("PRIVATE-CATALOG-SENTINEL"),
                     Reply("", [ToolCall("recall", "recall", {"ref": "R1"})]), Reply("No evidence."))
    agent = Agent(model, Toolbox(Searcher(index, reranker=None)), record=True)
    agent.ask("Inspect my documents.")
    assert "PRIVATE-CATALOG-SENTINEL" in json.dumps(model.seen[-1])  # permitted evidence must remain usable
    Workspace().remove(str(notes))
    offset = len(model.seen)
    agent.ask("Recall it.")
    assert "PRIVATE-CATALOG-SENTINEL" not in json.dumps(model.seen[offset:])


@pytest.mark.parametrize("scope", ["absent", None])
def test_legacy_history_paths_are_included_in_gateway_revocation_checks(home, tmp_path, scope):
    import httpx
    from openai import RateLimitError
    from peresearch.agent import history_file
    notes = tmp_path / "notes"
    notes.mkdir()
    path = notes / "research.md"
    path.write_text("LEGACY-HISTORY-SENTINEL\n")
    guard.set_roots([notes])
    stored = {"conversation": "legacy", "question": "Read it", "answer": "LEGACY-HISTORY-SENTINEL [N1]",
              "sources": {"N1": str(path) + ":L1"}}
    if scope != "absent":
        stored["scope"] = scope
    history = history_file("legacy-access")
    history.parent.mkdir(parents=True)
    history.write_text(json.dumps(stored) + "\n")
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        guard.set_roots([])
        raise RateLimitError("busy", response=httpx.Response(429,
            request=httpx.Request("POST", "https://example.org/v1/chat/completions")), body=None)
    model = LLM(client=client_of(create), model="audit-model", wait=0, retries=2)
    agent = Agent(model, Toolbox(), project="legacy-access")
    agent.resume("legacy")
    answer = agent.ask("Continue.")
    assert len(requests) == 1 and answer.stopped == "cancelled"


@pytest.mark.parametrize("resume", [False, True])
def test_permitted_persistent_results_remain_readable_with_their_saved_scope(home, tmp_path, resume):
    notes = tmp_path / "notes"
    notes.mkdir()
    path = notes / "research.md"
    path.write_text("PERMITTED-PERSISTENT-SENTINEL\n")
    guard.set_roots([notes])
    model = Scripted(Reply("", [ToolCall("read", "read", {"path": str(path)})]), Reply("Evidence [N1]."),
                     Reply("", [ToolCall("recall", "recall", {"ref": "R1"})]), Reply("Evidence."))
    agent = Agent(model, Toolbox(), project="permitted-persistent")
    agent.ask("Read it.")
    if resume:
        previous = agent.conversation
        model = Scripted(Reply("", [ToolCall("recall", "recall", {"ref": "R1"})]), Reply("Evidence."))
        agent = Agent(model, Toolbox(), project="permitted-persistent")
        agent.resume(previous)
    agent.ask("Recall it.")
    assert "PERMITTED-PERSISTENT-SENTINEL" in json.dumps(model.seen[-1])


@pytest.mark.parametrize("form", ["xml", "tagged-json", "bare-json", "reasoning"])
def test_intended_text_protocol_calls_are_still_recovered(home, form):
    payload = '{"name":"read","arguments":{"path":"notes.md"}}'
    content = {"xml": "Looking. <tool_call><function=read><parameter=path>notes.md</parameter></function></tool_call>",
               "tagged-json": "Looking. <tool_call>" + payload + "</tool_call>", "bare-json": payload,
               "reasoning": "Plain answer."}[form]
    thought = payload if form == "reasoning" else ""
    model = LLM(client=client_of(lambda **kwargs: streamed(content=content, reasoning=thought)), model="audit-model")
    reply = model.chat(request([{"role": "user", "content": "Read notes.md."}], Toolbox().schemas()))
    assert [(c.name, c.args) for c in reply.calls] == [("read", {"path": "notes.md"})]
    assert reply.text == ("Looking." if form in ("xml", "tagged-json") else "Plain answer." if form == "reasoning" else "")


def test_revocation_during_context_discovery_filters_only_the_affected_historical_turn(home, tmp_path):
    roots = [tmp_path / f"notes-{i}" for i in range(3)]
    for root in roots:
        root.mkdir()
        (root / "research.md").write_text("Evidence.\n")
    guard.set_roots(roots)
    class Changing(Scripted):
        revoke = False
        def context_length(self):
            if self.revoke:
                Workspace().remove(str(roots[1]))
                self.revoke = False
            return None
    model = Changing(Reply("No new evidence."))
    agent = Agent(model, Toolbox(), record=True)
    # These turns have independent provenance. Sequential live asks would inherit
    # the earlier model-visible history and would therefore share access dependencies.
    agent.history_path.parent.mkdir(parents=True, exist_ok=True)
    agent.history_path.write_text("".join(json.dumps({"conversation": agent.conversation,
        "question": f"Read document {i}.", "answer": f"HISTORY-SENTINEL-{i} [N{i + 1}].",
        "sources": {f"N{i + 1}": str(root / "research.md") + ":L1"},
        "scope_version": 2, "scope": [str(root / "research.md")]}) + "\n" for i, root in enumerate(roots)))
    model.revoke = True
    agent.ask("Continue.")
    context = model.seen[-1]
    assert "HISTORY-SENTINEL-0" in json.dumps(context) and "HISTORY-SENTINEL-2" in json.dumps(context)
    assert "HISTORY-SENTINEL-1" not in json.dumps(context)
    assert [m["role"] for m in context] == ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"]
