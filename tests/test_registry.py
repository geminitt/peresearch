"""Contracts of the tool registry (peresearch.registry): one pipeline for every tool the model may call — the
arguments checked against the tool's declared schema, then its permission, then the question's budget, then the run
(errors as text, HTTP 429 waited out) — each refusal said in words the model can act on."""

from types import SimpleNamespace as NS

import pytest

from peresearch import guard
from peresearch.messages import Tool
from peresearch.registry import Budget, Kept, RateLimited, Registry, ToolSpec

SCHEMA = {"type": "object", "required": ["pattern"],
          "properties": {"pattern": {"type": "string"}, "n": {"type": "integer"}, "langs": {"type": "array"}}}


def spec(name="grep", group="files", run=None, budget="", offered=lambda: True, store=True, parameters=SCHEMA):
    ran = []

    def default(**kw):
        ran.append(kw)
        return f"{name} ran with {sorted(kw.items())}"
    s = ToolSpec(name, f"the {name} tool", parameters, group, run or default, budget=budget, offered=offered,
                 store=store)
    return s, ran


class Clock:
    def __init__(self):
        self.now, self.slept = 100.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def registry(*specs, web_on=lambda: True, clock=None):
    clock = clock or Clock()
    return Registry(list(specs), web_on=web_on, sleep=clock.sleep, clock=clock)


# --- stage 1: the arguments ---

def test_arguments_are_read_by_their_declared_type_and_unknown_or_null_ones_dropped():
    s, ran = spec()
    out = registry(s).run("grep", {"pattern": "x", "n": "10", "other": 1, "langs": None})
    assert ran == [{"pattern": "x", "n": 10}] and out.store and out.text.startswith("grep ran")


def test_a_bad_or_missing_argument_is_said_and_the_tool_does_not_run():
    s, ran = spec()
    r = registry(s)
    assert r.run("grep", {"pattern": "x", "n": "ten"}).text == "bad arguments for grep: n must be a whole number (got 'ten')"
    assert r.run("grep", {"n": 2}).text == "bad arguments for grep: pattern is required"
    assert ran == [] and not r.run("grep", {}).store and r.run("grep", {}).refused == "arguments"


def test_a_schema_without_properties_passes_the_arguments_as_given():
    s, ran = spec(parameters={})
    registry(s).run("grep", {"q": 1, "r": [1]})
    assert ran == [{"q": 1, "r": [1]}]


def test_an_unknown_tool_is_said():
    out = registry(spec()[0]).run("nope", {})
    assert out.text == "unknown tool 'nope'" and out.refused == "tool" and not out.store


# --- stage 2: permission ---

def test_a_web_tool_is_refused_when_the_web_is_off_and_one_not_offered_is_refused_by_name():
    web, ran = spec("web_search", group="web", parameters={})
    absent, ran2 = spec("search_papers", group="web", parameters={}, offered=lambda: False)
    on = {"value": False}
    r = registry(web, absent, web_on=lambda: on["value"])
    assert r.run("web_search", {}).text == "refused: the user turned the web off for this session (local only)"
    assert r.run("web_search", {}).refused == "permission"
    on["value"] = True
    assert r.run("search_papers", {}).text == "refused: search_papers is not available in this session"
    assert ran == ran2 == []
    assert r.run("web_search", {}).store and ran == [{}]


def test_only_offered_tools_are_listed_and_web_ones_only_while_the_web_is_on():
    files, _ = spec("grep")
    web, _ = spec("web_search", group="web")
    absent, _ = spec("fetch", group="web", offered=lambda: False)
    on = {"value": True}
    r = registry(files, web, absent, web_on=lambda: on["value"])
    assert r.tools() == [Tool("grep", "the grep tool", SCHEMA), Tool("web_search", "the web_search tool", SCHEMA)]
    on["value"] = False
    assert [t.name for t in r.tools()] == ["grep"]


def test_a_group_outside_the_known_ones_is_refused_when_declared():
    with pytest.raises(AssertionError):
        spec(group="shell")


# --- stage 3: the budget ---

def test_a_repeated_call_is_refused_only_within_a_question_budget():
    s, ran = spec()
    r, budget = registry(s), Budget()
    assert r.run("grep", {"pattern": "x"}, budget).store
    again = r.run("grep", {"pattern": "x"}, budget)
    assert again.text == "already called with the same arguments; its result is above. Use it, or answer."
    assert again.refused == "budget" and not again.store
    assert r.run("grep", {"pattern": "y"}, budget).store
    r.run("grep", {"pattern": "x"})                                   # no budget (a direct call): run again
    assert [kw["pattern"] for kw in ran] == ["x", "y", "x"]


def test_searches_are_limited_only_when_the_user_set_a_limit():
    s, ran = spec("web_search", group="web", budget="search", parameters={})
    r = registry(s)
    unlimited = Budget()
    assert all(r.run("web_search", {"q": i}, unlimited).store for i in range(30))
    limited = Budget(searches=2)
    assert [r.run("web_search", {"q": i}, limited).text.startswith("web_search ran") for i in range(3)] == [True, True, False]
    assert r.run("web_search", {"q": 9}, limited).text == "limit: at most 2 web searches per question"


def test_pages_count_new_addresses_only():
    s, ran = spec("fetch", group="web", budget="page", parameters={})
    r, budget = registry(s), Budget(pages=1)
    assert r.run("fetch", {"url": "https://a"}, budget).store
    assert r.run("fetch", {"url": "https://a", "part": 2}, budget).store          # the same page: not another
    assert r.run("fetch", {"url": "https://b"}, budget).text == "limit: at most 1 pages per question"
    assert r.run("fetch", {"q": 1}, budget).text == "limit: at most 1 pages per question"   # no address: a new page


# --- stage 4: the run ---

def test_errors_become_text_for_the_model_and_a_failed_check_is_raised():
    def refuse(**kw):
        raise guard.Refused("outside the declared folders")

    def fail(**kw):
        raise RuntimeError("x" * 300)

    def broken(**kw):
        raise AssertionError("a bug")
    r = registry(spec("a", run=refuse, parameters={})[0], spec("b", run=fail, parameters={})[0],
                 spec("c", run=broken, parameters={})[0])
    assert r.run("a", {}).text == "refused: outside the declared folders"
    assert r.run("b", {}).text == "b failed: RuntimeError: " + "x" * 200
    with pytest.raises(AssertionError):
        r.run("c", {})


def test_on_run_is_told_just_before_a_tool_runs_and_never_for_a_refused_call():
    s, _ = spec()
    told = []
    r, budget = registry(s), Budget()
    r.run("grep", {"pattern": "x"}, budget, on_run=lambda name, args: told.append((name, args)))
    r.run("grep", {"pattern": "x"}, budget, on_run=lambda name, args: told.append((name, args)))
    r.run("grep", {}, budget, on_run=lambda name, args: told.append((name, args)))
    assert told == [("grep", {"pattern": "x"})]


def test_a_kept_result_passes_through_and_a_tool_that_keeps_nothing_is_not_stored():
    recall, _ = spec("recall", run=lambda **kw: Kept("R2", "kept text", 5), store=False, parameters={})
    wrong, _ = spec("recall2", run=lambda **kw: "no result R9", store=False, parameters={})
    r = registry(recall, wrong)
    out = r.run("recall", {"ref": "R2"})
    assert out.kept == Kept("R2", "kept text", 5) and out.text == "kept text" and not out.store
    assert r.run("recall2", {}).text == "no result R9" and not r.run("recall2", {}).store


# --- rate limits (HTTP 429), the same for every tool ---

def limited(times, error):
    calls = []

    def run(**kw):
        calls.append(kw)
        if len(calls) <= times:
            raise error
        return "answered"
    return run, calls


def http_429(retry_after=None):
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = NS(status_code=429, headers=headers)
    return type("HTTPStatusError", (Exception,), {})("429 Too Many Requests"), response


def test_a_rate_limited_tool_is_waited_out_as_its_service_asks():
    clock = Clock()
    run, calls = limited(1, RateLimited("Semantic Scholar", retry_after=3.0))
    r = registry(spec("search_papers", run=run, parameters={})[0], clock=clock)
    assert r.run("search_papers", {}).text == "answered" and clock.slept == [3.0] and len(calls) == 2


def test_an_http_429_from_any_client_is_read_with_its_retry_after_header():
    clock = Clock()
    error, response = http_429("2")
    error.response = response
    run, calls = limited(2, error)
    r = registry(spec("fetch", run=run, parameters={})[0], clock=clock)
    assert r.run("fetch", {}).text == "answered" and clock.slept == [2.0, 2.0]


def test_without_retry_after_the_waits_double_and_three_attempts_end_it():
    clock = Clock()
    run, calls = limited(9, RateLimited("Wikipedia"))
    r = registry(spec("search_wikipedia", run=run, parameters={})[0], clock=clock)
    assert r.run("search_wikipedia", {}).text == "rate limited by Wikipedia: try again later or use another source"
    assert clock.slept == [1.0, 2.0] and len(calls) == 3


def test_a_wait_past_the_question_deadline_or_too_long_is_not_waited():
    clock = Clock()
    run, calls = limited(9, RateLimited("Semantic Scholar", retry_after=30.0))
    r = registry(spec("search_papers", run=run, parameters={})[0], clock=clock)
    budget = Budget(deadline=clock.now + 10)
    assert r.run("search_papers", {}, budget).text.startswith("rate limited by Semantic Scholar")
    assert clock.slept == [] and len(calls) == 1
    run, calls = limited(9, RateLimited("x", retry_after=3600.0))     # an hour: longer than any wait
    r = registry(spec("search_papers", run=run, parameters={})[0], clock=clock)
    assert r.run("search_papers", {}).text.startswith("rate limited by x") and clock.slept == []


# --- a tool from an MCP server goes through the same pipeline ---

def test_a_tool_listed_by_an_mcp_server_is_checked_like_a_built_in_one():
    listing = {"name": "lookup", "description": "Look a term up.",
               "inputSchema": {"type": "object", "required": ["term"],
                               "properties": {"term": {"type": "string"}, "limit": {"type": "integer"}}}}
    sent = []
    s = ToolSpec(f"server_{listing['name']}", listing["description"], listing["inputSchema"], "web",
                 lambda **kw: sent.append(kw) or "found")
    r, budget = registry(s), Budget()
    assert r.run("server_lookup", {"limit": "3"}, budget).text == "bad arguments for server_lookup: term is required"
    assert r.run("server_lookup", {"term": "BPE", "limit": "3"}, budget).text == "found"
    assert r.run("server_lookup", {"term": "BPE", "limit": "3"}, budget).text.startswith("already called")
    assert sent == [{"term": "BPE", "limit": 3}]
    assert registry(s, web_on=lambda: False).run("server_lookup", {"term": "x"}).text.startswith("refused:")


# --- edges the mutation review of 2026-10-21 found untested ---

def test_whole_numbers_keep_every_digit_and_outcomes_say_plainly_what_happened():
    from peresearch.registry import whole
    assert whole("9007199254740993") == 9007199254740993                  # past a float's precision: exact
    s, _ = spec()
    ran = registry(s).run("grep", {"pattern": "x"})
    assert (ran.store, ran.kept, ran.refused) == (True, None, "")
    refused = registry(s).run("grep", {})
    assert (refused.store, refused.kept, refused.refused) == (False, None, "arguments")


def test_a_schema_may_declare_no_required_list_or_require_an_undeclared_argument():
    loose, ran = spec(parameters={"type": "object", "properties": {"q": {"type": "string"}}})
    assert registry(loose).run("grep", {"q": "x"}).store and ran == [{"q": "x"}]
    odd, _ = spec(parameters={"type": "object", "required": ["term"], "properties": {"q": {"type": "string"}}})
    assert registry(odd).run("grep", {"q": "x"}).text == "bad arguments for grep: term is required"


def test_pages_are_not_limited_by_default():
    s, ran = spec("fetch", group="web", budget="page", parameters={})
    r, budget = registry(s), Budget()
    assert all(r.run("fetch", {"url": f"https://u/{i}"}, budget).store for i in range(40)) and len(ran) == 40


def test_the_longest_wait_and_the_deadline_are_inclusive():
    clock = Clock()
    run, calls = limited(1, RateLimited("s", retry_after=60.0))           # exactly the longest wait: waited
    assert registry(spec("a", run=run, parameters={})[0], clock=clock).run("a", {}).text == "answered"
    run, calls = limited(1, RateLimited("s", retry_after=60.5))
    assert registry(spec("a", run=run, parameters={})[0], clock=clock).run("a", {}).text.startswith("rate limited")
    run, calls = limited(1, RateLimited("s", retry_after=5.0))            # ends exactly at the deadline: waited
    budget = Budget(deadline=clock.now + 5.0)
    assert registry(spec("a", run=run, parameters={})[0], clock=clock).run("a", {}, budget).text == "answered"
    run, calls = limited(1, RateLimited("s", retry_after=5.0))            # a budget with no deadline: waited
    assert registry(spec("a", run=run, parameters={})[0], clock=clock).run("a", {}, Budget()).text == "answered"


def test_a_429_names_the_host_it_came_from_and_reads_any_retry_after():
    clock = Clock()
    error, _ = http_429()
    error.response = NS(status_code=429)                                  # no headers at all
    error.request = NS(url=NS(host="api.semanticscholar.org"))
    run, calls = limited(9, error)
    out = registry(spec("search_papers", run=run, parameters={})[0], clock=clock).run("search_papers", {})
    assert out.text == "rate limited by api.semanticscholar.org: try again later or use another source"
    assert clock.slept == [1.0, 2.0]
    error, response = http_429("Wed, 21 Oct 2026 07:28:00 GMT")          # an HTTP date: waited as if unsaid
    error.response, error.request = response, NS(url=NS())               # an address without a host
    run, calls = limited(1, error)
    clock = Clock()
    assert registry(spec("fetch", run=run, parameters={})[0], clock=clock).run("fetch", {}).text == "answered"
    assert clock.slept == [1.0]
    error, response = http_429()
    error.response = response                                             # no request: the tool's name
    run, calls = limited(9, error)
    assert registry(spec("fetch", run=run, parameters={})[0], clock=Clock()).run("fetch", {}).text.startswith(
        "rate limited by fetch:")
