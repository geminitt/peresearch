"""Contracts of the model profile (peresearch.profiles).

- The table is models.dev trimmed: per normalized name, tool calls and reasoning when any provider says so,
  temperature refused only when all refuse it, the largest limits, the most common family (a property against a
  reference).
- A served name finds its row however a server spells it: an organization, a tag, an ending, dots for dashes
  (a property); a name the table does not know finds none.
- A profile is built in layers — defaults, the table, the family, the user's settings — and says where each field
  came from; a bad setting is refused with its name.
- The shipped table says where it came from and knows the models peresearch is checked with.
"""

import json
from collections import Counter

import pytest
from hypothesis import example, given, settings as hsettings
from hypothesis import strategies as st

from peresearch import profiles
from peresearch.profiles import Profile, build, lookup, norm, profile

ids = st.from_regex(r"[a-z][a-z0-9]{1,6}(-[a-z0-9]{1,4}){0,2}", fullmatch=True)
entry = st.fixed_dictionaries({}, optional={
    "family": st.sampled_from(["qwen", "llama", "gemma", ""]), "tool_call": st.booleans(), "reasoning": st.booleans(),
    "temperature": st.booleans(),
    "limit": st.fixed_dictionaries({}, optional={"context": st.integers(1, 10 ** 6), "output": st.integers(1, 10 ** 5)})})
raws = st.dictionaries(st.sampled_from(["p1", "p2", "p3"]),
                       st.fixed_dictionaries({"models": st.dictionaries(
                           st.tuples(st.sampled_from(["", "org/", "Org/"]), ids).map("".join), entry, max_size=4)}),
                       max_size=3)


@given(raws)
@hsettings(max_examples=200)
@example({"p1": {"models": {"/": {"tool_call": True}, "zeta-1": {}}}})     # a name that keeps nothing comes first
def test_the_table_is_models_dev_trimmed_by_one_rule(raw):
    seen = {}
    for provider in raw.values():
        for name, m in provider["models"].items():
            if norm(name):
                seen.setdefault(norm(name), []).append(m)
    want = {}
    for key, ms in seen.items():
        fam = Counter(m.get("family") or "" for m in ms).most_common(1)[0][0]
        want[key] = [fam, any(m.get("tool_call", False) for m in ms), any(m.get("reasoning", False) for m in ms),
                     any(m.get("temperature", True) for m in ms),
                     max((m.get("limit") or {}).get("context") or 0 for m in ms) or None,
                     max((m.get("limit") or {}).get("output") or 0 for m in ms) or None]
    assert build(raw) == want


rows = st.dictionaries(ids, st.just(["f", True, False, True, None, None]), min_size=1, max_size=8)


@given(rows, st.data())
@example({"llama-3-1-8b": ["llama", True, False, True, 128000, 8192]}, None)
def test_a_served_name_finds_its_row_however_a_server_spells_it(table, data):
    if data is None:                                      # Ollama's spelling of a models.dev name
        assert lookup("llama3.1:8b", table) == ("llama-3-1-8b", table["llama-3-1-8b"])
        assert lookup("meta-llama/Llama-3.1-8B-Instruct", table)[0] == "llama-3-1-8b"
        # a chat-tuned row beside the base one is the served model's: Ollama's "llama3.2:1b" is the instruct model
        # (it calls tools natively, recorded 2026-10-06) while models.dev's base row says no tool calls
        chat = {"llama-3-2-1b": ["llama", False, False, True, 131072, 131072],
                "llama-3-2-1b-instruct": ["llama", True, True, True, 131072, 60000],
                "gemma-3-4b": ["gemma", False, False, True, 1, 1], "gemma-3-4b-it": ["gemma", True, False, True, 2, 2],
                "phi-4": ["phi", False, False, True, 1, 1], "phi-4-chat": ["phi", True, False, True, 3, 3]}
        assert [lookup(n, chat)[0] for n in ("llama3.2:1b", "Llama-3.2-1B", "gemma3:4b", "phi-4", "llama-3-2-1b-instruct")] \
            == ["llama-3-2-1b-instruct", "llama-3-2-1b-instruct", "gemma-3-4b-it", "phi-4-chat", "llama-3-2-1b-instruct"]
        # a GGUF pulled from Hugging Face through Ollama has two slashes and a quantization tag
        assert lookup("hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", chat)[0] == "llama-3-2-1b-instruct"
        only_it = {"gemma-3-4b-it": ["gemma", True, False, True, 1, 1]}           # no base row: found without dashes
        assert lookup("gemma3:4b", only_it)[0] == "gemma-3-4b-it"
        spelled = {"gemma-3-4b-it": ["gemma", True, False, True, 1, 1], "gemma3-4b": ["gemma", False, False, True, 2, 2]}
        assert lookup("Gemma 3 4B", spelled)[0] == "gemma3-4b"                  # the shortest key of one spelling
        return
    key = data.draw(st.sampled_from(sorted(table)))
    org = data.draw(st.sampled_from(["", "org/", "Some-Org/"]))
    ending = data.draw(st.sampled_from(["", "-instruct", ":latest", "-FP8", "-instruct-awq", "-q4_K_M"]))
    spelled = org + data.draw(st.sampled_from([key, key.upper(), key.replace("-", ".", 1), key.replace("-", "_")]))
    found = lookup(spelled + ending, table)
    longer = [k for k in table if k != key and norm(spelled + ending).startswith(k)]
    chat = {k + e for k in (key, *longer) for e in profiles.CHAT_ENDINGS}      # the chat-tuned row beside it
    assert found is not None and (found[0] == key or found[0] in longer or norm(found[0]) == norm(key) or found[0] in chat)
    assert lookup("zz" + key + "zz9", table) is None or lookup("zz" + key + "zz9", table)[0].startswith("zz")


QWEN = ["qwen", True, True, True, 262144, 81920]
TABLE = {"qwen3-6-35b-a3b": QWEN, "gemma-3-27b": ["gemma", False, False, True, 128000, 8192],
         "o-reasoner": ["qwen", True, True, False, 200000, 100000]}


@pytest.mark.parametrize("name, settings, want, source", [
    ("unknown", {}, Profile(model="unknown"), {}),
    ("Qwen/Qwen3.6-35B-A3B", {}, Profile(model="Qwen/Qwen3.6-35B-A3B", family="qwen", tools="native", reasoning=True,
                                         thinking="template", temperature=0.6, top_p=0.95, context=262144, output=81920),
     {"family": "table", "tools": "table", "reasoning": "table", "context": "table", "output": "table",
      "thinking": "family", "temperature": "family", "top_p": "family"}),
    ("gemma3:27b", {}, Profile(model="gemma3:27b", family="gemma", tools="prompt", context=128000, output=8192),
     {"family": "table", "tools": "table", "reasoning": "table", "context": "table", "output": "table"}),
    ("o-reasoner", {}, Profile(model="o-reasoner", family="qwen", tools="native", reasoning=True, thinking="template",
                               context=200000, output=100000),            # temperature refused: the family's not forced
     {"family": "table", "tools": "table", "reasoning": "table", "context": "table", "output": "table",
      "temperature": "table", "thinking": "family"}),
    ("gemma3:27b", {"PERESEARCH_LLM_TOOLS": "native", "PERESEARCH_LLM_TEMPERATURE": "0.2", "PERESEARCH_LLM_TOP_P": "1"},
     Profile(model="gemma3:27b", family="gemma", tools="native", temperature=0.2, top_p=1.0, context=128000, output=8192),
     {"family": "table", "tools": "setting", "reasoning": "table", "context": "table", "output": "table",
      "temperature": "setting", "top_p": "setting"}),
    ("unknown", {"PERESEARCH_LLM_TOOLS": "prompt", "PERESEARCH_LLM_TEMPERATURE": "0", "PERESEARCH_LLM_TOP_P": "0"},
     Profile(model="unknown", tools="prompt", temperature=0.0, top_p=0.0),      # 0: greedy, the lowest accepted
     {"tools": "setting", "temperature": "setting", "top_p": "setting"})])
def test_a_profile_is_built_in_layers_and_says_where_each_field_came_from(monkeypatch, home, name, settings, want,
                                                                         source):
    for k in ("PERESEARCH_LLM_TOOLS", "PERESEARCH_LLM_TEMPERATURE", "PERESEARCH_LLM_TOP_P"):
        monkeypatch.delenv(k, raising=False)
    for k, v in settings.items():
        monkeypatch.setenv(k, v)
    got = profile(name, TABLE)
    assert got == Profile(**{**want.__dict__, "source": source}) and got.source == source


@pytest.mark.parametrize("name, value, message", [
    ("PERESEARCH_LLM_TOOLS", "xml", "PERESEARCH_LLM_TOOLS must be native or prompt"),
    ("PERESEARCH_LLM_TEMPERATURE", "hot", "PERESEARCH_LLM_TEMPERATURE must be a number from 0 to 2"),
    ("PERESEARCH_LLM_TEMPERATURE", "2.5", "PERESEARCH_LLM_TEMPERATURE must be a number from 0 to 2"),
    ("PERESEARCH_LLM_TOP_P", "-0.1", "PERESEARCH_LLM_TOP_P must be a number from 0 to 1"),
    ("PERESEARCH_LLM_TOP_P", "nan", "PERESEARCH_LLM_TOP_P must be a number from 0 to 1")])
def test_a_bad_setting_is_refused_with_its_name(monkeypatch, home, name, value, message):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f"^{message}") as e:
        profile("qwen3-8b", TABLE)
    assert str(e.value).endswith(f"(got {value!r})")


def test_a_name_is_normalized_as_a_server_writes_it():
    assert [norm(n) for n in (" Qwen/Qwen3.6-35B-A3B ", "-qwen3:8b-", "hf.co/org/Name_v2:Q4", "", None)] \
        == ["qwen3-6-35b-a3b", "qwen3-8b", "name-v2-q4", "", ""]


def test_the_shipped_table_says_where_it_came_from_and_knows_the_checked_models(monkeypatch):
    monkeypatch.setattr(profiles, "_loaded", None)                 # read from the file, then kept
    monkeypatch.setattr(profiles, "_compact", None)
    assert lookup("llama3.2:1b")[0] == lookup("llama3.2:1b")[0] == "llama-3-2-1b-instruct"
    assert profiles._loaded is not None and profiles._compact is not None
    shipped = json.loads(profiles.TABLE.read_text())
    assert shipped["source"] == "https://models.dev/api.json" and len(shipped["sha256"]) == 64
    assert shipped["fields"] == ["family", "tool_call", "reasoning", "temperature", "context", "output"]
    assert (profiles.TABLE.parent / "MODELS-DEV-LICENSE").read_text().startswith("MIT License")
    assert all(k == norm(k) and len(row) == 6 for k, row in shipped["models"].items())
    for served, family in (("Qwen/Qwen3.6-35B-A3B-FP8", "qwen"), ("Qwen/Qwen3.5-0.8B", "qwen"), ("llama3.2:1b", "llama")):
        found = lookup(served)
        assert found is not None and found[1][0] == family and found[1][1] is True
