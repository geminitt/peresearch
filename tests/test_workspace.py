"""Contracts of the declared folders (peresearch.workspace) and the settings (peresearch.settings)."""

import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from hypothesis import given, settings as hsettings
from hypothesis import strategies as st

from peresearch import guard, settings
from peresearch.workspace import PROBLEMS, Workspace, complete, summary


@given(st.lists(st.tuples(st.sampled_from(["add", "remove"]), st.sampled_from(["a", "a/b", "a/b/c", "d", "a/../a/b"])),
                max_size=8))
@hsettings(deadline=None, max_examples=80)
def test_the_declared_folders_never_overlap_and_a_parent_absorbs_its_children(tmp_path_factory, ops):
    base = tmp_path_factory.mktemp("w")
    home = base / "home"
    for d in ("a/b/c", "d"):
        (base / d).mkdir(parents=True, exist_ok=True)
    saved, os.environ["PERESEARCH_HOME"] = os.environ.get("PERESEARCH_HOME"), str(home)
    try:
        ws, model = Workspace(index=None), []
        for op, rel in ops:
            p = (base / rel).resolve()
            if op == "remove":
                assert ws.remove(str(base / rel)) == (p in model)
                model = [r for r in model if r != p]
                continue
            inside = next((r for r in model if r == p or r in p.parents), None)
            if inside:
                with pytest.raises(ValueError, match=f"already inside the declared folder {inside}"):
                    ws.add(str(base / rel))
            else:
                assert ws.add(str(base / rel)) == p
                model = [r for r in model if p not in r.parents] + [p]
            assert ws.folders() == sorted(model, key=str)
            assert not any(r != q and r in q.parents for r in model for q in model)
    finally:
        os.environ.pop("PERESEARCH_HOME") if saved is None else os.environ.__setitem__("PERESEARCH_HOME", saved)


def test_a_folder_that_is_not_one_protected_or_peresearchs_own_is_refused(home, tmp_path):
    (tmp_path / ".ssh").mkdir()
    (tmp_path / "f.md").write_text("x")
    home.mkdir(parents=True, exist_ok=True)
    (home / "sub").mkdir()
    ws = Workspace(index=None)
    for path, why in ((tmp_path / "nope", "not a folder"), (tmp_path / "f.md", "not a folder"),
                      (tmp_path / ".ssh", "matches a protected pattern"), (home, "peresearch's own data"),
                      (home / "sub", "peresearch's own data")):
        with pytest.raises(ValueError, match=why):
            ws.add(str(path))
    assert ws.folders() == []


def test_the_index_is_made_when_first_needed_and_updated_over_the_declared_folders(home, tmp_path):
    calls = []
    ws = Workspace(index=NS(update=lambda: calls.append("update") or "report"))
    assert ws.update() == "report" and calls == ["update"]
    from peresearch.zetokrag.index import Index
    lazy = Workspace()
    assert isinstance(lazy.index, Index) and lazy.index is lazy.index and lazy.index.home == home


def test_the_summary_lists_counts_problems_and_withheld_files():
    r = NS(added=2, changed=1, unchanged=7, removed=0, chunks=40, denied=3,
           problems={"too_large": ["big.pdf"], "duplicate": ["b (same as a)"], "odd": ["x"]},
           withheld={"keys.md": 2, "env.md": 1})
    assert summary(r).splitlines() == [
        "2 added · 1 changed · 7 unchanged · 0 removed · 40 chunks · 3 protected paths skipped",
        f"1 {PROBLEMS['duplicate']}:", "  b (same as a)", "1 odd:", "  x", f"1 {PROBLEMS['too_large']}:", "  big.pdf",
        "3 chunks withheld because they look like credentials:", "  keys.md (2)", "  env.md (1)"]
    many = NS(**{**vars(r), "problems": {"error": [f"f{i}" for i in range(30)]}, "withheld": {}})
    assert len(summary(many, limit=5).splitlines()) == 1 + 1 + 5


def test_completion_offers_folders_as_a_shell_would(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for d in ("projects", "papers", ".hidden", "pics"):
        (tmp_path / d).mkdir()
    (tmp_path / "plan.md").write_text("x")
    assert complete(str(tmp_path) + "/p") == [str(tmp_path / n) + "/" for n in ("papers", "pics", "projects")]
    assert complete("~/pa") == ["~/papers/"]
    assert complete("~") == ["~/papers/", "~/pics/", "~/projects/"]                 # hidden only when asked
    assert complete("~/.h") == ["~/.hidden/"]
    assert complete("~/pl") == [] and complete("~/pl", folders_only=False) == ["~/plan.md/"]
    assert complete(str(tmp_path / "nothing" / "x")) == [] and complete(None) == complete("")
    assert complete(str(tmp_path) + "/") == [str(tmp_path / n) + "/" for n in ("papers", "pics", "projects")]


def test_a_setting_comes_from_the_environment_then_the_saved_file_then_the_default(home, monkeypatch):
    monkeypatch.delenv("PERESEARCH_LLM_MODEL", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert settings.saved() == {} and settings.get("PERESEARCH_LLM_MODEL") == "llm" and settings.get("TAVILY_API_KEY") == ""
    home.mkdir(parents=True, exist_ok=True)
    (home / "settings.env").write_text("# a comment\n#EXA_API_KEY=old\n\n TAVILY_API_KEY = 'tvly-a=b' \n"
                                       "PERESEARCH_LLM_MODEL=\"qwen\"\nbroken line\n")
    assert settings.saved() == {"TAVILY_API_KEY": "tvly-a=b", "PERESEARCH_LLM_MODEL": "qwen"}
    assert settings.get("TAVILY_API_KEY") == "tvly-a=b" and settings.get("PERESEARCH_LLM_MODEL") == "qwen"
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    assert settings.get("TAVILY_API_KEY") == "from-env"
    assert guard.denied(home / "settings.env")                                      # never indexed or read
