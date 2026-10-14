"""ResultStore lookups: one line read per ref, whatever the conversation's length (2026-10-14).

get() used to read and parse the whole sessions/<project>/<conversation>.jsonl for every recall and every answer's
source check, so a long conversation with large results (a broad grep stores 79–114 MiB) slowed each step down.
What is kept does not change: no result is cut or dropped (no fixed limit)."""

from pathlib import Path

import pytest

from peresearch.agent import ResultStore, _append_private


@pytest.fixture
def store(home):
    return ResultStore("default", "c-one")


def test_a_lookup_reads_one_line_not_the_whole_file(store, monkeypatch):
    for n in range(3):
        store.put("grep", {"pattern": str(n)}, f"result {n} " + "x" * 1000)
    assert store.get("R1")["text"].startswith("result 0")          # the index is built here, by one pass
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("the whole file was read again"))
    assert store.get("R3")["text"].startswith("result 2") and store.get("R2")["args"] == {"pattern": "1"}
    assert store.get("R9") is None


def test_what_another_store_appends_is_found(store):
    store.put("read", {"path": "a"}, "first")
    other = ResultStore("default", "c-one")                          # e.g. the interface resuming this conversation
    assert other.get("R1")["text"] == "first"
    store.put("read", {"path": "b"}, "second")
    assert other.get("R2")["text"] == "second"
    other.put("read", {"path": "c"}, "third")                       # numbering goes on from the file
    assert store.get("R3")["text"] == "third"


def test_a_scope_recorded_after_a_lookup_is_returned_with_its_result(store):
    store.put("grep", {"pattern": "x"}, "found")
    assert "scope" not in store.get("R1")
    store.set_scope("R1", ["/notes"])
    assert store.get("R1")["scope"] == ["/notes"]


def test_a_line_a_crash_cut_is_skipped_and_the_next_one_is_found(store):
    store.put("read", {"path": "a"}, "whole")
    with store.path.open("ab") as f:
        f.write(b'{"ref": "R2", "text": "cut sho')                    # no end: a power cut mid-line
    assert store.get("R1")["text"] == "whole" and store.get("R2") is None
    _append_private(store.path, '{"ref": "R2", "time": 0, "tool": "read", "args": {}, "text": "after"}')
    assert store.get("R2")["text"] == "after"


def test_a_replaced_file_is_indexed_again(store):
    store.put("read", {"path": "a"}, "a long first result " * 20)
    assert store.get("R1")["text"].startswith("a long")
    store.path.write_text('{"ref": "R1", "time": 0, "tool": "read", "args": {}, "text": "new"}\n')
    assert store.get("R1")["text"] == "new"


def test_nothing_kept_is_cut(store):
    big = "line of a broad grep\n" * 200_000                         # about 4 MB: kept and read back whole
    store.put("grep", {"pattern": "line"}, big)
    assert store.get("R1")["text"] == big


def test_a_second_lookup_parses_only_its_own_line(store, monkeypatch):
    """Found by mutation (2026-10-14): a store that indexed the whole file again on every lookup passed the tests."""
    import json
    for n in range(5):
        store.put("grep", {"pattern": str(n)}, f"result {n}")
    store.get("R1")
    parsed, loads = [], json.loads
    monkeypatch.setattr(json, "loads", lambda raw, *a, **k: parsed.append(raw) or loads(raw, *a, **k))
    assert store.get("R4")["text"] == "result 3" and store.get("R2")["text"] == "result 1"
    assert len(parsed) == 2                                           # one line each, nothing scanned again
    store.put("grep", {"pattern": "5"}, "result 5")
    parsed.clear()
    assert store.get("R6")["text"] == "result 5" and len(parsed) == 2   # the new line indexed, then read


def test_a_file_put_in_place_of_the_indexed_one_is_indexed_again(store):
    """Found by mutation (2026-10-14): replacing the file with a longer one, R1 no longer first, went unnoticed."""
    import os
    store.put("read", {"path": "a"}, "old")
    assert store.get("R1")["text"] == "old"
    other = store.path.with_name("other.jsonl")
    other.write_text('{"ref": "R0", "time": 0, "tool": "read", "args": {}, "text": "' + "z" * 200 + '"}\n'
                     '{"ref": "R1", "time": 0, "tool": "read", "args": {}, "text": "new"}\n')
    os.replace(other, store.path)                                    # another file (inode), longer than the first
    assert store.get("R1")["text"] == "new" and store.get("R0")["text"] == "z" * 200
