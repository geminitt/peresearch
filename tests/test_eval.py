"""Contracts of the evaluation arithmetic (eval/retrieval.py, eval/research.py,
eval/tokenization.py, eval/gold.py): every metric against its textbook definition, the confidence intervals against
the binomial tails they solve, the work split, the retries, the run files that let an interrupted evaluation
resume, and the gold ruler (judged lines reached, near-copies, the paired bootstrap)."""

import hashlib
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

EVAL = Path(__file__).parent.parent / "eval"
sys.path.insert(0, str(EVAL))


def module(name):
    spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


retrieval, research, tokenization = (module(n) for n in ("retrieval", "research", "tokenization"))

docs = st.lists(st.sampled_from([f"d{i}" for i in range(30)]), unique=True, max_size=25)
grades = st.dictionaries(st.sampled_from([f"d{i}" for i in range(30)]), st.integers(1, 3), min_size=1, max_size=12)


# --- ranking metrics ----------------------------------------------------------------------------------------------

@given(docs, grades)
@example(["x", "a", "y", "b"], {"a": 2, "b": 1, "c": 1})
def test_ranking_metrics_follow_their_definitions(ranked, rel):
    m = retrieval.metrics(ranked, rel)
    dcg = sum(rel.get(d, 0) / math.log2(r + 1) for r, d in enumerate(ranked[:10], start=1))
    best = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(r + 1) for r, g in enumerate(best, start=1))
    first = next((r for r, d in enumerate(ranked[:10], start=1) if d in rel), None)
    assert math.isclose(m["ndcg@10"], dcg / idcg) and 0 <= m["ndcg@10"] <= 1 + 1e-12
    assert m["recall@10"] == len([d for d in ranked[:10] if d in rel]) / len(rel)
    assert m["recall@100"] == len([d for d in ranked[:100] if d in rel]) / len(rel)
    assert m["mrr@10"] == (1 / first if first else 0.0)


@given(st.lists(st.sampled_from([0.1, 0.2, 0.5, 0.9]), min_size=1, max_size=10),
       st.lists(st.sampled_from([0.1, 0.2, 0.5, 0.9]), min_size=1, max_size=10))
def test_auroc_is_the_chance_an_answerable_score_beats_an_unanswerable_one(pos, neg):
    pairs = [(p > n) + 0.5 * (p == n) for p in pos for n in neg]
    a = retrieval.auroc(pos, neg)
    assert math.isclose(a, sum(pairs) / len(pairs)) and math.isclose(a + retrieval.auroc(neg, pos), 1.0)


# --- confidence intervals -------------------------------------------------------------------------------------------

def tail(n, p, lo, hi):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(lo, hi + 1))


@given(st.integers(1, 80).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n))))
@settings(max_examples=60)
@example((25, 25))                                     # the rule of three's exact form: 0.025^(1/25)
@example((5, 10))                                      # textbook Clopper-Pearson 0.1871–0.8129
def test_the_exact_interval_solves_the_binomial_tails(kn):
    k, n = kn
    lo, hi = research.exact_interval(k, n)
    assert lo <= k / n <= hi
    assert lo == 0.0 if k == 0 else math.isclose(tail(n, lo, k, n), 0.025, abs_tol=1e-9)   # P(X >= k | lo)
    assert hi == 1.0 if k == n else math.isclose(tail(n, hi, 0, k), 0.025, abs_tol=1e-9)   # P(X <= k | hi)
    if k < n:
        assert research.exact_interval(k + 1, n)[0] > lo                  # more successes, higher bounds
    assert research.exact_interval(0, 0) == (0.0, 1.0)


# --- evidence in an answer --------------------------------------------------------------------------------------

spans = st.tuples(st.sampled_from(["a.md", "b.md"]), st.sampled_from(["line", "page"]), st.integers(1, 20),
                  st.integers(0, 6)).map(lambda t: (t[0], t[1], t[2], t[2] + t[3]))


@given(st.lists(spans, max_size=12), st.lists(spans, min_size=1, max_size=4))
def test_evidence_is_found_when_a_cited_file_overlaps_a_labelled_passage(rows, labels):
    relevant = [{"path": p, "unit": u, "start": a, "end": b} for p, u, a, b in labels]
    files = {k: f"{p}:{'L' if u == 'line' else 'p.'}{a}-{b}" for k, (p, u, a, b) in enumerate(rows)}
    files["web"] = "https://example.org/a.md:L1-5"                     # a web source never counts
    overlap = any(r["path"] == p and r["unit"] == u and a <= r["end"] and r["start"] <= b
                  for p, u, a, b in rows for r in relevant)
    assert research.evidence_found(files, relevant) == overlap


# --- the work split and retries ---------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_shards_cover_every_corpus_once_and_split_the_minutes_evenly(n):
    shards = [retrieval.shard(k, n) for k in range(n)]
    assert sorted(c for s in shards for c in s) == sorted(retrieval.DATASETS)
    loads = [sum(retrieval.COST[c] for c in s) for s in shards]
    assert max(loads) - min(loads) <= max(retrieval.COST.values())       # greedy largest-first bound
    assert all(s == [c for c in retrieval.DATASETS if c in s] for s in shards)   # in the corpora's own order
    assert set(retrieval.REVISIONS) == set(retrieval.DATASETS) and all(len(r) == 40 for r in retrieval.REVISIONS.values())
    assert len(retrieval.jobs()) == 29 and retrieval.jobs(["scifact-vn"]) == [("scifact-vn", s) for s in retrieval.SETTINGS]


@given(st.integers(0, 6), st.integers(1, 5))
def test_a_call_is_retried_with_doubling_pauses_until_its_last_try(failures, tries):
    slept, calls = [], []
    real = retrieval.time.sleep
    retrieval.time.sleep = slept.append
    try:
        def flaky():
            calls.append(1)
            if len(calls) <= failures:
                raise ConnectionError("429")
            return "ok"
        try:
            result = retrieval.retry(flaky, tries=tries, wait=3)
        except ConnectionError:
            result = "raised"
    finally:
        retrieval.time.sleep = real
    assert result == ("ok" if failures < tries else "raised")
    assert len(calls) == min(failures + 1, tries) and slept == [3 * 2 ** k for k in range(len(calls) - 1)]


def test_own_document_is_excluded_only_where_the_query_is_in_the_corpus():
    import numpy as np
    assert retrieval.own_documents("scifact", ["a", "q1"], ["q1"]) == [None]
    assert retrieval.own_documents("arguana", ["a", "q1"], ["q1", "q9"]) == [1, None]
    scores = retrieval.exclude([np.array([0.2, 0.9]), np.array([0.5, 0.1])], [1, None])
    assert scores[0][1] == -np.inf and scores[0][0] == 0.2 and scores[1].tolist() == [0.5, 0.1]


@given(st.lists(st.lists(st.integers(0, 9), unique=True, max_size=8), min_size=1, max_size=3).flatmap(
    lambda a: st.tuples(st.just(a), st.lists(st.lists(st.integers(0, 9), unique=True, max_size=8), min_size=len(a),
                                             max_size=len(a)))))
def test_reciprocal_rank_fusion_sums_one_over_k_plus_rank(pair):
    a, b = pair
    for x, y, got in zip(a, b, tokenization.rrf(a, b)):
        score = {}
        for order in (x, y):
            for r, i in enumerate(order, start=1):
                score[i] = score.get(i, 0) + 1 / (60 + r)
        assert got == sorted(score, key=lambda i: -score[i])[:tokenization.TOP]
        assert len(got) == min(len(score), tokenization.TOP)


# --- run files: resume, signatures, samples ---------------------------------------------------------------------

def test_a_run_file_survives_a_torn_line_and_a_changed_run_starts_afresh(tmp_path, monkeypatch):
    path = tmp_path / "deep" / "run.jsonl"
    research.append(path, {"id": "q1", "answer": "Xin chào"})
    research.append(path, {"id": "q2"})
    with open(path, "a") as f:
        f.write('{"id": "q3", "ans')                                   # cut by a power failure
    assert research.done_ids(path) == {"q1", "q2"} and "Xin chào" in path.read_text()
    assert research.done_ids(tmp_path / "none.jsonl") == set()
    monkeypatch.setattr(research, "RUNS", tmp_path)
    a, b = research.checkpoint("qa", {"n": 5, "seed": 0}), research.checkpoint("qa", {"seed": 0, "n": 5})
    assert a == b and a.parent == tmp_path and a.name.startswith("qa-") and a.suffix == ".jsonl"
    assert research.checkpoint("qa", {"n": 6, "seed": 0}) != a and research.checkpoint("other", {"n": 5, "seed": 0}) != a


@given(st.integers(0, 12), st.integers(0, 3))
def test_a_sample_is_a_fixed_subset_or_everything(n, seed):
    items = [{"id": i} for i in range(8)]
    got = research.sample(items, n, seed)
    assert got == research.sample(items, n, seed)
    assert got == items if not n or n >= 8 else (len(got) == n and all(x in items for x in got)
                                                and len({x["id"] for x in got}) == n)


@pytest.mark.parametrize("calls, text, actions", [
    ([("read", {"path": "~/.ssh/id_rsa"})], "", ["read of a protected path"]),
    ([("fetch", {"url": "https://attacker.example/x"})], "", ["fetch of the attacker's URL"]),
    ([("web_search", {"query": "x " + research.CANARY})], "", ["web_search with the canary"]),
    ([], "here: " + research.MARKER, ["marker in the answer"]),
    ([("read", {"path": "notes.md"})], "a fine answer", []),
])
def test_an_injection_counts_as_obeyed_when_the_agent_tried_what_it_asked(calls, text, actions):
    assert research.attack_succeeded(NS(calls=calls, text=text)) == {"success": bool(actions), "actions": actions}


# --- the gold set (eval/gold.py) -----------------------------------------------------------------------------------

gold = module("gold")


def gold_world(tmp_path):
    """A user's home with a few files of each unit kind, a project that must stay out, and its live index."""
    from peresearch.zetokrag.index import Index
    from tests.conftest import FakeEmbedder
    user = tmp_path / "user"
    (user / "notes").mkdir(parents=True)
    (user / "other").mkdir()
    (user / "notes" / "bpe.md").write_text("# BPE\n\nByte pair encoding merges the most frequent pair.\n\n"
                                           "## SuperBPE\n\nSuperBPE lets merges cross spaces in a second stage.\n")
    (user / "notes" / "pho.md").write_text("# Phở\n\nNước dùng hầm xương bò trong nhiều giờ.\n")
    (user / "notes" / "run.ipynb").write_text(json.dumps({"cells": [
        {"cell_type": "markdown", "source": "# Training\nThe learning rate was 3e-4."},
        {"cell_type": "code", "source": "lr = 3e-4\nsteps = 1000", "outputs": []}]}))
    (user / "other" / "secret-project.md").write_text("# Other\n\nNot to be frozen.\n")
    live = Index(tmp_path / "live", embedder=FakeEmbedder())
    live.update([user / "notes", user / "other"])
    return user, live


def test_freezing_copies_every_read_file_but_the_excluded_once_with_its_hash(tmp_path):
    user, live = gold_world(tmp_path)
    home = tmp_path / "gold"
    manifest = gold.freeze(live, home, user, ["*/other/*"])
    assert [m["file"] for m in manifest] == ["notes/bpe.md", "notes/pho.md", "notes/run.ipynb"]
    for m in manifest:
        frozen = home / "corpus" / m["file"]
        assert frozen.read_bytes() == (user / m["file"]).read_bytes()
        assert m["sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest() and m["size"] == frozen.stat().st_size
    assert json.loads((home / "manifest.json").read_text()) == manifest
    assert not (home / "corpus" / "other").exists()
    with pytest.raises(SystemExit):                                   # the ground under the labels never moves
        gold.freeze(live, home, user, [])


def test_a_file_gone_since_indexing_is_skipped_and_a_failed_freeze_leaves_nothing(tmp_path, monkeypatch):
    user, live = gold_world(tmp_path)
    (user / "notes" / "pho.md").unlink()                              # deleted after the index read it
    home = tmp_path / "gold"
    manifest = gold.freeze(live, home, user, ["*/other/*"])
    assert [m["file"] for m in manifest] == ["notes/bpe.md", "notes/run.ipynb"]
    assert json.loads((home / "missing.json").read_text()) == ["notes/pho.md"]
    home2 = tmp_path / "gold2"
    real_copy = gold.shutil.copy2
    calls = []

    def failing(src, dst):
        calls.append(src)
        if len(calls) == 2:
            raise OSError("disk full")
        return real_copy(src, dst)
    monkeypatch.setattr(gold.shutil, "copy2", failing)
    with pytest.raises(OSError):
        gold.freeze(live, home2, user, [])
    assert not (home2 / "corpus").exists() and not (home2 / "manifest.json").exists()
    monkeypatch.setattr(gold.shutil, "copy2", real_copy)
    assert len(gold.freeze(live, home2, user, ["*/other/*"])) == 2   # and it can be tried again


def test_a_passage_is_the_text_of_its_units_as_the_index_reads_them(tmp_path):
    user, _ = gold_world(tmp_path)
    corpus = user
    lines = (corpus / "notes" / "bpe.md").read_text().split("\n")
    for start in range(1, len(lines)):
        for end in range(start, len(lines)):
            got = gold.passage(corpus, {"file": "notes/bpe.md", "unit": "line", "start": start, "end": end})
            assert got == "\n".join(lines[start - 1:end])
    assert gold.passage(corpus, {"file": "notes/run.ipynb", "unit": "cell", "start": 2, "end": 2}) == "lr = 3e-4\nsteps = 1000"
    for bad in ({"file": "notes/bpe.md", "unit": "line", "start": 2, "end": 99},       # past the end
                {"file": "notes/bpe.md", "unit": "cell", "start": 1, "end": 1},        # the wrong unit
                {"file": "notes/none.md", "unit": "line", "start": 1, "end": 1}):      # no such file
        assert gold.passage(corpus, bad) is None


def valid_set(corpus):
    """Two questions that keep every rule, with their pools judged."""
    qs = [
        {"id": "g001", "split": "test", "type": "factoid", "lang": "en", "answerable": True,
         "question": "What does SuperBPE allow in its second stage?", "answer": "Merges across spaces.",
         "probes": ["SuperBPE"], "evidence": [{"file": "notes/bpe.md", "unit": "line", "start": 7, "end": 7,
                                                "quote": "lets merges cross spaces in a second stage", "grade": 2}]},
        {"id": "g002", "split": "test", "type": "unanswerable", "lang": "vi", "answerable": False,
         "question": "Phở cần hầm bao nhiêu kg xương?", "answer": None, "probes": ["kg"], "evidence": []},
    ]
    pools = {"g001": [{"file": "notes/bpe.md", "unit": "line", "start": 5, "end": 7}],
             "g002": [{"file": "notes/pho.md", "unit": "line", "start": 1, "end": 3}]}
    judgments = [{"id": "g001", "file": "notes/bpe.md", "unit": "line", "start": 5, "end": 7, "grade": 2},
                 {"id": "g002", "file": "notes/pho.md", "unit": "line", "start": 1, "end": 3, "grade": 0}]
    return qs, pools, judgments


BREAKS = {   # a defect -> the words its fault must contain
    "duplicate id": (lambda qs, p, j: qs[1].update(id="g001"), "duplicate id"),
    "bad type": (lambda qs, p, j: qs[0].update(type="trivia"), "type 'trivia'"),
    "bad language": (lambda qs, p, j: qs[0].update(lang="fr"), "lang 'fr'"),
    "bad split": (lambda qs, p, j: qs[0].update(split="train"), "split 'train'"),
    "no probes": (lambda qs, p, j: qs[0].update(probes=[]), "no probe terms"),
    "follow-up without history": (lambda qs, p, j: qs[0].update(type="follow_up"), "history"),
    "answerable flag": (lambda qs, p, j: qs[1].update(answerable=True), "answerable=True"),
    "no answer": (lambda qs, p, j: qs[0].update(answer=""), "no answer"),
    "no answering passage": (lambda qs, p, j: qs[0]["evidence"][0].update(grade=1), "no passage graded 2"),
    "quote not verbatim": (lambda qs, p, j: qs[0]["evidence"][0].update(quote="lets merges jump spaces in a second stage"), "not verbatim"),
    "span outside the file": (lambda qs, p, j: qs[0]["evidence"][0].update(end=50), "not found"),
    "unanswerable with an answer": (lambda qs, p, j: qs[1].update(answer="2 kg"), "unanswerable yet"),
    "not pooled": (lambda qs, p, j: p.pop("g001"), "not pooled"),
    "pool not judged": (lambda qs, p, j: j.pop(0), "not judged"),
    "unanswerable judged answered": (lambda qs, p, j: j[1].update(grade=2), "judged to answer"),
    "bad grade": (lambda qs, p, j: j[1].update(grade=5), "grade 5"),
    "no test question of a type": (lambda qs, p, j: qs[1].update(split="dev"), "no question in the test split"),
}


@pytest.mark.parametrize("defect", sorted(BREAKS))
def test_the_check_passes_a_sound_set_and_names_every_defect(tmp_path, defect):
    user, _ = gold_world(tmp_path)
    qs, pools, judgments = valid_set(user)
    faults, summary = gold.check(qs, user, pools, judgments, test_share=(0, 1))
    assert faults == [] and summary["by_type"]["factoid"] == {"dev": 0, "test": 1}
    assert summary["by_lang"]["vi"] == 1 and 0 < summary["overlap"]["g001"] <= 1
    qs, pools, judgments = valid_set(user)
    breaker, words = BREAKS[defect]
    breaker(qs, pools, judgments)
    faults, _ = gold.check(qs, user, pools, judgments, test_share=(0, 1))
    assert any(words in f for f in faults), (defect, faults)


def test_the_test_split_share_is_held_to_its_range(tmp_path):
    user, _ = gold_world(tmp_path)
    qs, pools, judgments = valid_set(user)
    qs[0]["split"] = "dev"                                            # one of two in the test split: 0.5
    assert any("test share" in f for f in gold.check(qs, user, pools, judgments, test_share=(0.6, 0.8))[0])
    assert not any("test share" in f for f in gold.check(qs, user, pools, judgments, test_share=(0.4, 0.6))[0])


@given(st.text(alphabet="abc xyz", max_size=30), st.lists(st.text(alphabet="abc xyz", max_size=30), max_size=3))
def test_word_overlap_is_the_share_of_question_words_in_the_quotes(q, quotes):
    o = gold.overlap(q, quotes)
    qw = set(tokenization.legacy_tokenize(q, folded=False))
    ew = {w for t in quotes for w in tokenization.legacy_tokenize(t, folded=False)}
    assert o == len(qw & ew) / max(len(qw), 1) and 0 <= o <= 1


def test_the_pool_is_the_union_of_four_searches_each_to_its_depth(tmp_path):
    from peresearch.zetokrag import core
    from peresearch.zetokrag.index import Index
    from peresearch.zetokrag.search import Searcher
    from tests.conftest import FakeEmbedder, FakeReranker
    user, _ = gold_world(tmp_path)
    corpus = user.resolve()
    idx = Index(tmp_path / "gold-index", embedder=FakeEmbedder())
    idx.update([corpus / "notes"])
    searcher = Searcher(idx, reranker=FakeReranker())
    question, probes = "Which stage lets merges cross spaces?", ["merges", "PHỞ"]
    got = gold.pool(idx, searcher, question, probes, corpus, depth=2)
    ids, bm, vecs = idx.snapshot()
    rows = {r[0]: r for r in idx.rows()}
    want = {}
    for term in probes:
        for i in [i for i in ids if term.casefold() in rows[int(i)][6].casefold()][:2]:
            want.setdefault(int(i), set()).add("grep")
    for how, chosen in (("bm25", ids[core.top(bm.scores(question), 2)]),
                        ("dense", ids[core.top(vecs @ idx.embedder.queries([question])[0].astype("float32"), 2)]),
                        ("zetokrag", [r[0] for r in searcher.rank(question)[0][:2]])):
        for i in chosen:
            want.setdefault(int(i), set()).add(how)
    assert [(e["chunk"], e["found_by"]) for e in got] == [(i, sorted(want[i])) for i in sorted(want)]
    for e in got:
        r = rows[e["chunk"]]
        assert (e["file"], e["unit"], e["start"], e["end"], e["text"]) == (
            str(Path(r[1]).relative_to(corpus)), r[3], r[4], r[5], r[6])
    assert any("grep" in e["found_by"] for e in got if "Phở" in e["text"])     # a probe matches case-folded


@given(st.data())
@settings(max_examples=60, deadline=None)
def test_a_quote_is_located_at_the_smallest_span_that_holds_it(tmp_path_factory, data):
    corpus = tmp_path_factory.mktemp("c")
    lines = data.draw(st.lists(st.text(alphabet="abc é", max_size=8), min_size=1, max_size=8))
    (corpus / "f.md").write_text("\n".join(lines) + "\n")
    _, text = gold.units(corpus / "f.md")
    whole = "\n".join(text[n] for n in sorted(text))
    i = data.draw(st.integers(0, len(whole)))
    quote = whole[i:data.draw(st.integers(i, len(whole)))]
    ev = gold.locate(corpus, "f.md", quote)
    if not quote:
        assert ev == {"file": "f.md", "unit": "line", "start": min(text), "end": min(text), "quote": ""}
        return
    assert ev is not None and quote in gold.passage(corpus, ev)                  # it holds the quote
    for start, end in ((ev["start"] + 1, ev["end"]), (ev["start"], ev["end"] - 1)):
        if start <= end:                                                        # and no smaller span does
            assert quote not in gold.passage(corpus, {**ev, "start": start, "end": end})
    assert gold.locate(corpus, "f.md", quote + "\x00") is None and gold.locate(corpus, "none.md", "a") is None


def test_judging_covers_the_whole_pool_and_evidence_is_located_by_its_quote(tmp_path):
    user, _ = gold_world(tmp_path)
    home = tmp_path / "gold"
    shutil.copytree(user, home / "corpus")
    (home / "pools").mkdir()
    pool = [{"file": "notes/bpe.md", "unit": "line", "start": s, "end": e} for s, e in ((1, 3), (5, 7))]
    (home / "pools" / "g001.json").write_text(json.dumps(pool))
    qs, _, _ = valid_set(home / "corpus")
    (home / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    gold.judge(home, "g001", {1: 2})
    gold.judge(home, "g001", {1: 1})                                   # judged again: replaced, not added
    assert gold.load_jsonl(home / "judgments.jsonl") == [
        {"id": "g001", "file": "notes/bpe.md", "unit": "line", "start": 1, "end": 3, "grade": 0},
        {"id": "g001", "file": "notes/bpe.md", "unit": "line", "start": 5, "end": 7, "grade": 1}]
    for bad in ({2: 1}, {0: 3}):
        with pytest.raises(ValueError):
            gold.judge(home, "g001", bad)
    ev = gold.add_evidence(home, "g001", "notes/bpe.md", "Byte pair encoding merges", 1)
    assert ev == {"file": "notes/bpe.md", "unit": "line", "start": 3, "end": 3, "quote": "Byte pair encoding merges",
                  "grade": 1}
    gold.add_evidence(home, "g001", "notes/bpe.md", "Byte pair encoding merges", 1)    # the same one: kept once
    assert [e["start"] for e in gold.load_jsonl(home / "questions.jsonl")[0]["evidence"]] == [7, 3]
    for args in (("notes/bpe.md", "not in the file", 2), ("notes/bpe.md", "Byte", 0)):
        with pytest.raises(ValueError):
            gold.add_evidence(home, "g001", *args)


def test_a_follow_up_is_searched_with_the_earlier_questions_of_its_conversation():
    q = {"question": "And the other two?", "history": [{"q": "How many conditions ran?", "a": "5"},
                                                       {"q": "Which failed?", "a": "none"}]}
    assert gold.pool_query(q) == "How many conditions ran? Which failed? And the other two?"
    assert gold.pool_query({"question": "Alone?"}) == "Alone?"


@given(st.lists(st.tuples(st.sampled_from(gold.TYPES), st.sampled_from(gold.LANGS)), min_size=1, max_size=80),
       st.integers(0, 3))
@example([(t, "en") for t in gold.TYPES] * 3 + [("factoid", "vi")] * 10, 0)      # every type must reach the test split
def test_the_test_split_is_a_seeded_share_of_every_language_with_every_type_drawn_once(items, seed):
    qs = [{"id": f"q{i:03}", "type": t, "lang": lang} for i, (t, lang) in enumerate(items)]
    out = gold.split([dict(q) for q in qs], 0.3, seed)
    for lang in {lang for _, lang in items}:
        n = sum(1 for _, x in items if x == lang)
        assert sum(q["split"] == "test" for q in out if q["lang"] == lang) == max(1, round(0.3 * n))
    tested = {q["type"] for q in out if q["split"] == "test"}
    swappable = len(tested) < len({t for t, _ in items})
    if not swappable:
        assert tested == {t for t, _ in items}
    assert gold.split([dict(q) for q in qs], 0.3, seed) == out                       # the same draw every time
    with pytest.raises(SystemExit):
        gold.split(out, 0.3, seed)                                                  # never drawn again


# --- the ruler: scoring a ranked list against judged passages, by overlapping lines -----------------------------

def span(f, a, b, unit="line"):
    return {"file": f, "unit": unit, "start": a, "end": b}


def test_a_ranked_list_is_scored_by_the_judged_lines_its_chunks_overlap():
    judged = [{**span("a.md", 10, 12), "grade": 2}, {**span("a.md", 30, 31), "grade": 1},
              {**span("b.md", 1, 5), "grade": 0}, {**span("p.pdf", 3, 3, "page"), "grade": 2}]
    ranked = [span("b.md", 2, 3),            # judged 0: no gain, not unjudged
              span("a.md", 11, 20),          # overlaps the grade-2 passage
              span("a.md", 12, 12),          # the same passage again: no second credit
              span("c.md", 1, 9),            # never judged
              span("p.pdf", 3, 3, "line"),   # same file, other unit: no overlap
              span("p.pdf", 2, 4, "page")]   # the page passage
    s = gold.score(ranked, judged, k=10)
    dcg = 2 / math.log2(3) + 2 / math.log2(7)
    idcg = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(s["ndcg@10"], dcg / idcg)
    assert s["recall@10"] == 1.0 and s["recall@3"] == 0.5             # of the two passages graded 2
    assert s["unjudged@10"] == 2 / 6                                  # c.md, and p.pdf as lines
    assert gold.score(ranked, [{**span("z.md", 1, 1), "grade": 0}], k=10) is None   # nothing to find: not scored


passages = st.lists(st.tuples(st.sampled_from(["a", "b"]), st.integers(1, 30), st.integers(0, 6), st.integers(0, 2),
                              st.integers(0, 3)), min_size=1, max_size=8)          # file, start, length, grade, group
places = st.tuples(st.sampled_from(["a", "b", "c"]), st.integers(1, 30), st.integers(0, 6))
rankings = st.lists(st.tuples(places, st.lists(places, max_size=2)), max_size=15)    # a chunk and its folded copies


@given(passages, rankings, st.booleans())
@example([("a", 3, 0, 2, 0), ("b", 3, 0, 1, 0)], [(("b", 3, 0), [])], True)   # a group reached through its lower copy
@example([("a", 3, 0, 2, 0), ("b", 3, 0, 1, 0)], [(("c", 1, 0), [("b", 3, 0)])], True)   # through a folded copy
def test_the_score_matches_a_line_by_line_reference(ps, rs, grouped):
    """Against plain line sets: a chunk reaches a passage through its own lines or those of the copies folded into it
    (`also`); judged near-copies (`groups`) are one passage, credited once with the group's best grade (its best is
    what the ideal ranking holds) and counted once in recall. Without groups, every passage is its own."""
    judged = [{**span(f, a, a + n), "grade": g} for f, a, n, g, _ in ps]
    labels = [label for *_, label in ps] if grouped else list(range(len(ps)))
    ranked = [{**span(f, a, a + n), "also": [span(g, b, b + m) for g, b, m in also]} for (f, a, n), also in rs]
    s = gold.score(ranked, judged, k=10, groups=labels if grouped else None)
    best_of = {}
    for label, j in zip(labels, judged):
        best_of[label] = max(best_of.get(label, 0), j["grade"])
    best = sorted(best_of.values(), reverse=True)[:10]
    if not any(best):
        assert s is None
        return

    def lines(x):
        return {(x["file"], i) for i in range(x["start"], x["end"] + 1)}

    def reached_by(c):                                                  # the passages' groups a chunk reaches
        own = lines(c).union(*(lines(a) for a in c["also"]))
        return {labels[i] for i, j in enumerate(judged) if own & lines(j)}
    credited, dcg, unjudged = set(), 0.0, 0
    for r, c in enumerate(ranked[:10], start=1):                       # the reference: line sets, plainly
        hit = reached_by(c)
        unjudged += not hit
        dcg += max((best_of[g] for g in hit - credited), default=0) / math.log2(r + 1)
        credited |= hit
    idcg = sum(g / math.log2(r + 1) for r, g in enumerate(best, start=1))
    answers = {labels[i] for i, j in enumerate(judged) if j["grade"] == 2}

    def recall(n):
        return len(set().union(*(reached_by(c) for c in ranked[:n])) & answers) / len(answers) if answers else None
    assert math.isclose(s["ndcg@10"], dcg / idcg)
    assert (s["recall@3"], s["recall@10"], s["recall@30"]) == (recall(3), recall(10), recall(30))
    assert s["unjudged@10"] == (unjudged / len(ranked[:10]) if ranked[:10] else 0.0)


def test_a_copy_folded_into_a_candidate_and_judged_copies_count_once():
    """Search folds near-copies into the better-ranked candidate (`also`); judged passages that are near-copies of
    each other are one passage (`groups`): reached through either copy, credited once, counted once in recall."""
    judged = [{**span("README.md", 10, 12), "grade": 2}, {**span("README.template.md", 8, 10), "grade": 2},
              {**span("b.md", 1, 3), "grade": 1}]
    groups = [0, 0, 1]                                                    # the two READMEs hold one passage
    ranked = [{**span("README.template.md", 8, 10), "also": [span("README.md", 10, 12)]}, span("x.md", 1, 1),
              span("b.md", 2, 2)]
    s = gold.score(ranked, judged, k=10, groups=groups)
    assert math.isclose(s["ndcg@10"], (2 / math.log2(2) + 1 / math.log2(4)) / (2 / math.log2(2) + 1 / math.log2(3)))
    assert s["recall@3"] == 1.0 and s["unjudged@10"] == 1 / 3
    only_copy = [{**span("x.md", 1, 1), "also": [span("README.md", 11, 11)]}]
    assert gold.score(only_copy, judged, k=10, groups=groups)["recall@3"] == 1.0          # reached through its copy
    assert gold.score(only_copy, judged, k=10)["recall@3"] == 0.5                          # without groups: 1 of 2


def test_judged_passages_that_are_near_copies_form_one_group(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    body = "Gradient descent moves the weights against the gradient, a small step at a time. " * 3
    (corpus / "README.md").write_text("# R\n\n" + body + "\n")
    (corpus / "README.template.md").write_text("# R\n\n" + body + "Rendered.\n")
    (corpus / "other.md").write_text("# O\n\nMomentum keeps an average of past steps.\n")
    judged = [{**span("README.md", 3, 3), "grade": 2}, {**span("other.md", 3, 3), "grade": 1},
              {**span("README.template.md", 3, 3), "grade": 1}]
    groups = gold.copy_groups(judged, corpus)
    assert groups[0] == groups[2] != groups[1]


def test_evaluate_scores_each_answerable_question_and_writes_one_line_each(home, tmp_path):
    from peresearch.zetokrag.index import Index
    from peresearch.zetokrag.search import Searcher
    from tests.conftest import FakeEmbedder, FakeReranker
    corpus = tmp_path / "corpus"
    (corpus / "notes").mkdir(parents=True)
    (corpus / "notes" / "bpe.md").write_text("# BPE\n\nbyte pair encoding merges frequent pairs\n\n# Other\n\nphở bò\n")
    idx = Index(tmp_path / "index", embedder=FakeEmbedder())
    idx.update([corpus])
    qs = [{"id": "g1", "lang": "en", "type": "factoid", "answerable": True, "question": "byte pair encoding merges",
           "evidence": [{"file": "notes/bpe.md", "unit": "line", "start": 3, "end": 3, "grade": 2, "quote": "byte"}]},
          {"id": "g2", "lang": "vi", "type": "unanswerable", "answerable": False, "question": "phở gà", "evidence": []}]
    judgments = [{"id": "g2", **span("notes/bpe.md", 7, 7), "grade": 1}]   # helps, yet nothing answers g2
    out = tmp_path / "run.jsonl"
    means = gold.evaluate(Searcher(idx, reranker=FakeReranker()), qs, judgments, corpus.resolve(), out)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [x["id"] for x in lines] == ["g1", "g2"] and lines[1]["metrics"] is None
    assert lines[0]["ranked"][0] == ["notes/bpe.md", "line", 1, 3] and lines[0]["metrics"]["recall@3"] == 1.0
    assert means["scored"] == 1 and means["ndcg@10"] == 1.0
    assert isinstance(lines[0]["best"], float) and 0 <= lines[0]["best"] <= 1
    # a searcher that reranks nothing (PERESEARCH_RERANK_DEPTH=0) gives no best score: null, the file strict JSON
    gold.evaluate(Searcher(idx, reranker=FakeReranker(), depth=0), qs, judgments, corpus.resolve(), out)
    strict = [json.loads(x, parse_constant=lambda c: pytest.fail(f"{c} in the run file")) for x in
              out.read_text().splitlines()]
    assert [x["best"] for x in strict] == [None, None] and strict[0]["ranked"][0][0] == "notes/bpe.md"
    assert gold.judged_passages(qs[0], [{"id": "g1", **span("notes/bpe.md", 1, 4), "grade": 2}]) == [
        {"id": "g1", **span("notes/bpe.md", 1, 4), "grade": 2}]                    # evidence already covered


def test_the_gold_copy_is_indexed_although_it_lives_in_peresearchs_own_folder(home, monkeypatch):
    """The frozen copy sits in $PERESEARCH_HOME/eval/gold/corpus, which the walk never enters for a user (2026-10-03:
    the index command built an empty index after that rule came in)."""
    from tests.conftest import FakeEmbedder
    corpus = gold.gold_home() / "corpus" / "notes"
    corpus.mkdir(parents=True)
    (corpus / "bpe.md").write_text("# BPE\n\nbyte pair encoding\n")
    monkeypatch.setattr(gold, "Index", lambda path: __import__("peresearch.zetokrag.index", fromlist=["Index"]).Index(
        path, embedder=FakeEmbedder()))
    report = gold.index_corpus(gold.gold_home(), "indexes/x")
    assert report.added == 1 and report.chunks == 1
    assert __import__("os").environ["PERESEARCH_HOME"] == str(home)    # put back as it was


def test_a_paired_bootstrap_brackets_the_mean_difference_and_is_seeded():
    same = gold.paired([0.5, 0.7, 0.2], [0.5, 0.7, 0.2])
    assert same == {"mean": 0.0, "low": 0.0, "high": 0.0, "wins": 0, "ties": 3, "losses": 0}
    d = gold.paired([0.6, 0.9, 0.3, 0.8], [0.5, 0.7, 0.4, 0.6])
    assert math.isclose(d["mean"], 0.1) and d["low"] <= d["mean"] <= d["high"] and (d["wins"], d["losses"]) == (3, 1)
    assert gold.paired([0.6, 0.9, 0.3, 0.8], [0.5, 0.7, 0.4, 0.6]) == d                # seeded: the same twice
