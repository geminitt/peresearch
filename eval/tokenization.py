"""BM25 under different ways of cutting text into terms, on MIRACL's 18 languages (CPU only, no model runs).

    pixi run python eval/tokenization.py run [lang ...]     # resumable: one result line per (language, variant)
    pixi run python eval/tokenization.py report             # results/tokenization.md
    pixi run python eval/tokenization.py same               # gate-1 corpora: texts each variant cuts differently

ZetokRAG's BM25 cuts text with `\\w+`. That keeps a Chinese or Japanese sentence as one term, and splits Thai and
Indic words at their vowel signs, which are combining marks and not `\\w`. The variants:

- `current`: ZetokRAG as used: `\\w+`, lower-cased, two indexes (as typed, and with combining marks removed) and
  the second one used for queries typed without marks ("auto", the Vietnamese no-diacritics rule).
- `words`: `\\w+`, lower-cased, one index: `current` without the mark-folding, to separate the two effects.
- `words+marks`: letters, marks and digits: a word keeps its vowel signs.
- `words+marks+bigrams`: as `words+marks`, and a stretch written in a script without spaces between words
  (Han, Hiragana, Katakana, Thai, Lao, Khmer, Myanmar) becomes overlapping pairs of characters (grapheme
  clusters), the way Lucene's CJK analyzer handles Han. No dictionary, no rule per language.
- `words+marks+xlmr`: as `words+marks`, and those stretches are cut by the XLM-R tokenizer (BGE-M3's), a
  tokenizer learned from text in 100 languages.
- `xlmr`, `qwen3`: every text cut by a model's tokenizer as is (BGE-M3's XLM-R, Qwen3-Embedding's byte BPE),
  the way the BGE-M3 paper ran its BM25 baseline.
- `rrf:A|B`: the rankings of variants A and B merged by reciprocal rank fusion (sum of 1 / (60 + rank)), the
  way "Better Than Whitespace" (Ogundepo et al. 2022) combined a word analyzer with a subword tokenizer.

Data: mteb/MIRACLRetrievalHardNegatives, dev split, pinned. Each language's corpus is the pool of the top 250
documents per query of BM25, multilingual-e5-large and e5-mistral-instruct, so the scores are higher than on
MIRACL's full corpora and are comparable only between the variants here. BM25 is bm25s with ZetokRAG's
parameters (k1 = 1.5, b = 0.75); the top 100 are kept; a document with score 0 shares no term with the query and
is not counted as retrieved.
"""

import importlib.util
import json
import multiprocessing
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import regex

from peresearch.zetokrag import core

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("PERESEARCH_RUNS", ROOT / "runs")) / "tokenization"
RESULTS = Path(os.environ.get("PERESEARCH_RESULTS", ROOT / "results"))
REPO, REVISION = "mteb/MIRACLRetrievalHardNegatives", "332a9acb49f5e83d5397683f79d23e588f685916"
LANGUAGES = ["ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "yo", "zh"]
TOKENIZERS = {"xlmr": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
              "qwen3": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")}
VARIANTS = ["current", "words", "words+marks", "words+marks+bigrams", "words+marks+xlmr", "xlmr", "qwen3",
            "rrf:words+marks+bigrams|xlmr", "rrf:words+marks+bigrams|qwen3"]
VERSION = 1           # bump when a variant's definition changes: the results of the old one are not reused
TOP = 100
WORKERS = os.cpu_count() or 1   # every core WSL is given (the owner's choice: 16 of the laptop's 22)
BATCH = 4096          # texts per call of a model tokenizer

_spec = importlib.util.spec_from_file_location("retrieval", Path(__file__).parent / "retrieval.py")
retrieval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrieval)
log, metrics, retry = retrieval.log, retrieval.metrics, retrieval.retry

# A word: letters, combining marks (Thai and Indic vowel signs, Vietnamese tones in decomposed form) and digits.
WORD = regex.compile(r"[\p{L}\p{M}\p{N}_]+")
# Scripts written without spaces between words; Script_Extensions takes in the signs they share (ー, 々).
UNSPACED = r"\p{scx=Han}\p{scx=Hiragana}\p{scx=Katakana}\p{scx=Thai}\p{scx=Lao}\p{scx=Khmer}\p{scx=Myanmar}"
UNSPACED_RUN = regex.compile(rf"((?:[{UNSPACED}]\p{{M}}*)+)")
GRAPHEME = regex.compile(r"\X")


def words_marks(text: str) -> list[str]:
    return WORD.findall(unicodedata.normalize("NFC", text).lower())


def split_unspaced(text: str, cut) -> list[str]:
    """Words as `words_marks`; a stretch in a script without spaces is cut by `cut`."""
    out = []
    for word in words_marks(text):
        for k, piece in enumerate(UNSPACED_RUN.split(word)):
            if piece:
                out.extend(cut(piece) if k % 2 else [piece])
    return out


def bigrams(run: str) -> list[str]:
    g = GRAPHEME.findall(run)
    return g if len(g) == 1 else [g[i] + g[i + 1] for i in range(len(g) - 1)]


_tokenizers = {}


def model_tokenizer(name: str):
    if name not in _tokenizers:
        from transformers import AutoTokenizer
        from transformers.utils import logging

        logging.set_verbosity_error()          # "sequence longer than 512": BM25 reads whole texts on purpose
        repo, rev = TOKENIZERS[name]
        _tokenizers[name] = AutoTokenizer.from_pretrained(repo, revision=rev)
    return _tokenizers[name]


def _plain(text: str) -> list[str]:
    return core.tokenize(text, folded=False)


def _folded(text: str) -> list[str]:
    return core.tokenize(text, folded=True)


def _bigrams(text: str) -> list[str]:
    return split_unspaced(text, bigrams)


# The variants cut by a regex; "words-folded" is the second index of `current`.
REGEX_CUTS = {"words": _plain, "words-folded": _folded, "words+marks": words_marks, "words+marks+bigrams": _bigrams}


def cut_all(variant: str, texts: list[str]) -> list[list[str]]:
    """Every text's terms under `variant`, on all cores: the regex variants in a pool of processes, the model
    tokenizers in batches (their Rust code spreads a batch over the cores). Same terms as cutting one text at a
    time, which the tests check."""
    if variant in REGEX_CUTS:
        if len(texts) < 2000:
            return [REGEX_CUTS[variant](t) for t in texts]
        # spawn, not fork: forking a process that already runs threads (the model tokenizers', bm25s') can deadlock
        with multiprocessing.get_context("spawn").Pool(WORKERS) as pool:
            return pool.map(REGEX_CUTS[variant], texts, chunksize=512)
    if variant in TOKENIZERS:
        tok, out = model_tokenizer(variant), []
        for a in range(0, len(texts), BATCH):
            batch = [unicodedata.normalize("NFC", t) for t in texts[a:a + BATCH]]
            out.extend(tok.convert_ids_to_tokens(ids) for ids in tok(batch, add_special_tokens=False)["input_ids"])
        return out
    if variant == "words+marks+xlmr":          # words, and every stretch without spaces cut by XLM-R
        words = cut_all("words+marks", texts)
        split = [[(k % 2, p) for w in ws for k, p in enumerate(UNSPACED_RUN.split(w)) if p] for ws in words]
        runs = sorted({p for parts in split for unspaced, p in parts if unspaced})
        cut = {r: [x for x in (y.lstrip("▁") for y in toks) if x] for r, toks in zip(runs, cut_all("xlmr", runs))}
        return [[t for unspaced, p in parts for t in (cut[p] if unspaced else [p])] for parts in split]
    raise KeyError(variant)


# --- data -------------------------------------------------------------------------------------------------

def load(lang: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    get = lambda part: pd.read_parquet(retry(lambda: hf_hub_download(
        REPO, f"{lang}-{part}/dev-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)))
    corpus, queries, qrels = get("corpus"), get("queries"), get("qrels")
    title = corpus["title"].fillna("") if "title" in corpus else ""
    texts = ((title + "\n" + corpus["text"].fillna("")).str.strip() if "title" in corpus
             else corpus["text"].fillna("")).tolist()
    doc_ids = corpus["_id"].astype(str).tolist() if "_id" in corpus else corpus["id"].astype(str).tolist()
    rel: dict[str, dict[str, int]] = {}
    for q, d, s in zip(qrels["query-id"].astype(str), qrels["corpus-id"].astype(str), qrels["score"]):
        if int(s) > 0:
            rel.setdefault(q, {})[d] = int(s)
    known = set(doc_ids)
    rel = {q: {d: s for d, s in r.items() if d in known} for q, r in rel.items()}
    qid = "_id" if "_id" in queries else "id"
    qtext = dict(zip(queries[qid].astype(str), queries["text"]))
    qs = sorted(q for q, r in rel.items() if r and q in qtext)
    return doc_ids, texts, qs, [qtext[q] for q in qs], [rel[q] for q in qs]


# --- run --------------------------------------------------------------------------------------------------

def signature(lang: str, variant: str) -> str:
    return f"{lang}|{variant}|{REVISION}|v{VERSION}"


def done() -> dict[str, dict]:
    """Finished results by signature; a line torn by a crash is skipped (that pair runs again)."""
    out, path = {}, RUNS / "results.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                out[r["signature"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def save(result: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rank(index_terms: list[list[str]], query_terms: list[list[str]]) -> list[list[int]]:
    import bm25s

    model = bm25s.BM25()
    model.index([t or ["∅"] for t in index_terms], show_progress=False)
    ids, scores = model.retrieve([q or ["∅"] for q in query_terms], k=min(TOP, len(index_terms)), show_progress=False,
                                 n_threads=WORKERS)
    return [[int(i) for i, s in zip(row, srow) if s > 0] for row, srow in zip(ids, scores)]


def orders_path(lang: str, variant: str) -> Path:
    return RUNS / "orders" / f"{lang}-{variant.replace('|', '_')}-v{VERSION}.npy"


def save_orders(lang: str, variant: str, orders: list[list[int]]) -> None:
    """The top-100 of every query (-1 pads), kept so that fusions can be computed without ranking again."""
    a = np.full((len(orders), TOP), -1, dtype=np.int32)
    for r, o in enumerate(orders):
        a[r, :len(o)] = o
    orders_path(lang, variant).parent.mkdir(parents=True, exist_ok=True)
    np.save(orders_path(lang, variant), a)


def load_orders(lang: str, variant: str) -> list[list[int]]:
    return [[int(i) for i in row if i >= 0] for row in np.load(orders_path(lang, variant))]


def rrf(a: list[list[int]], b: list[list[int]], k: int = 60) -> list[list[int]]:
    out = []
    for x, y in zip(a, b):
        score: dict[int, float] = {}
        for order in (x, y):
            for r, i in enumerate(order, start=1):
                score[i] = score.get(i, 0.0) + 1.0 / (k + r)
        out.append(sorted(score, key=lambda i: -score[i])[:TOP])
    return out


def run_language(lang: str, variants, finished: dict) -> None:
    todo = [v for v in variants if signature(lang, v) not in finished or
            (not v.startswith("rrf:") and not orders_path(lang, v).exists())]
    if not todo:
        return
    doc_ids, texts, qs, qtexts, rels = load(lang)
    log(lang, f"{len(texts)} documents, {len(qs)} queries")
    for variant in sorted(todo, key=lambda v: v.startswith("rrf:")):     # fusions after their parts
        t = time.time()
        terms = []
        if variant.startswith("rrf:"):
            a, b = variant[4:].split("|")
            orders, t_cut = rrf(load_orders(lang, a), load_orders(lang, b)), 0.0
        elif variant == "current":
            plain, folded = cut_all("words", texts), cut_all("words-folded", texts)
            t_cut = time.time() - t
            by_plain, by_folded = rank(plain, cut_all("words", qtexts)), rank(folded, cut_all("words-folded", qtexts))
            del folded
            orders = [p if core.accented(q) else f for q, p, f in zip(qtexts, by_plain, by_folded)]
            terms = plain
        else:
            terms = cut_all(variant, texts)
            t_cut = time.time() - t
            orders = rank(terms, cut_all(variant, qtexts))
        if not variant.startswith("rrf:"):
            save_orders(lang, variant, orders)
        per_query = [metrics([doc_ids[i] for i in o], r) for o, r in zip(orders, rels)]
        result = {"signature": signature(lang, variant), "lang": lang, "variant": variant, "queries": qs,
                  "documents": len(texts),
                  "terms_per_document": float(np.mean([len(x) for x in terms])) if terms else None,
                  "vocabulary": len({w for x in terms for w in x}), "seconds_cutting": round(t_cut, 1),
                  "seconds": round(time.time() - t, 1),
                  "per_query": {k: [m[k] for m in per_query] for k in per_query[0]}}
        save(result)
        log(lang, variant, f"nDCG@10 {100 * np.mean(result['per_query']['ndcg@10']):.1f}",
            f"R@100 {100 * np.mean(result['per_query']['recall@100']):.1f}", f"{result['seconds']:.0f}s")
        del terms


def run(langs, variants=VARIANTS) -> list[str]:
    import traceback

    failed, finished = [], done()
    for lang in langs:
        try:
            run_language(lang, variants, finished)
        except Exception:
            log(lang, "FAILED")
            traceback.print_exc()
            failed.append(lang)
    return failed


# --- report -----------------------------------------------------------------------------------------------

def report(out_path: Path = RESULTS / "tokenization.md", n_boot: int = 10_000) -> str:
    rng = np.random.default_rng(0)
    res = {(r["lang"], r["variant"]): r for r in done().values()}
    langs = [x for x in LANGUAGES if any((x, v) in res for v in VARIANTS)]
    lines = ["# BM25 tokenization across languages (MIRACL, hard-negative pools)", "",
             "BM25 only (bm25s, k1 = 1.5, b = 0.75), no embedding and no reranker, so the numbers isolate how text "
             "is cut into terms. Corpora are MIRACL's dev pools (top 250 per query of BM25, multilingual-e5-large "
             f"and e5-mistral-instruct; {REPO} at {REVISION[:7]}), so scores are comparable between variants only. "
             "Variants are defined in eval/tokenization.py.", ""]
    for metric in ["ndcg@10", "recall@100"]:
        lines += [f"## {metric}", "", "| lang | docs | queries | " + " | ".join(VARIANTS) + " |",
                  "|---|---:|---:|" + "---:|" * len(VARIANTS)]
        for lang in langs:
            vals = [100 * np.mean(res[(lang, v)]["per_query"][metric]) if (lang, v) in res else None for v in VARIANTS]
            best = max(x for x in vals if x is not None)
            r0 = next(res[(lang, v)] for v in VARIANTS if (lang, v) in res)
            lines.append(f"| {lang} | {r0['documents']} | {len(r0['queries'])} | " + " | ".join(
                "" if x is None else (f"**{x:.1f}**" if x == best else f"{x:.1f}") for x in vals) + " |")
        full = [lang for lang in langs if all((lang, v) in res for v in VARIANTS)]
        if full:
            vals = [100 * np.mean([np.mean(res[(lang, v)]["per_query"][metric]) for lang in full]) for v in VARIANTS]
            lines.append(f"| **mean ({len(full)})** | | | " + " | ".join(f"{x:.1f}" for x in vals) + " |")
        lines.append("")
    lines += ["## Each variant minus `current` (nDCG@10, points)", "",
              "Paired bootstrap within each language (the same queries for both), "
              f"{n_boot} resamples, percentile 95% interval.", "",
              "| lang | " + " | ".join(v for v in VARIANTS if v != "current") + " |",
              "|---|" + "---|" * (len(VARIANTS) - 1)]
    for lang in langs:
        if (lang, "current") not in res:
            continue
        base = np.array(res[(lang, "current")]["per_query"]["ndcg@10"])
        cells = []
        for v in VARIANTS[1:]:
            if (lang, v) not in res:
                cells.append("")
                continue
            d = np.array(res[(lang, v)]["per_query"]["ndcg@10"]) - base
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            cells.append(f"{100 * d.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
        lines.append(f"| {lang} | " + " | ".join(cells) + " |")
    lines += ["", "## Index size and time", "", "| lang | " + " | ".join(
        f"{v} terms/doc" for v in VARIANTS) + " |", "|---|" + "---:|" * len(VARIANTS)]
    base_variants = [v for v in VARIANTS if not v.startswith("rrf:")]
    lines[-2:] = ["| lang | " + " | ".join(f"{v} terms/doc" for v in base_variants) + " |",
                  "|---|" + "---:|" * len(base_variants)]
    for lang in langs:
        lines.append(f"| {lang} | " + " | ".join(
            f"{res[(lang, v)]['terms_per_document']:.0f}" if (lang, v) in res else "" for v in base_variants) + " |")
    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text


# --- the gate-1 corpora -------------------------------------------------------------------------------------

def same(variants=("words+marks", "words+marks+bigrams", "words+marks+xlmr")) -> dict:
    """How many texts (documents and queries, both query settings) of the 17 gate-1 corpora each variant cuts
    differently from `current`'s plain and folded terms. 0 means gate 1's BM25 results stand unchanged."""
    out = {}
    for name in retrieval.DATASETS:
        _, texts, _, qtexts, _ = retrieval.load(name)
        allq = list(qtexts) + ([core.fold(q) for q in qtexts] if retrieval.DATASETS[name][3] == "vi" else [])
        pool = texts + allq
        out[name] = {"texts": len(pool)}
        today = cut_all("words", pool)
        for v in variants:
            differ = [x for x, a, b in zip(pool, cut_all(v, pool), today) if a != b]
            out[name][v] = len(differ)
            if differ:
                out[name][v + ":example"] = differ[0][:200]
        log(name, out[name])
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "same.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["run"]:
        failed = run([a for a in args[1:] if a in LANGUAGES] or LANGUAGES)
        print(report())
        if failed:
            sys.exit(f"failed: {' '.join(failed)}")
    elif args == ["report"]:
        print(report())
    elif args == ["same"]:
        print(json.dumps(same(), ensure_ascii=False, indent=1))
    else:
        sys.exit(__doc__)
