# peresearch

**A personal research agent: it first checks what *you* already know — your notes, files and folders — then
searches the web, and cites every claim.**

Status: under construction — the retrieval layer (ZetokRAG) and the guard are done; the agent is next. Design:

- **ZetokRAG** (zero-token RAG) finds evidence in your own files without calling an LLM: BM25 plus dense
  embeddings (Qwen3-Embedding-0.6B), min-max score fusion, reranking (bge-reranker-v2-m3), and a calibrated
  "enough / partial / nothing" verdict. Excerpts are quoted verbatim with their source.
- **Agent**: one agent whose loop and tools run on your machine; the model (Qwen3.6-35B-A3B) runs on
  [Modal](https://modal.com). Your files come first, the web (free search APIs) second; answers separate
  "already in your notes", "new from the web" and the synthesis, each claim cited.
- **Guard**: only declared folders are read, credentials are never indexed, and everything that leaves the
  machine passes one filter.

## ZetokRAG: measured

In short: on Vietnamese, ZetokRAG is among the best methods measured but not ahead of all of them (it ties
BGE-M3 + reranking); on English, Qwen3-Embedding alone is better; on the owner's own files it is clearly
better than BM25 and statistically tied with every embedding-based method tried.

**Public corpora** (`eval/retrieval.py` → [`results/retrieval.md`](results/retrieval.md)): 17 corpora, about
915,000 documents — five BEIR sets in English, their Vietnamese translations and seven more Vietnamese sets
from VN-MTEB (including Zalo legal retrieval, natively Vietnamese); up to 1,000 queries each. In ArguAna each
query is itself a document of the corpus; as in BEIR it is excluded from every ranking. nDCG@10, mean over
corpora. This run measured an earlier ZetokRAG configuration — BGE-M3, and BM25 on diacritic-free words for
every query; the configuration in use differs only in those two choices, compared below:

| | BM25 | BGE-M3 | multilingual-e5-large | Qwen3-Embedding-0.6B | fusion, no rerank | BGE-M3 + rerank | **ZetokRAG** |
|---|---:|---:|---:|---:|---:|---:|---:|
| English (5) | 36.4 | 41.5 | 43.9 | **48.4** | 43.0 | 44.3 | 45.1 |
| Vietnamese (12) | 46.2 | 58.0 | 57.9 | 57.3 | 55.9 | **60.8** | 60.7 |

Paired bootstrap over queries (ZetokRAG minus the other method, points, 95% interval): on Vietnamese it is
ahead of BM25 (+14.5 [13.8, 15.2]) and of each embedding model alone (+2.7 to +3.4, all intervals above 0),
and tied with BGE-M3 + rerank (−0.2 [−0.5, +0.2]). On English, Qwen3-Embedding alone is 3.3 [2.4, 4.3] points
better: the reranker costs most on ArguAna (find the *counter*-argument, −12.3) and SCIDOCS (find the papers
a paper cites, −4.6), while on SciFact ZetokRAG is 4.2 ahead.

**Configuration** (`eval/variants.py` → [`results/variants.md`](results/variants.md), 200 queries per corpus):
matching every query on diacritic-free words ("ma" for ma, má, mà, mả, mã, mạ) makes BM25 alone 4.6 points
worse on Vietnamese, while inside the full pipeline it makes no difference; for queries typed **without**
diacritics, the full pipeline with it is 9.1 [8.2, 10.0] points better than with plain BM25. ZetokRAG
therefore folds diacritics only when the query has none. Swapping BGE-M3 for Qwen3-Embedding-0.6B changes
nothing measurable on queries as typed (+0.0 to +0.1, intervals include 0) and adds 0.9 [0.4, 1.4] points on
queries without diacritics, so ZetokRAG uses Qwen3-Embedding.

**The owner's own files** (`eval/personal.py`; the question set and index never leave the machine or enter
this repository): 85 questions (60 answerable, 25 about topics absent from the files) over 8,059 chunks from
453 files of notes, projects and course material. The questions were written by the assistant from the
labelled passages and reviewed by the owner; written that way they tend to reuse the passage's words, so real
questions are likely harder. Strict scoring (the labelled file and lines), the same chunks for every method:

| | hit@1 | hit@5 | hit@10 | nDCG@10 | ZetokRAG minus it [95% interval] |
|---|---:|---:|---:|---:|---|
| BM25 | 61.7% | 83.3% | 91.7% | 0.758 | +0.091 [+0.017, +0.171] |
| BGE-M3 | 68.3% | 91.7% | 95.0% | 0.818 | +0.032 [−0.039, +0.103] |
| Qwen3-Embedding-0.6B | 61.7% | 95.0% | 98.3% | 0.818 | +0.032 [−0.037, +0.102] |
| fusion, no rerank | 66.7% | 95.0% | 98.3% | 0.827 | +0.023 [−0.039, +0.089] |
| BGE-M3 + rerank | 68.3% | 91.7% | 96.7% | 0.837 | +0.013 [−0.022, +0.050] |
| Qwen3-Embedding + rerank | 68.3% | 95.0% | 100.0% | 0.856 | −0.006 [−0.018, +0.000] |
| BM25 + rerank | 66.7% | 91.7% | 95.0% | 0.822 | +0.027 [−0.007, +0.074] |
| **ZetokRAG** | 68.3% | 95.0% | 98.3% | 0.850 | |

Counting the same passage kept in another file (e.g. a solution notebook) as found, ZetokRAG reaches 76.7%
hit@1 and nDCG@10 0.885. With 60 answerable questions only the gap to BM25 is resolved.

"Does the index hold an answer?": the reranker's best score was at most 0.16 for every unanswerable question
and at least 0.56 for every answerable one, and the verdict thresholds ("partial" from 0.25, "enough" from
0.5) were set in that gap *after* seeing these scores — so the resulting 25/25 rejected and 0/60 wrongly
rejected are in-sample. Their exact 95% intervals, [86.3%, 100%] and [0%, 6.0%], are what 25 and 60 questions
can support. On the public corpora 86% of answerable queries reach 0.25, and the same test is weak (AUROC
0.54–0.96): removing a query's labelled documents leaves other, unlabelled relevant ones behind, so
"unanswerable" there is often not.

Limits: one answerable question was labelled with a single passage although the README of the same project
answers it too, which counts as a miss above; the set is small.

## Usage (so far)

```bash
pixi run peresearch add ~/notes ~/projects/x/docs   # declare the folders peresearch may read
pixi run peresearch index                           # index them on this machine (local GPU)
pixi run peresearch find "hybrid retrieval"         # search your files: no model call, nothing leaves
```

## Guard: what can go wrong, and the test that shows it doesn't

| Risk | Handling | Test |
|---|---|---|
| Credentials read or sent out | Only declared folders; symlinks resolved and must stay inside them; protected names (`.ssh`, `.env`, token and key files, `.modal.toml`...) never read; paragraphs that look like a credential never indexed; everything leaving the machine redacted, search queries with a credential refused | `tests/test_canary.py`: fake credentials in 12 formats planted in notes, `.env`, key files, notebooks and behind symlinks; none reaches the index, the search results, the outbound text or the logs |
| Broken or unreadable files | Reported by name, never fatal: binary, too large, unsupported, corrupt, PDF without a text layer ("needs OCR") | `tests/test_parse_chunk.py` |
| Quoting a file that changed | The file is re-checked before an excerpt is shown; an excerpt no longer in the file is dropped, one still present is flagged | `tests/test_index_search.py` |
| Instructions hidden in a document | Documents are data: indexed and returned as text, never executed | `tests/test_canary.py` (the agent-side check comes with the agent) |
| Terminal escape sequences in untrusted text | Stripped before printing | `tests/test_guard.py` |
| Logs holding sensitive text | The audit log records events and counts, never content | `tests/test_guard.py`, `tests/test_canary.py` |
| Changed models or packages | Hugging Face revisions pinned to commits; `pixi.lock` | `tests/test_supply_chain.py` |

## Reproduce the measurements

```bash
pixi run python eval/retrieval.py run     # 17 corpora x 10 methods, ~14 h on a 6 GB laptop GPU; resumable
pixi run python eval/variants.py run      # ZetokRAG variants, ~1.5 h
pixi run python eval/personal.py ~/notes  # your own question set in $PERESEARCH_HOME/eval/personal.jsonl
pixi run python eval/personal.py compare  # the same questions for eight other methods
```

## Development

```bash
pixi run test            # tests (the default environment includes PyTorch for the local models)
pixi run -e ci test      # the same tests without PyTorch, as CI runs them
```
