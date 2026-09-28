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

**Public corpora** (`eval/retrieval.py` → [`results/retrieval.md`](results/retrieval.md)): 17 corpora, about
915,000 documents — five BEIR sets in English, their Vietnamese translations and seven more Vietnamese sets
from VN-MTEB (including Zalo legal retrieval, natively Vietnamese); up to 1,000 queries each. nDCG@10, mean
over corpora (the configuration measured here used BGE-M3; see the variants below):

| | BM25 | BGE-M3 | multilingual-e5-large | Qwen3-Embedding-0.6B | fusion, no rerank | BGE-M3 + rerank | **ZetokRAG** |
|---|---:|---:|---:|---:|---:|---:|---:|
| English (5) | 34.0 | 38.6 | 41.0 | **44.6** | 40.5 | 42.2 | 42.8 |
| Vietnamese (12) | 45.4 | 56.9 | 56.8 | 56.0 | 55.1 | **60.2** | 59.9 |

With a paired bootstrap over queries, ZetokRAG beats every single retriever on Vietnamese by 3.0–3.9 points
(intervals exclude 0) and ties BGE-M3 + rerank (−0.2 [−0.6, +0.1]); on English, Qwen3-Embedding alone is
1.7 [0.8, 2.6] points better — the reranker is the limit there: SCIDOCS (find the papers a paper cites,
−4.6) and ArguAna (find the *counter*-argument, −4.2) cost the most, while on SciFact ZetokRAG is 4.2 ahead.

**Configuration** (`eval/variants.py` → [`results/variants.md`](results/variants.md), 200 queries per corpus):
matching every query on diacritic-free words ("ma" for ma, má, mà, mả, mã, mạ) makes BM25 alone 4.4 points
worse on Vietnamese, but inside the full pipeline it makes no difference; for queries typed **without**
diacritics it is 9.0 [8.1, 9.9] points better than plain BM25. ZetokRAG therefore folds diacritics only when
the query has none. Swapping BGE-M3 for Qwen3-Embedding-0.6B changes nothing measurable on queries as typed
(+0.1 to +0.2, intervals include 0) and adds 1.0 [0.5, 1.4] points on queries without diacritics, so ZetokRAG
uses Qwen3-Embedding.

**Your own files** (`eval/personal.py`; the question set and index never leave the machine or enter this
repository): 85 questions written from the owner's notes, projects and course material (8,059 chunks from 453
files) and reviewed by them — 60 answerable, 25 about topics absent from the files.

| | hit@1 | hit@5 | hit@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| exact labelled passage | 68.3% | 95.0% | 98.3% | 0.850 |
| same passage in any file (e.g. a solution notebook) | 76.7% | 95.0% | 98.3% | 0.885 |

The reranker's best score separates the two kinds of question completely (AUROC 1.000): every unanswerable
question scored at most 0.16, every answerable one at least 0.56. The verdict thresholds sit in that gap —
"partial" from 0.25 (on the public corpora 90% of answerable queries reach it), "enough" from 0.5. On the
public corpora the same test is much weaker (AUROC 0.51–0.96): removing a query's labelled documents leaves
other, unlabelled relevant ones behind, so "unanswerable" there is often not.

Limits: 25 unanswerable questions is a small set; one answerable question was labelled with a single passage
although the README of the same project answers it too, which counts as a miss above.

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
```

## Development

```bash
pixi run test            # tests (the default environment includes PyTorch for the local models)
pixi run -e ci test      # the same tests without PyTorch, as CI runs them
```
