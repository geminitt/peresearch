<div align="center">

# peresearch

[![CI](https://img.shields.io/github/actions/workflow/status/geminitt/peresearch/ci.yml?branch=main&label=CI&style=for-the-badge)](https://github.com/geminitt/peresearch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/PYTHON-3.12-498AF2?style=for-the-badge)](./pixi.toml)
[![License](https://img.shields.io/badge/LICENSE-AGPL--3.0-6B7F4E?style=for-the-badge)](./LICENSE)

**A personal research agent: it first checks what *you* already know — your notes, files and folders — then
searches the web, and cites every claim.**

</div>

---

> **Status:** under construction — the retrieval layer (ZetokRAG) and the guard are done and measured; the agent
> is next.

## How it works

| Part | What it does |
|---|---|
| **ZetokRAG** (zero-token RAG) | Finds evidence in your own files without calling an LLM: BM25 (on diacritic-free words only when the query is typed without them) plus Qwen3-Embedding-0.6B, min-max score fusion, reranking of the top 30 with bge-reranker-v2-m3, and a calibrated "enough / partial / nothing" verdict. Excerpts are quoted verbatim with their source. |
| **Agent** | One agent whose loop and tools run on your machine; the model (Qwen3.6-35B-A3B) runs on [Modal](https://modal.com). Your files come first, the web (free search APIs) second; answers separate "already in your notes", "new from the web" and the synthesis, each claim cited. |
| **Guard** | Only declared folders are read, credentials are never indexed, and everything that leaves the machine passes one filter. |

---

## Results

### Public corpora

17 corpora, about 915,000 documents: five BEIR sets in English, their Vietnamese translations and seven more
Vietnamese sets from VN-MTEB (including Zalo legal retrieval, natively Vietnamese); up to 1,000 queries per
corpus. Every Vietnamese corpus is run twice: queries as written, and the same queries without diacritics
("hoc may"), the way they are often typed. nDCG@10, mean over corpora; measured on Kaggle T4s at commit
`fad66ca`.

| Method | English (5) | Vietnamese (12) | Vietnamese, no diacritics (12) |
|---|---:|---:|---:|
| BM25 | 36.4 | 46.2 | 24.7 |
| BM25, diacritic-free when the query is | 36.4 | 46.2 | 44.2 |
| BGE-M3 | 41.4 | 58.0 | 30.9 |
| Qwen3-Embedding-0.6B | **48.4** | 57.3 | 36.3 |
| Fusion, no rerank | 46.3 | 56.7 | 47.3 |
| BGE-M3 + rerank | 44.3 | 60.8 | 36.1 |
| ZetokRAG with BGE-M3 | 45.1 | 60.8 | 46.7 |
| **ZetokRAG** | 45.1 | **60.9** | **47.7** |

- **Vietnamese as typed:** ZetokRAG ties BGE-M3 + rerank (+0.1 [−0.2, +0.5], paired bootstrap, 95%) and is
  ahead of every other method.
- **Vietnamese without diacritics:** ZetokRAG is best, tied only with fusion without reranking (+0.5
  [−0.1, +1.1]); embedding-only methods fall 11–27 points behind.
- **English:** Qwen3-Embedding alone is better by 3.3 [2.3, 4.2] points; the reranker costs most on ArguAna
  and SCIDOCS.
- **Embedder:** a rule fixed before the run kept Qwen3-Embedding over BGE-M3 only if it helped on queries
  without diacritics and did not hurt as typed: +1.1 [+0.8, +1.4] and +0.1 [−0.2, +0.5].

All 14 methods, four metrics per corpus, every paired comparison and the "does the corpus hold an answer?"
test: [`results/retrieval.md`](results/retrieval.md).

### The owner's own files

85 questions (60 answerable, 25 about topics absent from the files) over 8,059 chunks of notes, projects and
course material; the questions and index never leave the machine. ZetokRAG: hit@5 95.0%, nDCG@10 0.850 —
ahead of BM25 (+0.091 [+0.017, +0.171]) and statistically tied with every embedding-based method. Every
unanswerable question scored below the "partial" threshold and every answerable one above it. Details:
[`results/personal.md`](results/personal.md).

### Speed

On a T4, indexing runs at 115 documents/s with BGE-M3 and 43 with Qwen3-Embedding (2.7x slower); answering
one query takes 0.26–0.89 s (median, by corpus), mostly the reranker. Details: [`results/speed.md`](results/speed.md).

> **Limits:** the verdict thresholds were set on the same 85 questions (in-sample); the questions were written by
> the assistant from the labelled passages, so real questions are likely harder; one answerable question is
> also answered by a file it was not labelled with; on the public corpora the "no answer" test is weak (AUROC
> 0.54–0.97), because removing a query's labelled documents leaves unlabelled relevant ones behind.

---

## Usage (so far)

```bash
pixi run peresearch add ~/notes ~/projects/x/docs   # declare the folders peresearch may read
pixi run peresearch index                           # index them on this machine (local GPU)
pixi run peresearch find "hybrid retrieval"         # search your files: no model call, nothing leaves
```

---

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

---

## Reproduce the measurements

```bash
# public corpora: two Kaggle kernels, 2 x T4 each (5.2 h and 4.4 h), resumable
python kaggle/push.py retrieval --job full-shards-0-1 --set 'SHARDS=[0, 1]'
python kaggle/push.py retrieval --job full-shards-2-3 --set 'SHARDS=[2, 3]'
# with both kernels' runs/ outputs downloaded into $PERESEARCH_RUNS:
pixi run python eval/retrieval.py report  # results/retrieval.md
pixi run python eval/speed.py report      # results/speed.md
# your own question set in $PERESEARCH_HOME/eval/personal.jsonl
pixi run python eval/personal.py ~/notes                       # Qwen3-Embedding index
pixi run python eval/personal.py --embedder bge-m3 ~/notes     # BGE-M3 index
pixi run python eval/personal.py compare                       # results/personal.md
```

---

## Development

```bash
pixi run test            # tests (the default environment includes PyTorch for the local models)
pixi run -e ci test      # the same tests without PyTorch, as CI runs them
```
