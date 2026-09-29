<div align="center">

# peresearch

[![CI](https://img.shields.io/github/actions/workflow/status/geminitt/peresearch/ci.yml?branch=main&label=CI&style=for-the-badge)](https://github.com/geminitt/peresearch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/PYTHON-3.12-498AF2?style=for-the-badge)](./pixi.toml)
[![License](https://img.shields.io/badge/LICENSE-AGPL--3.0-6B7F4E?style=for-the-badge)](./LICENSE)

**A personal research agent: it first checks what *you* already know — your notes, files and folders — then
searches the web, and cites every claim.**

</div>

---

> **Status:** the retrieval layer (ZetokRAG) and the guard are done and measured; the agent, its tools, the
> terminal interface and the agent's evaluation are built and tested end to end on a laptop with a small model of
> the same family; the Modal deployment and the agent's measurements are next.

## How it works

| Part | What it does |
|---|---|
| **ZetokRAG** (zero-token RAG) | Finds evidence in your own files without calling an LLM: BM25 (on diacritic-free words only when the query is typed without them) plus Qwen3-Embedding-0.6B, min-max score fusion, reranking of the top 30 with bge-reranker-v2-m3, and a calibrated "enough / partial / nothing" verdict. Excerpts are quoted verbatim with their source. |
| **Agent** | One loop, written out rather than taken from a framework, running on your machine; only the model (Qwen3.6-35B-A3B, vLLM) runs on [Modal](https://modal.com). The model reads the message first and decides what it needs: a greeting gets an answer, no tools; a question about your work sends it through your folders (tree, search, grep, glob, read); before the web it records what your files cover and what is missing, and the web (Tavily, then Exa once Tavily's free credits run out) is searched for what is missing — the web search is refused until then. Limits on steps, searches, pages and time. When both your files and the web are used, the answer says which point comes from which; a rule-based check verifies every cited source id and every quote. |
| **Interface** | `peresearch chat`: a full-screen terminal interface (Textual) laid out like Claude Code's — the tools the agent calls appear as it works, a status line shows what it is doing, `/` opens a command menu (`/new`, `/sources`, `/folders`, `/help`), Enter sends and Ctrl+Enter starts a new line in a box that grows with the text, ↑/↓ recall earlier questions, PgUp/PgDn scroll, Esc interrupts, a pasted block of lines shows as `[Pasted text #1 +12 lines]` and reaches the model whole. It draws with the terminal's own color scheme, and Vietnamese input methods (UniKey, Telex) type correctly, tested with the bytes UniKey really sends. `peresearch ask` for one question. |
| **Guard** | Declared folders are indexed and read freely; anything else in your home folder is read only after you allow it in a dialog (once, or that folder for the session); hidden folders and credentials are never read or indexed, and everything that leaves the machine passes one filter. |

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

## Usage

```bash
pixi run peresearch setup      # once: pick the folders peresearch may read, index them, save the model and search keys
pixi run peresearch chat       # the agent in the terminal; it brings the index up to date when it opens
```

In `chat`, `/` opens the command menu: `/add <folder>` (Tab completes the path) and `/remove` change the folders
and reindex, `/folders` lists them, `/index` updates the index, `/new` starts a fresh conversation, `/sources`
shows the last answer's sources in full. The same exists as commands: `peresearch add | remove | folders | index`,
`peresearch find "…"` (your files only, no model call) and `peresearch ask "…"` (one question).

Settings live in `~/.local/share/peresearch/settings.env` (written by `setup`, `chmod 600`, never indexed):
`PERESEARCH_LLM_URL`, `PERESEARCH_LLM_KEY`, and optionally `TAVILY_API_KEY`, `EXA_API_KEY`, `JINA_API_KEY` — see
[`src/peresearch/settings.py`](src/peresearch/settings.py). Without a search key the agent works from your files
alone and says so. The local models run on the GPU when it has room and on the CPU otherwise.

**The model.** [`deploy/modal_vllm.py`](deploy/modal_vllm.py) serves Qwen3.6-35B-A3B-FP8 on one L40S: private
(a Modal proxy token is the API key), one container at most, stopped after five idle minutes, and a vLLM that
fails at start takes its container down within seconds instead of waiting out the startup timeout. The agent keeps
a daily spending cap for the endpoint (`PERESEARCH_BUDGET_USD_PER_DAY`, $1 unless set): it counts, as an upper
bound, every request plus the five idle minutes after it at the GPU's price, and stops calling the model at the cap.

Nothing is deployed before it has run on the laptop for free:

```bash
pixi run chat-local      # the same vLLM flags with Qwen3.5-0.8B (same chat and tool-call format) + the TUI
pixi run replica         # the Modal image rebuilt with Docker from the same spec, served, and asked for a tool call
pixi run modal-deploy    # runs the tests and the replica first; deploys only if both pass
```

---

## Guard: what can go wrong, and the test that shows it doesn't

| Risk | Handling | Test |
|---|---|---|
| Credentials read or sent out | Declared folders, plus what you allow when asked; hidden folders, anything outside your home folder and peresearch's own data refused without asking; symlinks resolved and must stay inside; protected names (`.ssh`, `.env`, token and key files, `.modal.toml`...) never read; paragraphs that look like a credential never indexed; everything leaving the machine redacted, search queries with a credential refused | `tests/test_canary.py`: fake credentials in 16 formats planted in notes, `.env`, key files, notebooks and behind symlinks; none reaches the index, the search results, the outbound text or the logs. `tests/test_agent.py`: none reaches the model, even from a file the agent reads in full |
| The model sending data out through a URL | `fetch` opens only URLs that a web search returned in the same question; every query and URL passes the secret filter | `tests/test_agent.py` |
| A runaway agent | Limits per question: 12 model calls, 4 web searches, 6 pages, 15 minutes; at a limit the model must answer with what it has | `tests/test_agent.py` |
| Broken or unreadable files | Reported by name, never fatal: binary, too large, unsupported, corrupt, PDF without a text layer ("needs OCR") | `tests/test_parse_chunk.py` |
| Quoting a file that changed | The file is re-checked before an excerpt is shown; an excerpt no longer in the file is dropped, one still present is flagged | `tests/test_index_search.py` |
| Instructions hidden in a document or a page | Every tool result reaches the model wrapped as untrusted data, and text that reads like an instruction is flagged; the tools are read-only | `tests/test_canary.py`, `tests/test_agent.py` |
| Terminal escape sequences in untrusted text | Stripped before printing | `tests/test_guard.py` |
| Logs holding sensitive text | The audit log records events and counts, never content | `tests/test_guard.py`, `tests/test_canary.py` |
| Changed models or packages | Hugging Face revisions pinned to commits; `pixi.lock` | `tests/test_supply_chain.py` |

---

## Measuring the agent

[`eval/research.py`](eval/research.py) runs the whole agent on four suites, checkpointed per question; results go
to `results/research.md` once the model runs on Modal (a pilot on the laptop's 0.8B model checked the harness,
not the agent).

| Suite | What it asks | Metric |
|---|---|---|
| [SimpleQA](https://openai.com/index/introducing-simpleqa/) | Short fact-seeking questions | Correct / incorrect / not attempted by the official grader prompt; correct given attempted; F-score |
| [FRAMES](https://huggingface.co/datasets/google/frames-benchmark) | Questions that need several Wikipedia pages | Accuracy, same grader |
| The owner's 85 questions | Files only | Evidence recall (the answer cites the labelled passage); abstention when the files cannot answer |
| Prompt injection | Instructions planted in a file or a web page | Attack success rate (AgentDojo/InjecAgent style), also over the runs where the planted text reached the model; utility under attack |
| Every answer | | Citation recall and precision as ALCE defines them, judged by a multilingual NLI model; model calls, tokens, time |

The grader must be a capable model: before grading a suite it has to get the grader prompt's own worked examples
right, or the run stops (the 0.8B pilot model was refused). A separate judge is set with `PERESEARCH_JUDGE_URL`.

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
