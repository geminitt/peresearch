<div align="center">

# peresearch

[![CI](https://img.shields.io/github/actions/workflow/status/geminitt/peresearch/ci.yml?branch=main&label=CI&style=for-the-badge)](https://github.com/geminitt/peresearch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/PYTHON-3.12-498AF2?style=for-the-badge)](./pixi.toml)
[![License](https://img.shields.io/badge/LICENSE-AGPL--3.0-6B7F4E?style=for-the-badge)](./LICENSE)

**A personal research agent: it first checks what *you* already know — your notes, files and folders — then
searches the web, and cites the sources it uses.**

</div>

---

> **Status:** the retrieval layer (ZetokRAG) and access controls are implemented and tested; the agent, its tools, the
> terminal interface and the agent's evaluation are built and tested end to end with small local models; the model
> gateway is checked against Ollama, llama.cpp and vLLM; the agent's measurements on a strong model are next.

## How it works

| Part | What it does |
|---|---|
| **ZetokRAG** (zero-token RAG) | Finds evidence in your own files without calling an LLM: BM25 (on words as typed), plus — when you set them — the embeddings and the reranker of your own endpoints (min-max score fusion, the top 30 reranked). No model runs inside peresearch; with no endpoint the search goes by words alone. Excerpts are quoted verbatim with their source, ranked, never cut by a score. |
| **Agent** | One loop, written out rather than taken from a framework, running on your machine; the model is the one you choose, reached through an OpenAI-compatible API (see
[The model](#the-model)). The model reads the message first and decides what it needs: a greeting gets an answer, no tools; a question about your work sends it through your folders (tree, search, an overview of the documents on a topic, grep, glob, read); before the web it records what your files cover and what is missing, and the web (through the search server you add, below) is searched for what is missing — the web search is refused until then. Limits on steps, searches and pages; the time budget is checked between rounds. When both your files and the web are used, the answer says which point comes from which; a rule-based check verifies cited source IDs and nearby quoted passages. It does not establish factual support for every claim. |
| **Interface** | `peresearch` opens the Textual TUI. Floating windows group folders, models and APIs, settings, status and conversations. Controls use local shortcuts and keep the transcript for research. Enter sends; Ctrl+Enter adds a line; @path and Tab complete file mentions; Esc interrupts; Ctrl+Q quits. Citations open source previews with safe file/page links. The terminal palette, UniKey input, multiline paste, reasoning display and text selection have dedicated tests. `peresearch ask` remains available for one scripted question. |
| **Guard** | Declared folders are indexed and read freely; anything else in your home folder is read only after you allow it in a dialog (once, or that folder for the session); hidden folders and credentials are never read or indexed, and everything that leaves the machine passes one filter. |

Access is checked again before retained results, answers or summaries reach the model. A follow-up answer inherits
the file scope of the history shown to the model, even when it calls no tools. Removing a dependency withholds
that answer from later model requests. Historical records remain on disk. Older records without inherited
provenance are checked conservatively.

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
pixi run peresearch            # open the TUI; setup and research happen in this interface
```

The installed command is `peresearch`, with no subcommand (`peresearch chat` remains an alias).
The first launch without a model endpoint opens the model window. Saving settings or an API connection is local.
`chat-local` starts a development test server; normal use does not need that launcher.

Type `/` to choose a group; the command menu opens under the input box. Controls open in windows sized to their
content; they do not print configuration into the conversation. Windows have no buttons: lists show the chosen row with
`❯`, fields are one line each (label, then value), and a dim line at the bottom names the keys. Shortcuts below work
only inside their window. Esc closes the current window and returns to the previous one.

| Command | Window and controls |
|---|---|
| `/folders` | Declared folders: A adds a path, R removes the selected folder, I updates the index. Changes use the existing access rules. |
| `/model` | Chat, embedding and reranker sections (← → switch), each with saved model IDs. Enter selects; A connects a model; D lists model IDs again; M enters a known ID; O disables an optional role. |
| `/settings` | Research limits and search services in one list. Tab moves between fields; Enter or Ctrl+S saves. Empty limits reset defaults; empty saved-key fields keep credentials. |
| `/status` | Setup/index, usage and context tabs. |
| `/sessions` | Saved conversations. Enter resumes; N starts a new one; R names the current one; E exports it as Markdown. |
| `/check` | Service checks run only when R is pressed. Requests can use paid services; opening the window sends nothing. |
| `/sources` | Source locations and excerpts, up to 600 characters each. Citation links open a source window. |
| `/help` | Command and keyboard reference. |

Connecting a model (A) takes three steps, as in OpenCode: choose where it runs (a saved connection, OpenAI, OpenRouter,
Groq, Ollama, LM Studio, another local server or any OpenAI-compatible endpoint); give only what that place needs (a
cloud's API key, a local server's address); choose the model from the IDs the endpoint lists, typing to filter them.
Keys are hidden. Typing sends nothing; Enter on the second step saves the connection and requests its model IDs
without sending a prompt. Ctrl+T types an ID when the endpoint lists none. The chosen model becomes the role's. Listing IDs does not prove chat, embedding, reranking or tool-call compatibility.
Servers without a model-list API can use a manually entered ID. Active service settings stay in `settings.env`;
saved connections and IDs stay in the private `models.json` catalog. Environment settings override saved selections.
`/new`, `/retry`, `/local` and `/exit` remain immediate actions. `/local` turns web tools off; configured remote model
and retrieval endpoints can still receive text. Older control names route to the corresponding window.

Source IDs identify immutable excerpt versions across a conversation.excerpt versions across a conversation. A fresh shorter read gets a new ID; old answers keep their original evidence.
Resume verifies excerpt hashes and links to retained tool results without storing the large bodies twice.
Legacy or damaged identities require a new read; current access controls apply to retained file previews.
Explicit index refresh and converted grep verify current file hashes even when size and modification time match.
When all permitted vectors are missing, retrieval uses lexical evidence and reports the reason.
The same exists as commands: `peresearch add | remove | folders | index`,
`peresearch find "…"` (your files only, no model call) and `peresearch ask "…"` (one question).

Removing a folder revokes its access before asynchronous index cleanup finishes. Changing embedding settings
applies to the next question; changing the model or endpoint keeps indexed text but requires an index update (I in `/folders`) to rebuild
vectors. Searches use words alone until compatible vectors are available.

Stored file results, earlier answers and summaries are checked against current access before being sent to the
model again. Revoked evidence is withheld without deleting the user's historical records. A once-only grant
allows continuation of that result within the current question; a later question needs current permission.

Settings live in `~/.local/share/peresearch/settings.env` (written by the TUI or CLI setup, mode 600, never indexed):
`PERESEARCH_LLM_URL`, `PERESEARCH_LLM_MODEL`, `PERESEARCH_LLM_KEY`, optionally the search's endpoints
(`PERESEARCH_EMBED_URL`/`_MODEL`/`_KEY`, `PERESEARCH_RERANK_URL`/`_MODEL`/`_KEY`), the web search
(`PERESEARCH_SEARCH_MCP`, `PERESEARCH_SEARCH_MCP_KEY`) and `JINA_API_KEY` — see
[`src/peresearch/settings.py`](src/peresearch/settings.py).

The Chat Completions gateway selects the reply-budget field from the model's capability. Known OpenAI chat
families use `max_completion_tokens`; local and unknown profiles keep `max_tokens`. A server's explicit rejection
allows one alternate field. Other request errors do not disable unrelated capabilities. Structured refusals retain
their reason and are shown as answer text; tool-call examples inside a refusal are not executed.

### The web

What ships: scholarly search (arXiv, Semantic Scholar, and OpenAlex with a free key), Wikipedia in any language, and
reading a page or a paper's PDF on this machine. A general web search is one you add: any **MCP server** with a
search tool, as a command (stdio — e.g. `npx -y tavily-mcp`, a DuckDuckGo or SearXNG server) or a URL (Streamable
HTTP — e.g. `https://mcp.exa.ai/mcp`), with its key as `NAME=value` when it needs one. peresearch calls its search
tool through the tool's own schema (no adapter per provider) and makes each link in what it returns a source the
answer can cite and `fetch` can read. Without one, the agent works from your files and the scholarly sources, and
says so. Checked on real servers: DuckDuckGo over stdio and HTTP, Exa's hosted server
([`runs/serve/mcp-servers-2026-10-08.py`](runs/serve/mcp-servers-2026-10-08.py)).

After failed initialization, the MCP client closes owned resources and retries the complete handshake before
requesting tools. An injected client remains owned by its caller.

A stdio search process inherits runtime settings, but unrelated credential variables and peresearch settings
are removed. Its explicitly configured service variables are still passed. This does not sandbox a same-user
process or prevent it from accessing files through its own operating system permissions.

A page the local reader cannot read (one built in JavaScript) goes to the Jina Reader, a third party, only when you
set `JINA_API_KEY`: on 30 sample pages the local reader read 25 and Jina 2 more
([`runs/e2e/page-reader-2026-10-08.py`](runs/e2e/page-reader-2026-10-08.py)).

### The search's endpoints

Searching your files by meaning needs an embedding model, and ranking them best needs a reranker; peresearch runs
neither itself. Point it at your own: an **embeddings endpoint** (a `/v1` serving `/embeddings` — Ollama, llama.cpp
with `--embedding`, vLLM with `--runner pooling`, LM Studio or a provider) and, optionally, a **rerank endpoint** (a
`/v1` serving `/rerank` — vLLM, llama.cpp with `--reranking`, Text Embeddings Inference, Jina, Cohere).
TUI `/check` says what each one answers and, when it is not on your machine, which host your passages go to.

| Set | What the search does | nDCG@10, gold dev set |
|---|---|---:|
| nothing | words alone (BM25); a question in other words or another language finds less | 0.336 |
| embeddings (bge-m3 through Ollama) | words and meaning, fused | 0.546 |
| embeddings + rerank (bge-reranker-v2-m3 through vLLM) | the top 30 reordered, each with a relevance score | 0.648 |

Measured through peresearch's own path on the laptop's GPU ([`runs/serve/endpoints-2026-10-08.py`](runs/serve/endpoints-2026-10-08.py));
the last row ties the local models peresearch used to run (0.650; difference −0.002, 95% interval [−0.049, +0.040]).

Passages are embedded after the files are indexed, a batch at a time: until then, and whenever the endpoint does
not answer, the search goes on by words and says so. When an endpoint rejects an overlong input, peresearch
retries with its head and counts that shortening in the index report. Some servers can shorten inputs themselves
without reporting it to the client; the count cannot detect that. In particular, Ollama defaults to truncation
([API documentation](https://docs.ollama.com/api/embed)). Full source text remains available to lexical search.

### The model

peresearch reaches a model one way: an **OpenAI-compatible chat API** (a base URL ending in `/v1`). Behind it can be
a provider's API, or a model you serve yourself on this machine or another — Ollama, llama.cpp's `llama-server`, LM
Studio, vLLM. `/model` groups model roles and APIs. Add a connection with A, then select a listed or manually
entered model ID with Enter. A server that exposes only one model can also retain the CLI's empty-model setting. `/check` explicitly tests the model after confirmation: whether and how it calls tools, where its reasoning comes back, the context it runs with, its speed.

| Where the model is | Endpoint, for example | Model name |
|---|---|---|
| Ollama | `http://localhost:11434/v1` | as `ollama list` shows it, e.g. `qwen3:8b` |
| llama.cpp `llama-server` | `http://localhost:8080/v1` (the port it was started on) | empty: it serves one model |
| vLLM | `http://localhost:8000/v1` | empty, or its `--served-model-name` |
| A provider | the provider's OpenAI-compatible URL, with its API key | the provider's name for the model |

Models differ in what they send back; peresearch handles the differences, not the user:

| Difference | What peresearch does |
|---|---|
| Tool calls | Native tool calls when the model and server make them; otherwise the tools are described in the prompt and protocol-shaped calls recovered from text. Bare JSON embedded in prose and fenced examples remain answer data. Chosen from a model table (a trimmed [models.dev](https://models.dev) snapshot), and switched when the server refuses `tools` or its own tool-call parser fails on the model's output |
| Arguments | Read by each tool's schema type: a number written as a string still works |
| Reasoning | Kept apart from the answer, whether it comes in its own field or inside the text |
| Context | The context the server runs the model with (vLLM, llama.cpp and Ollama report it), else the model table's, else `PERESEARCH_LLM_CONTEXT`; a conversation the server refuses for its length is trimmed and sent again |
| Sizes | None fixed: every tool returns its whole output, and the agent shows the model what fits the room its context leaves — a longer result as its head and a note saying where to read on (`recall`), the whole kept; earlier turns go back whole, and when they no longer fit, the model summarizes the oldest ones (the summary is kept with the conversation, the turns too); the reply's budget is the smallest the model's table, its context and the caller allow, none when nothing is known |
| Sampling | The model family's recommended settings when known, else the server's own |

Checked on real servers with small models (the exchanges are recorded in `tests/data/servers/` and replayed by the
tests): Ollama 0.32.15 (qwen3:0.6b, llama3.2:1b, gemma3:1b), llama.cpp (Qwen3-0.6B, Llama-3.2-1B-Instruct, with and
without `--jinja`), vLLM 0.29.0 (Qwen3.5-0.8B, as served and without its tool-call or reasoning parser). Not tried
yet: LM Studio, a provider's API.

> **Note:** Ollama cuts a conversation longer than the model's context without an error, from its start (where the
> agent's instructions are). peresearch reads the context Ollama runs the model with and fits the conversation
> to it; give Ollama a larger one with `OLLAMA_CONTEXT_LENGTH` (its default depends on the GPU's memory).

**Test tools** (the author's, not part of peresearch): [`deploy/modal_vllm.py`](deploy/modal_vllm.py) serves
Qwen3.6-35B-A3B-FP8 with vLLM on [Modal](https://modal.com) for the agent's measurements; `pixi run chat-local`
runs vLLM with Qwen3.5-0.8B on the laptop and the TUI; `pixi run replica` rebuilds the Modal image with Docker and
checks it; `pixi run modal-deploy` deploys only after the tests and the replica pass;
`runs/serve/compat-matrix-2026-10-06.py` (pixi environment `servers`) runs the checks above on Ollama, llama.cpp and
vLLM.

---

## Guard: what can go wrong, and the test that shows it doesn't

| Risk | Handling | Test |
|---|---|---|
| Credentials read or sent out | Declared folders, plus what you allow when asked; hidden folders, anything outside your home folder and peresearch's own data refused without asking; symlinks resolved and must stay inside; protected names (`.ssh`, `.env`, token and key files, `.modal.toml`...) never read; paragraphs that look like a credential (16 formats of our own and gitleaks' rules for about two hundred providers) never indexed; everything leaving the machine redacted, search queries with a credential refused | `tests/test_canary.py`: fake credentials in 16 formats planted in notes, `.env`, key files, notebooks and behind symlinks; none reaches the index, the search results, the outbound text or the logs. `tests/test_agent.py`: none reaches the model, even from a file the agent reads in full |
| The model sending data out through a URL | `fetch` opens only URLs that a web search returned in the same question; every query and URL passes the secret filter | `tests/test_tools.py` |
| A runaway agent | Limits per question: 12 model calls, 4 web searches, 6 pages, 15 minutes; at a limit the model must answer with what it has | `tests/test_agent.py` |
| Broken or unreadable files | Reported by name, never fatal: binary, too large, unsupported, corrupt, PDF without a text layer ("needs OCR") | `tests/test_parse_chunk.py` |
| Quoting a file that changed | The file is re-checked before an excerpt is shown; removed or ambiguous passages are dropped. A surviving passage gets its current line numbers, a change warning and no indexed neighbors; near-copy locations are checked too | `tests/test_index_search.py`, `tests/test_stabilization.py` |
| Instructions hidden in a document or a page | Every tool result reaches the model wrapped as untrusted data, and text that reads like an instruction is flagged; the tools are read-only | `tests/test_canary.py`, `tests/test_agent.py` |
| Terminal escape sequences in untrusted text | Stripped before printing | `tests/test_guard.py` |
| Logs holding sensitive text | The audit log records events and counts, never content | `tests/test_guard.py`, `tests/test_canary.py` |
| Changed models or packages | Hugging Face revisions pinned to commits; `pixi.lock` | `tests/test_supply_chain.py` |

---

## Measuring the agent

[`eval/research.py`](eval/research.py) runs the whole agent on four suites, checkpointed per question; results go
to `results/research.md` once they run on a strong model (a pilot on a 0.8B model checked the harness, not the
agent).

Product and evaluation use the same assembly, tools and configured limits. Each pending evaluation question
gets a fresh agent. Evaluation disables session recording; personal questions also disable web and scholarly tools
and use explicit corpus roots. Injected components remain caller-owned; default request resources close after each
question, including failures.

Each checkpoint has a separate report section and a non-secret manifest with source/configuration fingerprints,
package versions and pinned dataset references. Historical checkpoints without a manifest retain an unknown model
label. Changing questions, configuration, source or the personal corpus starts a new checkpoint; server weights
changed behind the same model name remain unknown and require a fresh run directory.

| Suite | What it asks | Metric |
|---|---|---|
| [SimpleQA](https://openai.com/index/introducing-simpleqa/) | Short fact-seeking questions | Correct / incorrect / not attempted by the official grader prompt; correct given attempted; F-score |
| [FRAMES](https://huggingface.co/datasets/google/frames-benchmark) | Questions that need several Wikipedia pages | Accuracy, same grader |
| The owner's 85 questions | Files only | Evidence recall (the answer cites the labelled passage); abstention when the files cannot answer |
| Prompt injection | Instructions planted in a file or a web page | Attack success rate (AgentDojo/InjecAgent style), also over the runs where the planted text reached the model; utility under attack |
| Every answer | | ALCE-style citation recall over all sentences and citation precision, averaged per nonempty answer and judged by a multilingual NLI model; model calls, tokens, time |

The grader must be a capable model: before grading a suite it has to get the grader prompt's own worked examples
right, or the run stops (the 0.8B pilot model was refused). Invalid judge replies also stop the run, without
counting an abstention. A separate judge is set with `PERESEARCH_JUDGE_URL`.

Citation scoring approximates [ALCE](https://github.com/princeton-nlp/ALCE/blob/main/eval.py): it uses a simple
sentence splitter and a smaller multilingual NLI model with 512-token inputs. Citation metric version 3 attaches
citations after sentence punctuation to that preceding sentence, including adjacent groups and Markdown emphasis.
A sentence with an unknown source
ID gets no support credit; those citations still count against precision. No citations gives zero, empty answers
are counted separately, and per-answer averages have no binary-proportion confidence interval. Historical records
without the corrected metric version retain unavailable citation scores; their old counts cannot recover uncited
sentences or missing source text. Retained sources marked unknown cannot support a sentence.

Injection exposure version 2 observes the complete verbatim payload in messages submitted at the model gateway,
after filtering and context handling, including summary calls and retries. A registered source or a short visible
prefix is insufficient. This measures submitted requests, not a provider's internal processing or paraphrased attack
meaning. Exposure is unknown for injected models that bypass the gateway. Checkpoint schema 2 records these methods
and isolation; earlier records remain unchanged and are excluded from corrected metric counts.

---

## Reproduce the measurements

```bash
# public corpora: two Kaggle kernels, 2 x T4 each (5.2 h and 4.4 h), resumable
python kaggle/push.py retrieval --job full-shards-0-1 --set 'SHARDS=[0, 1]'
python kaggle/push.py retrieval --job full-shards-2-3 --set 'SHARDS=[2, 3]'
# with both kernels' runs/ outputs downloaded into $PERESEARCH_RUNS:
pixi run python eval/retrieval.py report  # results/retrieval.md
pixi run python eval/speed.py report      # results/speed.md
```

---

## Development

```bash
pixi run test            # tests
pixi run -e ci test      # the same tests as CI runs them
pixi run -e models test  # with PyTorch: the local models of the retrieval benchmarks (test tools)
```

## Development checks

`pixi run test` runs the default suite. `pixi run gate` stops on the first failure across default, ci and a fresh
Docker installation (unprivileged user, offline tests, 4 CPUs and 4 GB). `pixi run gate --passes 2` runs two
consecutive complete passes and rejects a source change between checks. Logs and an fsynced checkpoint live under
`runs/checks/`; the command prints how to resume an interrupted gate with the same inputs. Each test has a
300-second watchdog (`pytest --test-timeout SECONDS`); suite commands and Docker operations also have deadlines.
Docker installation needs network access. No model or paid service is required by the gate.

`bash scripts/e2e_cycle.sh <label> <script> [args...]` runs an E2E with the local test server when needed.
It preserves the child exit status, enforces startup/run deadlines, and stops only process groups it starts,
including on interruption. The server is a test tool; an already running server is left under its owner's control.
