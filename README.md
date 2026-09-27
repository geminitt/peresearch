# peresearch

**A personal research agent: it first checks what *you* already know — your notes, files and folders — then
searches the web, and cites every claim.**

Status: under construction. Design:

- **ZetokRAG** (zero-token RAG) finds evidence in your own files without calling an LLM: BM25 plus dense
  embeddings, score fusion, reranking, and a calibrated "enough / partial / nothing" verdict. Excerpts are
  quoted verbatim with their source.
- **Agent**: one agent whose loop and tools run on your machine; the model (Qwen3.6-35B-A3B) runs on
  [Modal](https://modal.com). Your files come first, the web (free search APIs) second; answers separate
  "already in your notes", "new from the web" and the synthesis, each claim cited.
- **Guard**: only declared folders are read, credentials are never indexed, and everything that leaves the
  machine passes one filter.

## Development

```bash
pixi run test            # tests (the default environment includes PyTorch for the local models)
pixi run -e ci test      # the same tests without PyTorch, as CI runs them
```
