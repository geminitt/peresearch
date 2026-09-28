# ZetokRAG on the owner's own files

85 questions (60 answerable, 25 about topics absent from the files) over 8,059 chunks of notes, projects and course material; the questions and passages stay on the owner's machine, only these aggregate numbers are published. Strict scoring: a hit is a chunk from the labelled file that overlaps the labelled lines (pages, cells). The interval is a paired bootstrap over questions (10,000 resamples, 95%).

| method | hit@1 | hit@5 | hit@10 | nDCG@10 | ZetokRAG minus it [95% interval] |
|---|---:|---:|---:|---:|---|
| bm25 | 61.7% | 83.3% | 91.7% | 0.758 | +0.091 [+0.017, +0.171] |
| bm25-auto | 61.7% | 83.3% | 91.7% | 0.757 | +0.093 [+0.019, +0.172] |
| dense-bge-m3 | 68.3% | 91.7% | 95.0% | 0.818 | +0.032 [-0.039, +0.103] |
| dense-qwen3 | 61.7% | 95.0% | 98.3% | 0.818 | +0.032 [-0.037, +0.102] |
| hybrid (no rerank) | 66.7% | 95.0% | 98.3% | 0.827 | +0.023 [-0.039, +0.089] |
| dense-bge-m3+rerank | 68.3% | 91.7% | 96.7% | 0.837 | +0.013 [-0.022, +0.050] |
| dense-qwen3+rerank | 68.3% | 95.0% | 100.0% | 0.856 | -0.006 [-0.018, +0.000] |
| bm25-auto+rerank | 66.7% | 91.7% | 95.0% | 0.822 | +0.027 [-0.007, +0.074] |
| zetokrag-v0 | 68.3% | 93.3% | 98.3% | 0.846 | +0.004 [-0.014, +0.025] |
| zetokrag-bge | 68.3% | 93.3% | 98.3% | 0.846 | +0.004 [-0.014, +0.025] |
| **zetokrag** | 68.3% | 95.0% | 98.3% | 0.850 |  |

Counting the same passage kept in another file as found (word overlap with the labelled passage of at least 80%, Jaccard), ZetokRAG reaches hit@1 76.7% and nDCG@10 0.885.

## Does the index hold an answer?

ZetokRAG's verdict compares the reranker's best score with two thresholds. Intervals are exact (Clopper-Pearson) 95%.

Best score: answerable questions 0.559–0.999 (median 0.973), unanswerable 0.001–0.160 (median 0.004); AUROC 1.000.

| threshold | unanswerable rejected | answerable wrongly rejected |
|---|---|---|
| PARTIAL = 0.25 | 25/25 [86.3%, 100.0%] | 0/60 [0.0%, 6.0%] |
| ENOUGH = 0.5 | 25/25 [86.3%, 100.0%] | 0/60 [0.0%, 6.0%] |

The thresholds were set on this same question set, so these rejection rates are in-sample.
