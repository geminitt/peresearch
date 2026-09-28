# ZetokRAG speed

Measured on Tesla T4 (one GPU at a time), fp16, batch 32, texts cut at 512 tokens. Indexing: 1024 sampled documents per corpus; querying: 100 queries per corpus, median (90th percentile) in milliseconds.

## Indexing (documents per second)

| corpus | docs | mean tokens | BGE-M3 | Qwen3-Embedding-0.6B | ratio |
|---|---:|---:|---:|---:|---:|
| scifact | 5183 | 360 | 51 | 23 | 2.2x |
| nfcorpus | 3633 | 377 | 50 | 21 | 2.4x |
| fiqa | 57638 | 180 | 100 | 41 | 2.5x |
| arguana | 8674 | 229 | 87 | 36 | 2.4x |
| scidocs | 25657 | 255 | 91 | 34 | 2.6x |
| scifact-vn | 5183 | 380 | 53 | 19 | 2.8x |
| nfcorpus-vn | 3618 | 400 | 51 | 19 | 2.7x |
| fiqa-vn | 57638 | 197 | 96 | 36 | 2.7x |
| arguana-vn | 8674 | 263 | 78 | 28 | 2.8x |
| scidocs-vn | 25657 | 288 | 71 | 27 | 2.6x |
| nano-nq-vn | 102195 | 137 | 141 | 51 | 2.8x |
| nano-hotpotqa-vn | 111653 | 86 | 247 | 89 | 2.8x |
| nano-fever-vn | 101308 | 124 | 164 | 63 | 2.6x |
| nano-dbpedia-vn | 131815 | 91 | 209 | 73 | 2.9x |
| nano-climate-fever-vn | 101097 | 128 | 162 | 60 | 2.7x |
| nano-msmarco-vn | 103655 | 89 | 232 | 83 | 2.8x |
| zalo-legal-vn | 61425 | 262 | 73 | 23 | 3.1x |
| **mean** | | | 115 | 43 | 2.7x |

## Answering one query with ZetokRAG (ms)

| corpus | embedder | embed query | BM25 | dense + fusion | rerank top 30 | total |
|---|---|---:|---:|---:|---:|---:|
| scifact | bge-m3 | 20 (21) | 0 (0) | 1 (2) | 731 (777) | 753 (800) |
| scifact | qwen3-embedding-0.6b | 51 (53) | 0 (0) | 2 (2) | 729 (760) | 782 (814) |
| nfcorpus | bge-m3 | 20 (21) | 0 (0) | 1 (1) | 737 (762) | 758 (788) |
| nfcorpus | qwen3-embedding-0.6b | 52 (53) | 0 (0) | 1 (1) | 733 (747) | 786 (799) |
| fiqa | bge-m3 | 20 (20) | 1 (1) | 10 (10) | 690 (757) | 719 (787) |
| fiqa | qwen3-embedding-0.6b | 51 (53) | 1 (1) | 10 (10) | 687 (710) | 748 (772) |
| arguana | bge-m3 | 24 (31) | 3 (4) | 2 (2) | 791 (831) | 820 (863) |
| arguana | qwen3-embedding-0.6b | 55 (83) | 3 (4) | 2 (2) | 774 (817) | 835 (898) |
| scidocs | bge-m3 | 19 (20) | 0 (0) | 5 (5) | 544 (603) | 568 (627) |
| scidocs | qwen3-embedding-0.6b | 49 (51) | 0 (0) | 5 (5) | 640 (681) | 695 (735) |
| scifact-vn | bge-m3 | 19 (21) | 0 (0) | 1 (2) | 721 (779) | 746 (800) |
| scifact-vn | qwen3-embedding-0.6b | 50 (51) | 0 (0) | 1 (2) | 725 (751) | 777 (803) |
| nfcorpus-vn | bge-m3 | 19 (20) | 0 (0) | 1 (1) | 725 (763) | 745 (783) |
| nfcorpus-vn | qwen3-embedding-0.6b | 50 (51) | 0 (0) | 1 (1) | 726 (739) | 777 (792) |
| fiqa-vn | bge-m3 | 20 (21) | 1 (2) | 10 (10) | 715 (779) | 746 (812) |
| fiqa-vn | qwen3-embedding-0.6b | 52 (53) | 1 (2) | 10 (10) | 714 (751) | 776 (814) |
| arguana-vn | bge-m3 | 25 (33) | 4 (6) | 2 (2) | 816 (855) | 848 (895) |
| arguana-vn | qwen3-embedding-0.6b | 68 (99) | 4 (7) | 2 (2) | 813 (840) | 893 (945) |
| scidocs-vn | bge-m3 | 19 (20) | 1 (1) | 5 (5) | 689 (743) | 714 (766) |
| scidocs-vn | qwen3-embedding-0.6b | 49 (50) | 1 (1) | 5 (5) | 693 (712) | 748 (768) |
| nano-nq-vn | bge-m3 | 19 (20) | 1 (1) | 16 (17) | 558 (700) | 594 (737) |
| nano-nq-vn | qwen3-embedding-0.6b | 50 (51) | 1 (1) | 16 (17) | 581 (684) | 646 (752) |
| nano-hotpotqa-vn | bge-m3 | 18 (20) | 1 (2) | 18 (18) | 403 (509) | 441 (546) |
| nano-hotpotqa-vn | qwen3-embedding-0.6b | 48 (50) | 1 (2) | 18 (19) | 366 (508) | 436 (576) |
| nano-fever-vn | bge-m3 | 18 (19) | 1 (1) | 16 (17) | 618 (687) | 654 (721) |
| nano-fever-vn | qwen3-embedding-0.6b | 50 (52) | 1 (1) | 16 (17) | 624 (663) | 692 (730) |
| nano-dbpedia-vn | bge-m3 | 18 (19) | 1 (1) | 21 (21) | 221 (319) | 261 (359) |
| nano-dbpedia-vn | qwen3-embedding-0.6b | 49 (52) | 1 (1) | 21 (22) | 207 (335) | 282 (405) |
| nano-climate-fever-vn | bge-m3 | 19 (20) | 1 (3) | 16 (17) | 702 (756) | 740 (795) |
| nano-climate-fever-vn | qwen3-embedding-0.6b | 50 (51) | 1 (3) | 16 (17) | 710 (738) | 779 (809) |
| nano-msmarco-vn | bge-m3 | 18 (19) | 0 (1) | 17 (17) | 233 (297) | 268 (332) |
| nano-msmarco-vn | qwen3-embedding-0.6b | 48 (50) | 0 (1) | 17 (17) | 213 (272) | 280 (337) |
| zalo-legal-vn | bge-m3 | 20 (21) | 1 (2) | 10 (10) | 782 (857) | 814 (897) |
| zalo-legal-vn | qwen3-embedding-0.6b | 52 (53) | 1 (2) | 10 (11) | 771 (833) | 834 (897) |
