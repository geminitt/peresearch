# Choosing ZetokRAG's ranking core: six languages

28 set-settings scored in full, 3 with BM25 only (MLDR). nDCG@10 unless stated; a language is the mean of its sets; `mean 6` counts Vietnamese once (the mean of as typed and without diacritics). Sets, revisions and variants are defined in eval/multilingual.py.

## Ranking cores by language

| core | models | en | vi | vi-no-diacritics | zh | ja | ko | hi | mean 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zetokrag with bm25 unspaced-xlmr | 3 | 48.5 | 71.7 | 58.1 | 63.7 | 83.8 | 81.2 | 79.1 | **70.2** |
| bge-m3-hybrid + rerank | 2 | 48.2 | 71.9 | 42.6 | 63.8 | 84.4 | 83.1 | 78.8 | **69.3** |
| qwen3 + rerank | 2 | 48.0 | 70.5 | 44.5 | 64.4 | 83.3 | 81.6 | 77.4 | **68.7** |
| bge-m3-hybrid | 1 | 45.2 | 68.3 | 36.4 | 60.5 | 80.7 | 78.6 | 75.9 | **65.5** |
| fusion unspaced-hangul-xlmr rho=0.3 | 2 | 50.1 | 67.4 | 43.7 | 59.6 | 76.2 | 75.0 | 73.0 | **64.9** |
| fusion unspaced-bigrams rho=0.3 | 2 | 50.1 | 67.4 | 43.7 | 60.3 | 76.4 | 73.5 | 73.0 | **64.8** |
| fusion unspaced-xlmr rho=0.3 | 2 | 50.1 | 67.4 | 43.7 | 59.6 | 76.2 | 73.5 | 73.0 | **64.6** |
| fusion xlmr-whole rho=0.3 | 2 | 49.5 | 67.1 | 40.9 | 59.6 | 76.3 | 74.5 | 72.8 | **64.4** |
| fusion current rho=0.3 | 2 | 50.1 | 67.4 | 43.7 | 59.7 | 73.5 | 73.5 | 71.6 | **64.0** |
| fusion unspaced-hangul-xlmr rho=0.5 guarded | 2 | 47.6 | 66.2 | 55.8 | 52.7 | 74.2 | 74.5 | 72.7 | **63.8** |
| fusion unspaced-hangul-xlmr rho=0.5 | 2 | 47.5 | 66.2 | 55.8 | 52.7 | 74.2 | 74.5 | 72.7 | **63.8** |
| qwen3 | 1 | 49.4 | 64.9 | 38.4 | 60.6 | 75.5 | 72.3 | 69.5 | **63.2** |
| fusion unspaced-xlmr rho=0.5 guarded | 2 | 47.6 | 66.2 | 55.8 | 52.7 | 74.2 | 69.2 | 72.7 | **62.9** |
| fusion unspaced-bigrams rho=0.5 guarded | 2 | 47.6 | 66.2 | 55.8 | 52.8 | 73.7 | 69.2 | 72.7 | **62.8** |
| fusion unspaced-xlmr rho=0.5 | 2 | 47.5 | 66.2 | 55.8 | 52.7 | 74.2 | 68.9 | 72.7 | **62.8** |
| fusion unspaced-bigrams rho=0.5 | 2 | 47.5 | 66.2 | 55.8 | 52.8 | 73.7 | 68.9 | 72.7 | **62.8** |
| fusion current rho=0.5 guarded | 2 | 47.6 | 66.2 | 55.8 | 58.7 | 71.8 | 69.2 | 67.1 | **62.6** |
| fusion xlmr-whole rho=0.5 | 2 | 46.0 | 65.0 | 42.3 | 51.9 | 74.1 | 74.0 | 72.5 | **62.0** |
| fusion xlmr-whole rho=0.5 guarded | 2 | 46.0 | 65.0 | 42.3 | 51.9 | 74.1 | 74.0 | 72.5 | **62.0** |
| rrf unspaced-hangul-xlmr | 2 | 45.0 | 63.8 | 52.1 | 47.7 | 71.4 | 71.9 | 70.3 | **60.7** |
| rrf unspaced-xlmr | 2 | 45.0 | 63.8 | 52.1 | 47.7 | 71.4 | 62.9 | 70.3 | **59.2** |
| rrf xlmr-whole | 2 | 43.1 | 62.2 | 41.4 | 46.1 | 71.6 | 71.9 | 70.6 | **59.2** |
| rrf unspaced-bigrams | 2 | 45.0 | 63.8 | 52.1 | 46.8 | 71.1 | 62.9 | 70.3 | **59.0** |
| fusion unspaced-hangul-xlmr rho=0.7 | 2 | 41.9 | 62.4 | 57.0 | 41.5 | 69.5 | 68.8 | 68.0 | **58.2** |
| fusion unspaced-xlmr rho=0.7 | 2 | 41.9 | 62.4 | 57.0 | 41.5 | 69.4 | 58.1 | 68.0 | **56.4** |
| fusion unspaced-bigrams rho=0.7 | 2 | 41.9 | 62.4 | 57.0 | 38.8 | 68.3 | 58.0 | 68.0 | **55.8** |
| fusion xlmr-whole rho=0.7 | 2 | 38.9 | 59.9 | 38.9 | 39.6 | 69.0 | 68.5 | 68.4 | **55.6** |
| zetokrag (as used) | 3 | 48.5 | 71.7 | 58.1 | 8.0 | 27.6 | 81.3 | 77.8 | **51.4** |
| zetokrag, rerank top 50 | 3 | 47.1 | 71.9 | 57.3 | 8.0 | 28.0 | 82.0 | 78.1 | **51.3** |
| rrf current | 2 | 45.0 | 63.8 | 52.1 | 36.6 | 35.4 | 62.9 | 63.2 | **50.2** |
| fusion current rho=0.5 | 2 | 47.5 | 66.2 | 55.8 | 5.5 | 21.7 | 68.8 | 67.1 | **45.3** |
| fusion current rho=0.7 | 2 | 41.9 | 62.4 | 57.0 | 3.6 | 17.9 | 58.1 | 56.0 | **39.5** |

## Decision by the rule set beforehand

Best mean over the six languages; a core with fewer models within 1 point of the best wins; no language may fall more than 1 point below `zetokrag (as used)`.

- best mean: `zetokrag with bm25 unspaced-xlmr` (70.2); ZetokRAG as used: 51.4
- cores that keep every language within 1 point: 2 of 32
- **pick: `zetokrag with bm25 unspaced-xlmr`** (70.2, 3 models)

32 cores are compared on the same queries, so the best of them is favoured by chance: the intervals below are per comparison, not corrected for the number of comparisons.

## Each core minus `zetokrag (as used)` (nDCG@10 points, paired bootstrap 95% interval)

| core | en | vi | vi-no-diacritics | zh | ja | ko | hi |
|---|---|---|---|---|---|---|---|
| zetokrag with bm25 unspaced-xlmr | +0.0 [+0.0, +0.0] | -0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.0] | +55.7 [+54.2, +57.1] | +56.2 [+54.3, +58.0] | -0.1 [-0.3, +0.0] | +1.3 [+0.7, +2.0] |
| bge-m3-hybrid + rerank | -0.4 [-0.8, +0.1] | +0.2 [-0.5, +0.8] | -15.4 [-16.6, -14.3] | +55.8 [+54.3, +57.3] | +56.8 [+55.0, +58.7] | +1.7 [+0.9, +2.6] | +1.0 [+0.3, +1.8] |
| qwen3 + rerank | -0.6 [-1.0, -0.2] | -1.1 [-1.6, -0.7] | -13.6 [-14.7, -12.5] | +56.4 [+55.0, +57.9] | +55.6 [+53.7, +57.5] | +0.3 [-0.9, +1.3] | -0.4 [-1.1, +0.3] |
| bge-m3-hybrid | -3.4 [-4.2, -2.6] | -3.4 [-4.5, -2.3] | -21.7 [-23.1, -20.2] | +52.5 [+50.9, +54.0] | +53.1 [+51.2, +55.0] | -2.8 [-4.3, -1.3] | -1.9 [-2.9, -0.8] |
| fusion unspaced-hangul-xlmr rho=0.3 | +1.6 [+0.9, +2.3] | -4.3 [-5.3, -3.2] | -14.4 [-15.7, -13.1] | +51.6 [+50.0, +53.1] | +48.6 [+46.7, +50.5] | -6.4 [-8.2, -4.7] | -4.8 [-6.0, -3.7] |
| fusion unspaced-bigrams rho=0.3 | +1.6 [+0.8, +2.3] | -4.3 [-5.3, -3.2] | -14.4 [-15.7, -13.1] | +52.3 [+50.8, +53.9] | +48.7 [+46.8, +50.7] | -7.9 [-9.7, -6.1] | -4.8 [-6.0, -3.7] |
| fusion unspaced-xlmr rho=0.3 | +1.6 [+0.8, +2.3] | -4.3 [-5.4, -3.2] | -14.4 [-15.6, -13.1] | +51.6 [+50.0, +53.1] | +48.6 [+46.7, +50.5] | -7.9 [-9.7, -6.1] | -4.8 [-5.9, -3.8] |
| fusion xlmr-whole rho=0.3 | +1.0 [+0.2, +1.7] | -4.6 [-5.7, -3.6] | -17.1 [-18.5, -15.8] | +51.6 [+50.0, +53.1] | +48.7 [+46.8, +50.6] | -6.8 [-8.7, -5.0] | -5.1 [-6.2, -3.9] |
| fusion current rho=0.3 | +1.6 [+0.8, +2.3] | -4.3 [-5.3, -3.2] | -14.4 [-15.7, -13.1] | +51.7 [+50.1, +53.3] | +45.9 [+43.9, +47.8] | -7.8 [-9.7, -6.1] | -6.2 [-7.3, -5.1] |
| fusion unspaced-hangul-xlmr rho=0.5 guarded | -1.0 [-1.7, -0.2] | -5.4 [-6.5, -4.4] | -2.3 [-3.4, -1.1] | +44.6 [+43.1, +46.2] | +46.6 [+44.6, +48.5] | -6.8 [-8.6, -5.1] | -5.1 [-6.4, -3.9] |
| fusion unspaced-hangul-xlmr rho=0.5 | -1.0 [-1.8, -0.3] | -5.4 [-6.5, -4.4] | -2.3 [-3.4, -1.1] | +44.6 [+43.1, +46.2] | +46.6 [+44.7, +48.5] | -6.8 [-8.6, -5.1] | -5.1 [-6.3, -3.9] |
| qwen3 | +0.9 [+0.0, +1.7] | -6.7 [-7.9, -5.6] | -19.6 [-21.1, -18.2] | +52.6 [+51.1, +54.2] | +47.9 [+45.9, +49.8] | -9.0 [-11.1, -7.0] | -8.3 [-9.5, -7.1] |
| fusion unspaced-xlmr rho=0.5 guarded | -1.0 [-1.7, -0.2] | -5.4 [-6.5, -4.4] | -2.3 [-3.4, -1.1] | +44.6 [+43.1, +46.2] | +46.5 [+44.6, +48.5] | -12.1 [-14.0, -10.2] | -5.1 [-6.4, -3.9] |
| fusion unspaced-bigrams rho=0.5 guarded | -1.0 [-1.7, -0.2] | -5.4 [-6.5, -4.4] | -2.3 [-3.4, -1.1] | +44.8 [+43.2, +46.3] | +46.1 [+44.1, +48.0] | -12.1 [-14.0, -10.2] | -5.1 [-6.3, -3.9] |
| fusion unspaced-xlmr rho=0.5 | -1.0 [-1.8, -0.3] | -5.4 [-6.5, -4.4] | -2.3 [-3.4, -1.1] | +44.6 [+43.1, +46.2] | +46.5 [+44.6, +48.5] | -12.5 [-14.4, -10.7] | -5.1 [-6.4, -3.9] |
| fusion unspaced-bigrams rho=0.5 | -1.0 [-1.8, -0.3] | -5.4 [-6.5, -4.4] | -2.3 [-3.4, -1.1] | +44.8 [+43.2, +46.3] | +46.1 [+44.1, +48.0] | -12.5 [-14.4, -10.6] | -5.1 [-6.4, -4.0] |
| fusion current rho=0.5 guarded | -1.0 [-1.7, -0.2] | -5.4 [-6.5, -4.4] | -2.3 [-3.4, -1.1] | +50.7 [+49.0, +52.3] | +44.2 [+42.1, +46.4] | -12.1 [-14.0, -10.3] | -10.7 [-11.9, -9.5] |
| fusion xlmr-whole rho=0.5 | -2.5 [-3.3, -1.7] | -6.7 [-7.8, -5.6] | -15.8 [-17.2, -14.4] | +43.9 [+42.4, +45.5] | +46.4 [+44.5, +48.4] | -7.4 [-9.2, -5.5] | -5.3 [-6.5, -4.1] |
| fusion xlmr-whole rho=0.5 guarded | -2.5 [-3.3, -1.7] | -6.7 [-7.8, -5.6] | -15.8 [-17.2, -14.4] | +43.9 [+42.4, +45.5] | +46.4 [+44.5, +48.4] | -7.4 [-9.2, -5.5] | -5.3 [-6.6, -4.1] |
| rrf unspaced-hangul-xlmr | -3.5 [-4.3, -2.7] | -7.8 [-9.0, -6.7] | -5.9 [-7.2, -4.7] | +39.7 [+38.0, +41.3] | +43.8 [+41.8, +45.8] | -9.5 [-11.5, -7.5] | -7.5 [-8.9, -6.2] |
| rrf unspaced-xlmr | -3.5 [-4.3, -2.7] | -7.8 [-9.0, -6.7] | -5.9 [-7.2, -4.7] | +39.7 [+38.0, +41.3] | +43.8 [+41.8, +45.7] | -18.4 [-20.9, -16.1] | -7.5 [-8.9, -6.1] |
| rrf xlmr-whole | -5.4 [-6.3, -4.6] | -9.4 [-10.6, -8.3] | -16.7 [-18.1, -15.2] | +38.1 [+36.4, +39.7] | +44.0 [+42.0, +46.0] | -9.4 [-11.5, -7.4] | -7.2 [-8.5, -5.8] |
| rrf unspaced-bigrams | -3.5 [-4.4, -2.7] | -7.8 [-9.0, -6.7] | -5.9 [-7.2, -4.7] | +38.8 [+37.2, +40.5] | +43.5 [+41.4, +45.4] | -18.4 [-20.8, -16.1] | -7.5 [-8.9, -6.1] |
| fusion unspaced-hangul-xlmr rho=0.7 | -6.7 [-7.5, -5.8] | -9.3 [-10.5, -8.1] | -1.1 [-2.3, +0.1] | +33.5 [+31.9, +35.1] | +41.8 [+39.8, +43.8] | -12.5 [-14.5, -10.5] | -9.8 [-11.3, -8.4] |
| fusion unspaced-xlmr rho=0.7 | -6.7 [-7.5, -5.8] | -9.3 [-10.5, -8.1] | -1.1 [-2.3, +0.1] | +33.5 [+31.8, +35.1] | +41.8 [+39.9, +43.8] | -23.3 [-25.8, -20.9] | -9.8 [-11.3, -8.3] |
| fusion unspaced-bigrams rho=0.7 | -6.7 [-7.5, -5.8] | -9.3 [-10.5, -8.1] | -1.1 [-2.3, +0.1] | +30.8 [+29.2, +32.5] | +40.7 [+38.7, +42.7] | -23.3 [-25.9, -20.9] | -9.8 [-11.3, -8.4] |
| fusion xlmr-whole rho=0.7 | -9.6 [-10.5, -8.7] | -11.7 [-12.9, -10.5] | -19.1 [-20.6, -17.8] | +31.6 [+29.9, +33.2] | +41.4 [+39.5, +43.3] | -12.8 [-14.9, -10.9] | -9.4 [-10.9, -8.0] |
| zetokrag, rerank top 50 | -1.4 [-1.7, -1.2] | +0.3 [+0.1, +0.5] | -0.7 [-1.0, -0.4] | -0.0 [-0.2, +0.2] | +0.3 [+0.2, +0.6] | +0.7 [+0.3, +1.1] | +0.3 [-0.0, +0.7] |
| rrf current | -3.5 [-4.3, -2.7] | -7.8 [-9.0, -6.7] | -5.9 [-7.2, -4.7] | +28.6 [+27.4, +29.8] | +7.8 [+6.3, +9.3] | -18.5 [-20.8, -16.2] | -14.6 [-16.0, -13.2] |
| fusion current rho=0.5 | -1.0 [-1.8, -0.3] | -5.4 [-6.5, -4.4] | -2.3 [-3.4, -1.1] | -2.5 [-3.0, -2.1] | -5.9 [-6.6, -5.2] | -12.5 [-14.4, -10.7] | -10.7 [-11.9, -9.4] |
| fusion current rho=0.7 | -6.7 [-7.5, -5.9] | -9.2 [-10.5, -8.1] | -1.1 [-2.3, +0.1] | -4.4 [-5.1, -3.7] | -9.7 [-10.7, -8.8] | -23.3 [-25.8, -20.9] | -21.8 [-23.4, -20.3] |

## Per set (nDCG@10)

| set | setting | lang | queries | qwen3 | bge-m3-hybrid | fusion current rho=0.5 | fusion current rho=0.5 guarded | zetokrag (as used) | zetokrag, rerank top 50 | zetokrag with bm25 unspaced-xlmr |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| miracl-en | as-typed | en | 799 | 54.6 | 59.3 | 53.5 | 53.5 | 66.0 | 65.5 | 66.0 |
| scifact | as-typed | en | 300 | 70.1 | 68.0 | 73.5 | 73.5 | 74.4 | 74.3 | 74.4 |
| nfcorpus | as-typed | en | 323 | 35.4 | 33.2 | 36.0 | 36.3 | 35.2 | 34.8 | 35.2 |
| fiqa | as-typed | en | 648 | 46.8 | 42.1 | 40.9 | 40.9 | 44.7 | 44.2 | 44.7 |
| arguana | as-typed | en | 1000 | 67.3 | 51.4 | 60.7 | 60.7 | 52.4 | 45.8 | 52.4 |
| scidocs | as-typed | en | 1000 | 22.3 | 17.2 | 20.5 | 20.5 | 18.6 | 18.1 | 18.6 |
| zalo-legal-vn | as-typed | vi | 788 | 67.2 | 73.7 | 72.7 | 72.7 | 75.7 | 76.0 | 75.6 |
| zalo-legal-vn | no-diacritics | vi | 788 | 5.5 | 5.8 | 49.4 | 49.4 | 54.4 | 52.2 | 54.4 |
| nano-hotpotqa-vn | as-typed | vi | 1000 | 74.7 | 84.0 | 81.1 | 81.1 | 86.7 | 87.4 | 86.8 |
| nano-hotpotqa-vn | no-diacritics | vi | 1000 | 66.0 | 69.0 | 78.6 | 78.6 | 82.1 | 82.4 | 82.1 |
| nano-msmarco-vn | as-typed | vi | 1000 | 83.7 | 85.5 | 80.1 | 80.1 | 88.2 | 88.9 | 88.2 |
| nano-msmarco-vn | no-diacritics | vi | 1000 | 59.3 | 52.6 | 71.3 | 71.3 | 70.9 | 70.2 | 70.9 |
| fiqa-vn | as-typed | vi | 379 | 32.3 | 35.2 | 29.2 | 29.2 | 34.5 | 34.4 | 34.5 |
| fiqa-vn | no-diacritics | vi | 379 | 11.6 | 9.0 | 18.4 | 18.4 | 20.9 | 20.3 | 20.9 |
| scifact-vn | as-typed | vi | 134 | 66.7 | 63.0 | 68.1 | 68.1 | 73.3 | 73.1 | 73.3 |
| scifact-vn | no-diacritics | vi | 134 | 49.6 | 45.4 | 61.3 | 61.3 | 62.0 | 61.4 | 62.0 |
| miracl-zh | as-typed | zh | 393 | 60.9 | 64.2 | 1.1 | 60.4 | 2.3 | 2.5 | 65.8 |
| duretrieval-zh | as-typed | zh | 1000 | 83.7 | 84.9 | 6.2 | 82.5 | 7.7 | 7.7 | 88.5 |
| cmedqa-zh | as-typed | zh | 1000 | 37.3 | 32.4 | 9.1 | 33.1 | 14.1 | 13.7 | 36.7 |
| miracl-ja | as-typed | ja | 860 | 63.4 | 74.1 | 17.3 | 58.0 | 26.6 | 27.7 | 78.1 |
| jagovfaqs-ja | as-typed | ja | 1000 | 67.6 | 71.3 | 23.9 | 62.6 | 31.1 | 31.1 | 75.5 |
| nlp-journal-title-abs-ja | as-typed | ja | 510 | 95.4 | 96.8 | 24.0 | 94.9 | 25.1 | 25.1 | 97.8 |
| miracl-ko | as-typed | ko | 213 | 59.6 | 71.5 | 51.2 | 51.2 | 69.8 | 71.4 | 69.5 |
| ko-strategyqa | as-typed | ko | 592 | 76.8 | 79.8 | 66.3 | 67.4 | 80.2 | 80.8 | 80.2 |
| autorag-ko | as-typed | ko | 114 | 80.5 | 84.4 | 89.0 | 89.0 | 94.0 | 94.0 | 94.0 |
| miracl-hi | as-typed | hi | 350 | 51.7 | 63.5 | 46.9 | 46.9 | 66.6 | 67.0 | 69.3 |
| wikipedia-hi | as-typed | hi | 1000 | 82.7 | 88.0 | 81.4 | 81.4 | 90.5 | 91.1 | 91.4 |
| xpqa-hi | as-typed | hi | 925 | 74.2 | 76.3 | 73.2 | 73.2 | 76.3 | 76.4 | 76.8 |

## BM25 alone by language (MLDR included)

| view | en | vi | vi-no-diacritics | zh | ja | ko | hi | mean 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 current | 34.9 | 56.4 | 54.2 | 1.4 | 9.6 | 50.7 | 42.2 | 32.3 |
| bm25 unspaced-bigrams | 34.9 | 56.3 | 54.2 | 28.4 | 62.5 | 50.7 | 60.9 | 48.8 |
| bm25 unspaced-xlmr | 34.9 | 56.3 | 54.2 | 31.2 | 57.0 | 50.6 | 60.9 | 48.3 |
| bm25 unspaced-hangul-xlmr | 34.9 | 56.3 | 54.2 | 31.2 | 57.0 | 51.5 | 60.9 | 48.5 |
| bm25 xlmr-whole | 31.9 | 52.8 | 34.5 | 29.1 | 55.6 | 52.0 | 57.8 | 45.0 |

## MLDR (long documents), BM25 only

| set | current | unspaced-bigrams | unspaced-xlmr | unspaced-hangul-xlmr | xlmr-whole |
|---|---:|---:|---:|---:|---:|
| mldr-ja | 11.8 | 64.8 | 39.9 | 39.9 | 36.3 |
| mldr-ko | 54.8 | 54.7 | 54.3 | 24.1 | 26.3 |
| mldr-hi | 30.5 | 61.5 | 61.5 | 61.5 | 44.8 |

## Diagnostics

Queries for which BM25 finds no document at all (share), and how much of the relevant set ZetokRAG's fusion holds at each depth before reranking (recall).

| set | setting | empty BM25 current | empty BM25 unspaced-xlmr | recall@10 | @30 | @50 | @100 |
|---|---|---:|---:|---:|---:|---:|---:|
| miracl-en | as-typed | 0.0% | 0.0% | 65.2 | 84.3 | 89.9 | 96.3 |
| scifact | as-typed | 0.0% | 0.0% | 87.2 | 92.0 | 92.7 | 95.0 |
| nfcorpus | as-typed | 7.7% | 7.7% | 16.9 | 22.9 | 26.6 | 31.5 |
| fiqa | as-typed | 0.0% | 0.0% | 50.0 | 62.4 | 68.3 | 75.4 |
| arguana | as-typed | 0.0% | 0.0% | 89.6 | 98.3 | 98.9 | 99.7 |
| scidocs | as-typed | 0.0% | 0.0% | 21.5 | 33.5 | 38.7 | 46.3 |
| zalo-legal-vn | as-typed | 0.0% | 0.0% | 91.1 | 95.2 | 97.1 | 98.4 |
| zalo-legal-vn | no-diacritics | 0.0% | 0.0% | 69.8 | 83.2 | 86.0 | 90.2 |
| nano-hotpotqa-vn | as-typed | 0.0% | 0.0% | 82.3 | 86.0 | 87.4 | 89.6 |
| nano-hotpotqa-vn | no-diacritics | 0.0% | 0.0% | 79.0 | 83.0 | 84.8 | 87.0 |
| nano-msmarco-vn | as-typed | 0.0% | 0.0% | 91.8 | 95.1 | 96.7 | 98.1 |
| nano-msmarco-vn | no-diacritics | 0.0% | 0.0% | 82.4 | 87.0 | 88.9 | 91.3 |
| fiqa-vn | as-typed | 0.0% | 0.0% | 36.7 | 49.3 | 53.2 | 61.9 |
| fiqa-vn | no-diacritics | 0.0% | 0.0% | 23.1 | 32.2 | 35.9 | 40.7 |
| scifact-vn | as-typed | 0.0% | 0.0% | 81.9 | 90.3 | 92.8 | 94.4 |
| scifact-vn | no-diacritics | 0.0% | 0.0% | 74.8 | 86.4 | 86.9 | 87.9 |
| miracl-zh | as-typed | 96.9% | 0.0% | 2.0 | 2.4 | 2.8 | 3.0 |
| duretrieval-zh | as-typed | 91.6% | 0.0% | 6.8 | 7.8 | 8.0 | 8.3 |
| cmedqa-zh | as-typed | 59.7% | 0.0% | 14.8 | 21.3 | 23.9 | 28.6 |
| miracl-ja | as-typed | 67.3% | 0.0% | 23.1 | 28.7 | 30.8 | 32.1 |
| jagovfaqs-ja | as-typed | 60.3% | 0.0% | 32.7 | 35.9 | 36.3 | 37.0 |
| nlp-journal-title-abs-ja | as-typed | 88.6% | 0.4% | 24.9 | 25.1 | 25.1 | 25.3 |
| mldr-ja | as-typed | 65.0% | 0.0% |  |  |  |  |
| miracl-ko | as-typed | 0.5% | 0.5% | 63.8 | 80.0 | 85.7 | 90.8 |
| ko-strategyqa | as-typed | 1.7% | 1.7% | 79.4 | 85.2 | 87.3 | 89.1 |
| autorag-ko | as-typed | 0.0% | 0.0% | 98.2 | 100.0 | 100.0 | 100.0 |
| mldr-ko | as-typed | 0.0% | 0.0% |  |  |  |  |
| miracl-hi | as-typed | 0.0% | 0.0% | 63.2 | 80.0 | 84.3 | 92.0 |
| wikipedia-hi | as-typed | 0.0% | 0.0% | 92.1 | 96.1 | 97.4 | 98.0 |
| xpqa-hi | as-typed | 0.0% | 0.2% | 83.5 | 89.8 | 92.6 | 95.5 |
| mldr-hi | as-typed | 0.0% | 0.0% |  |  |  |  |

## Cross-check with gate 1

ZetokRAG as used, recomputed here, against gate 1's own numbers (same code, data and models; embeddings reused).

| set | setting | gate 1 | here | queries identical |
|---|---|---:|---:|---:|
| scifact | as-typed | 74.41 | 74.41 | 300/300 |
| nfcorpus | as-typed | 35.19 | 35.19 | 322/323 |
| fiqa | as-typed | 44.73 | 44.73 | 648/648 |
| arguana | as-typed | 52.49 | 52.36 | 972/1000 |
| scidocs | as-typed | 18.60 | 18.60 | 998/1000 |
| zalo-legal-vn | as-typed | 75.88 | 75.65 | 773/788 |
| zalo-legal-vn | no-diacritics | 54.39 | 54.41 | 777/788 |
| nano-hotpotqa-vn | as-typed | 86.74 | 86.74 | 1000/1000 |
| nano-hotpotqa-vn | no-diacritics | 82.06 | 82.06 | 1000/1000 |
| nano-msmarco-vn | as-typed | 88.16 | 88.16 | 1000/1000 |
| nano-msmarco-vn | no-diacritics | 70.92 | 70.91 | 999/1000 |
| fiqa-vn | as-typed | 34.47 | 34.47 | 379/379 |
| fiqa-vn | no-diacritics | 20.86 | 20.86 | 379/379 |
| scifact-vn | as-typed | 73.27 | 73.27 | 134/134 |
| scifact-vn | no-diacritics | 62.02 | 62.02 | 134/134 |
