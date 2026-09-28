# ZetokRAG variants

17 corpora, 200 queries each (fixed seed). nDCG@10; the interval is a paired bootstrap over queries within each corpus, averaged over corpora.

## Queries as typed

| variant | en | vi |
|---|---|---|
| fold | 45.8 | 60.7 |
| plain | 45.8 | 60.7 |
| auto | 45.8 | 60.7 |
| qwen | 45.9 | 60.8 |
| qwen-auto | 45.9 | 60.8 |

Paired differences, as typed (points):

| comparison | en | vi |
|---|---|---|
| auto − fold | +0.0 [+0.0, +0.0] | +0.0 [-0.2, +0.2] |
| auto − plain | -0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.0] |
| qwen-auto − auto | +0.1 [-0.5, +0.7] | +0.0 [-0.4, +0.5] |
| qwen − plain | +0.0 [-0.6, +0.7] | +0.0 [-0.4, +0.5] |

## Queries no diacritics

| variant | vi |
|---|---|
| fold | 46.6 |
| plain | 37.5 |
| auto | 46.6 |
| qwen | 39.7 |
| qwen-auto | 47.5 |

Paired differences, no diacritics (points):

| comparison | vi |
|---|---|
| auto − fold | +0.0 [+0.0, +0.0] |
| auto − plain | +9.1 [+8.2, +10.0] |
| qwen-auto − auto | +0.9 [+0.4, +1.4] |
| qwen − plain | +2.2 [+1.5, +2.9] |

