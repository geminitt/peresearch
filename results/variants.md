# ZetokRAG variants

17 corpora, 200 queries each (fixed seed). nDCG@10; the interval is a paired bootstrap over queries within each corpus, averaged over corpora.

## Queries as typed

| variant | en | vi |
|---|---|---|
| fold | 42.9 | 60.0 |
| plain | 42.9 | 60.0 |
| auto | 42.9 | 60.0 |
| qwen | 43.2 | 60.1 |
| qwen-auto | 43.2 | 60.1 |

Paired differences, as typed (points):

| comparison | en | vi |
|---|---|---|
| auto − fold | +0.0 [+0.0, +0.0] | +0.0 [-0.1, +0.2] |
| auto − plain | -0.0 [-0.0, +0.0] | +0.0 [+0.0, +0.0] |
| qwen-auto − auto | +0.2 [-0.3, +0.8] | +0.1 [-0.4, +0.5] |
| qwen − plain | +0.2 [-0.3, +0.8] | +0.1 [-0.4, +0.5] |

## Queries no diacritics

| variant | vi |
|---|---|
| fold | 46.4 |
| plain | 37.4 |
| auto | 46.4 |
| qwen | 39.6 |
| qwen-auto | 47.3 |

Paired differences, no diacritics (points):

| comparison | vi |
|---|---|
| auto − fold | +0.0 [+0.0, +0.0] |
| auto − plain | +9.0 [+8.1, +9.9] |
| qwen-auto − auto | +1.0 [+0.5, +1.4] |
| qwen − plain | +2.2 [+1.5, +2.9] |

