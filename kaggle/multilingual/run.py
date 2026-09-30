# peresearch ranking-core pass on Kaggle (2 x T4): eval/multilingual.py `scores` on every non-MLDR set, one
# worker per GPU over its share of the sets (balanced by measured cost, largest first).
# Inputs: the BM25 views computed on the laptop (a private dataset of <set>__<setting>__lexical.npz) and the gate-1
# kernels, whose Qwen3 document embeddings of the 10 English and Vietnamese sets are reused after a check that the
# same code re-embeds a sample of each to the same vectors. Outputs of an earlier run of this kernel, attached as an
# input, are copied in first, so a rerun resumes. Data and models are downloaded once before the workers start,
# which then run offline; they are stopped after 11 hours, before Kaggle's 12-hour limit, with every finished set
# (and every 100 reranked queries) saved in /kaggle/working.
COMMIT = "main"
SETS = None              # e.g. [["autorag-ko"], ["xpqa-hi"]]: these sets per GPU (smoke test), else all, balanced
MAX_QUERIES = None       # e.g. 50 for the smoke test
PINS = "sentence-transformers==6.1.0 transformers==5.17.0 bm25s==0.3.11 huggingface_hub==1.33.0"

import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEADLINE = time.time() + 11 * 3600
REPO, SRC = "https://github.com/geminitt/peresearch.git", Path(os.environ.get("DRY_SRC", "/tmp/peresearch"))
# DRY_* run the same script on the laptop against a local checkout and folders (the rehearsal before Kaggle)
WORK, INPUT = Path(os.environ.get("DRY_WORK", "/kaggle/working")), Path(os.environ.get("DRY_INPUT", "/kaggle/input"))
RUNS = WORK / "runs" / "multilingual"


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


if "DRY_SRC" not in os.environ:
    sh(f"pip install -q {PINS}")
    if not SRC.exists():
        sh(f"git clone -q {REPO} {SRC}")
    sh(f"cd {SRC} && git fetch -q origin && git checkout -q {COMMIT} && git log -1 --oneline")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
# The image ships JAX, which bm25s imports; kept off the GPU, or it takes 75% of it (see zetokrag/core.py).
env = {**os.environ, "PYTHONPATH": f"{SRC}/src", "PERESEARCH_RUNS": str(WORK / "runs"), "PERESEARCH_COMMIT": COMMIT,
       "PERESEARCH_RESULTS": str(WORK / "results"), "TOKENIZERS_PARALLELISM": "false", "JAX_PLATFORMS": "cpu"}
if MAX_QUERIES:
    env["PERESEARCH_MAX_QUERIES"] = str(MAX_QUERIES)

RUNS.mkdir(parents=True, exist_ok=True)
# 1. an earlier run of this kernel (resume), then the laptop's BM25 views
for earlier in sorted(INPUT.glob("**/runs/multilingual")):
    sh(f"cp -rn {earlier}/. {RUNS}/")
for lex in sorted(INPUT.glob("**/*__lexical.npz")):      # the dataset is flat: <set>__<setting>__
    name, setting, _ = lex.name.split("__")
    (RUNS / name / setting).mkdir(parents=True, exist_ok=True)
    if not (RUNS / name / setting / "lexical.npz").exists():
        sh(f"cp {lex} {RUNS / name / setting}/lexical.npz")
# 2. gate 1's Qwen3 document embeddings, as candidates (checked below before they are used)
reused = []
for emb in sorted(INPUT.glob("**/runs/retrieval/*/docs-qwen3-embedding-0.6b.npy")):
    name = emb.parent.name
    dest = RUNS / name / "docs-qwen3-embedding-0.6b.npy"
    if (RUNS / name).exists() and not dest.exists():
        sh(f"cp {emb} {dest}")
        reused.append(name)
sh(f"find {RUNS} -name '*.npz' -o -name '*.npy' | sort | head -200")

sys.path[:0] = [f"{SRC}/eval", f"{SRC}/src"]
import multilingual  # noqa: E402

names = [n for n in multilingual.SETS if not n.startswith("mldr")]
groups = SETS or multilingual.shards(names, 2)
print("sets per GPU:", groups, flush=True)
flat = [n for g in groups for n in g]
missing = [n for n in flat for st in multilingual.settings(n) if not (RUNS / n / st / "lexical.npz").exists()]
assert not missing, f"BM25 views missing for {missing}"
subprocess.run([sys.executable, f"{SRC}/eval/multilingual.py", "prefetch", *flat], cwd=SRC, check=True,
               env={**env, "CUDA_VISIBLE_DEVICES": ""})
env["HF_HUB_OFFLINE"] = "1"
# 3. the reuse check on GPU 0: a sample of each reused set re-embedded here must match gate 1's vectors
subprocess.run([sys.executable, f"{SRC}/eval/multilingual.py", "reuse-check", *reused], cwd=SRC, check=True,
               env={**env, "CUDA_VISIBLE_DEVICES": "0"})
workers = []
for gpu, group in enumerate(groups):
    if group:
        log = open(WORK / f"worker{gpu}.log", "w")
        workers.append(subprocess.Popen([sys.executable, f"{SRC}/eval/multilingual.py", "scores", *group], cwd=SRC,
                                        env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=log,
                                        stderr=subprocess.STDOUT))
codes = []
for w in workers:
    try:
        codes.append(w.wait(timeout=max(1, DEADLINE - time.time())))
    except subprocess.TimeoutExpired:
        w.terminate()
        codes.append(f"stopped at the deadline ({w.wait()})")
print("workers exited with", codes, flush=True)
sh(f"pip freeze > {WORK}/pip-freeze.txt && du -sh {RUNS} && df -h {WORK}")
sh(f"find {RUNS} -name rerank.npz | sort | wc -l; find {RUNS} -name 'rerank.part.npz' | sort")
sh(f"grep -hE 'FAILED|Error|retry|preflight|reuse' {WORK}/worker*.log || true")
