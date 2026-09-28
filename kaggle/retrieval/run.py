# peresearch retrieval benchmark on Kaggle (2 x T4): one worker per GPU, each over its shards of the corpora
# (eval/retrieval.py --shard k/4), then the speed measurement on the same corpora, alone on one GPU.
# Everything is written to /kaggle/working, the kernel's saved output: per corpus and query setting the metrics,
# a manifest (code, data and model revisions, packages, GPU), the first-stage candidates and every reranker
# score (to re-analyse without a GPU), plus the document embeddings (kept on Kaggle to reuse as a kernel input;
# not downloaded), the speed results and the logs. Outputs of earlier kernels attached as inputs are copied in
# first, so finished corpora are skipped and saved embeddings are reused.
COMMIT = "main"
SHARDS = [0, 1]          # of 4; the other kernel runs [2, 3]
CORPORA = None           # e.g. [["nfcorpus-vn", "scifact-vn"], []]: these corpora per GPU (smoke test), not SHARDS
PINS = "sentence-transformers==6.1.0 transformers==5.17.0 bm25s==0.3.11 huggingface_hub==1.33.0"

import os
import subprocess
import sys
from pathlib import Path

REPO, SRC = "https://github.com/geminitt/peresearch.git", Path("/tmp/peresearch")
WORK = Path("/kaggle/working")


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"pip install -q {PINS}")
if not SRC.exists():
    sh(f"git clone -q {REPO} {SRC}")
sh(f"cd {SRC} && git fetch -q origin && git checkout -q {COMMIT} && git log -1 --oneline")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
# The image ships JAX, which bm25s imports; kept off the GPU, or it takes 75% of it (see zetokrag/core.py).
env = {**os.environ, "PYTHONPATH": f"{SRC}/src", "PERESEARCH_RUNS": str(WORK / "runs"), "PERESEARCH_COMMIT": COMMIT,
       "PERESEARCH_RESULTS": str(WORK / "results"), "TOKENIZERS_PARALLELISM": "false", "JAX_PLATFORMS": "cpu"}

(WORK / "runs" / "retrieval").mkdir(parents=True, exist_ok=True)
for earlier in sorted(Path("/kaggle/input").glob("**/runs/retrieval")):
    sh(f"cp -rn {earlier}/. {WORK}/runs/retrieval/")
sh(f"find {WORK}/runs/retrieval -name 'metrics.json' -o -name 'docs-*.npy' | sort")

sys.path[:0] = [f"{SRC}/eval", f"{SRC}/src"]
from retrieval import shard  # noqa: E402

groups = CORPORA or [shard(k, 4) for k in SHARDS]
print("corpora per GPU:", groups, flush=True)
workers = []
for gpu, names in enumerate(groups):
    if not names:
        continue
    log = open(WORK / f"worker{gpu}.log", "w")
    workers.append(subprocess.Popen([sys.executable, f"{SRC}/eval/retrieval.py", "run", *names], cwd=SRC,
                                    env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=log, stderr=subprocess.STDOUT))
codes = [w.wait() for w in workers]
print("workers exited with", codes, flush=True)
with open(WORK / "speed.log", "w") as log:
    subprocess.run([sys.executable, f"{SRC}/eval/speed.py", *[n for g in groups for n in g]], cwd=SRC,
                   env={**env, "CUDA_VISIBLE_DEVICES": "0"}, stdout=log, stderr=subprocess.STDOUT)
sh(f"pip freeze > {WORK}/pip-freeze.txt && du -sh {WORK}/runs")
sh(f"tail -n 5 {WORK}/worker*.log {WORK}/speed.log")
