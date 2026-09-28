# peresearch retrieval benchmark on Kaggle (2 x T4): one worker per GPU, each over its shards of the corpora
# (eval/retrieval.py --shard k/4), then the speed measurement on this machine's corpora, alone on one GPU.
# Everything is written to /kaggle/working, the kernel's saved output: per corpus and query setting the metrics,
# a manifest (code, data and model revisions, packages, GPU), the first-stage candidates and every reranker
# score (to re-analyse without a GPU), plus the document embeddings (kept on Kaggle to reuse as a kernel input;
# not downloaded), the speed results and the logs.
COMMIT = "main"
SHARDS = [0, 1]          # of 4; the other kernel runs [2, 3]
CORPORA = None           # e.g. ["scifact", "scifact-vn"]: run only these (smoke test), ignoring SHARDS
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
env = {**os.environ, "PYTHONPATH": f"{SRC}/src", "PERESEARCH_RUNS": str(WORK / "runs"), "PERESEARCH_COMMIT": COMMIT,
       "PERESEARCH_RESULTS": str(WORK / "results"), "TOKENIZERS_PARALLELISM": "false",
       "PYTORCH_ALLOC_CONF": "expandable_segments:True"}


workers = []
if CORPORA:
    jobs = [CORPORA[i::2] for i in range(2)]
else:
    jobs = [["--shard", f"{k}/4"] for k in SHARDS]
for gpu, job in enumerate(jobs):
    log = open(WORK / f"worker{gpu}.log", "w")
    workers.append(subprocess.Popen([sys.executable, f"{SRC}/eval/retrieval.py", "run", *job], cwd=SRC,
                                    env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=log, stderr=subprocess.STDOUT))
codes = [w.wait() for w in workers]
print("workers exited with", codes, flush=True)
with open(WORK / "speed.log", "w") as log:
    subprocess.run([sys.executable, f"{SRC}/eval/speed.py"], cwd=SRC, env={**env, "CUDA_VISIBLE_DEVICES": "0"},
                   stdout=log, stderr=subprocess.STDOUT)
sh(f"pip freeze > {WORK}/pip-freeze.txt && du -sh {WORK}/runs")
sh(f"tail -n 5 {WORK}/worker0.log {WORK}/worker1.log {WORK}/speed.log")
