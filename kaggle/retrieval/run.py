# peresearch retrieval benchmark on Kaggle (2 x T4): one worker per GPU, each over its shards of the corpora
# (eval/retrieval.py --shard k/4), then the speed measurement on this machine's corpora, alone on one GPU.
# Outputs (/kaggle/working): metrics.json per corpus and query setting, speed json, logs. Large intermediate
# files (embeddings, first stages, reranker scores) stay in /kaggle/temp and are not kept.
COMMIT = "main"
SHARDS = [0, 1]          # of 4; the other kernel runs [2, 3]
CORPORA = None           # e.g. ["scifact", "scifact-vn"]: run only these (smoke test), ignoring SHARDS
PINS = "sentence-transformers==6.1.0 transformers==5.17.0 bm25s==0.3.11 huggingface_hub==1.33.0"

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO, SRC = "https://github.com/geminitt/peresearch.git", Path("/tmp/peresearch")
WORK, TEMP = Path("/kaggle/working"), Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp/work")


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"pip install -q {PINS}")
if not SRC.exists():
    sh(f"git clone -q {REPO} {SRC}")
sh(f"cd {SRC} && git fetch -q origin && git checkout -q {COMMIT} && git log -1 --oneline")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
env = {**os.environ, "PYTHONPATH": f"{SRC}/src", "PERESEARCH_RUNS": str(TEMP / "runs"),
       "PERESEARCH_RESULTS": str(WORK / "results"), "TOKENIZERS_PARALLELISM": "false"}


def sync():
    """Copy the small result files to the kernel output, so they survive a session that runs out of time."""
    for p in (TEMP / "runs").glob("**/*"):
        if p.name == "metrics.json" or p.parent.name == "speed":
            dst = WORK / "runs" / p.relative_to(TEMP / "runs")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst)


def keep_syncing():
    while True:
        time.sleep(600)
        sync()


threading.Thread(target=keep_syncing, daemon=True).start()
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
sync()
with open(WORK / "speed.log", "w") as log:
    subprocess.run([sys.executable, f"{SRC}/eval/speed.py"], cwd=SRC, env={**env, "CUDA_VISIBLE_DEVICES": "0"},
                   stdout=log, stderr=subprocess.STDOUT)
sync()
sh(f"tail -n 5 {WORK}/worker0.log {WORK}/worker1.log {WORK}/speed.log")
