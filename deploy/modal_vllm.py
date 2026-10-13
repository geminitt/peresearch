"""The agent's model on Modal: vLLM serving Qwen3.6-35B-A3B-FP8 on one L40S, behind Modal's proxy auth.

    modal run deploy/modal_vllm.py::download    # once: weights into a Volume, on a CPU container (no GPU billed)
    modal deploy deploy/modal_vllm.py           # the server; it costs nothing until a request wakes it
    modal app stop peresearch-llm               # take it down

Cost controls: one container at most, hard CPU and memory limits, scaled to zero after SCALEDOWN of silence, no
public access (a request needs a Modal proxy token, which the agent sends as its API key), and a vLLM that fails
at start takes its container down within seconds (deploy/vllm_server.py).

Before any GPU time is paid for, the same image (BASE, PYTHON, PIP, ENV) is rebuilt with Docker on the laptop and
the same FLAGS are run with a small model of the same family: `pixi run -e serve python deploy/replica.py`. The
image spec and the flags are plain literals so that the replica reads them from this file.

Client settings ($PERESEARCH_HOME/settings.env):
    PERESEARCH_LLM_URL=<the URL `modal deploy` prints>/v1
    PERESEARCH_LLM_KEY=wk-<id>.ws-<secret>        (a Modal proxy token: `modal workspace proxy-tokens create`)
"""

from pathlib import Path

import modal

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"     # pinned: a moved revision is a different model
GPU = "L40S"                 # 48 GB: 37.5 GB of FP8 weights, the rest for the KV cache
SCALEDOWN = 5 * 60           # seconds without requests before the container stops (and billing with it)
STARTUP = 20 * 60            # seconds vLLM may take to load the weights from the Volume and serve
PORT = 8000
# CPU cores and memory (MiB) as (request, limit). Modal bills the higher of request and use, so the request stays at
# Modal's minimum and the limit is a hard cost ceiling. The server's limits are wide: a starved CPU slows the load
# while the GPU bills ~41x a core, and 64 GiB holds all 37.5 GB of weights even if the page cache is counted.
SERVER_CPU = (0.125, 8.0)
SERVER_MEMORY = (128, 65536)     # 64 GiB
DOWNLOAD_CPU = (0.125, 4.0)
DOWNLOAD_MEMORY = (128, 16384)   # 16 GiB

# The image. vLLM 0.29 brings torch 2.13 built for CUDA 13.0, so the base image carries the same CUDA (its
# compiler is what just-in-time kernels are built with). Weights come through Hugging Face's Xet transfer.
BASE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"
PYTHON = "3.12"
PIP = ["vllm==0.29.0"]
ENV = {"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_LOG_STATS_INTERVAL": "10",
       "VLLM_USE_FLASHINFER_SAMPLER": "0"}   # PyTorch's sampler, as tested: no kernel compiled at start

# The flags vLLM's recipe gives for Qwen3.5/3.6 (reasoning, XML tool calls), text only, 32k tokens per request
# (a research turn with its sources fits well within it). CUDA graphs stay on: with --enforce-eager the 35B decoded
# at ~10 tokens/s on Modal, and the 0.8B on the laptop at 33 against 101 tokens/s with graphs (2026-10-06).
FLAGS = ["--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
         "--language-model-only", "--max-model-len", "32768", "--gpu-memory-utilization", "0.95"]

# The server process's own environment, not the image's ENV (the download function and the laptop server share
# that). Graphs are compiled at start (0.8B on the laptop: engine start 103 s with empty caches, 24 s with filled
# ones); a fresh container has no cache, so vLLM's (torch.compile) and Triton's caches live on the weights Volume,
# which Modal commits in the background and at shutdown. The weights are on the Volume too, so the server never
# calls the Hugging Face Hub: vLLM does at two points of its start, without a timeout, and a stalled Hub (6 and 15
# minutes in the laptop replica, 2026-10-06) would bill the GPU until the start deadline.
SERVER_ENV = {"VLLM_CACHE_ROOT": "/root/.cache/huggingface/compile-cache/vllm",
              "TRITON_CACHE_DIR": "/root/.cache/huggingface/compile-cache/triton",
              "HF_HUB_OFFLINE": "1"}

image = (modal.Image.from_registry(BASE, add_python=PYTHON).entrypoint([]).pip_install(*PIP).env(ENV)
         .add_local_file(Path(__file__).with_name("vllm_server.py"), "/root/vllm_server.py"))
weights = modal.Volume.from_name("peresearch-hf-cache", create_if_missing=True)
app = modal.App("peresearch-llm")


@app.function(image=image, volumes={"/root/.cache/huggingface": weights}, timeout=60 * 60, cpu=DOWNLOAD_CPU,
              memory=DOWNLOAD_MEMORY)
def download():
    """Fetch the weights into the Volume on a CPU container, so no GPU time is spent downloading."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, revision=REVISION)
    weights.commit()


@app.server(image=image, gpu=GPU, port=PORT, volumes={"/root/.cache/huggingface": weights},
            scaledown_window=SCALEDOWN, startup_timeout=STARTUP, min_containers=0, max_containers=1,
            target_concurrency=8, cpu=SERVER_CPU, memory=SERVER_MEMORY)
class Server:
    @modal.enter()
    def start(self):
        import os
        import sys

        sys.path.insert(0, "/root")
        import vllm_server

        cmd = ["vllm", "serve", MODEL, "--revision", REVISION, "--served-model-name", "llm",
               "--host", "0.0.0.0", "--port", str(PORT), *FLAGS]
        print(*cmd, flush=True)
        self.process = vllm_server.start(cmd, PORT, timeout=STARTUP - 60,   # fail before Modal's own timeout
                                         env={**os.environ, **SERVER_ENV})

    @modal.exit()
    def stop(self):
        self.process.terminate()
