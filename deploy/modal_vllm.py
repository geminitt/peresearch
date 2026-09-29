"""The agent's model on Modal: vLLM serving Qwen3.6-35B-A3B-FP8 on one L40S, behind Modal's proxy auth.

    modal run deploy/modal_vllm.py::download    # once: weights into a Volume, on a CPU container (no GPU billed)
    modal deploy deploy/modal_vllm.py           # the server; it costs nothing until a request wakes it
    modal app stop peresearch-llm               # take it down

Cost controls: one container at most, scaled to zero after SCALEDOWN of silence, no public access (a request needs
a Modal proxy token, which the agent sends as its API key), a vLLM that fails at start takes its container down
within seconds (deploy/vllm_server.py), and the agent keeps a daily spending cap (peresearch/llm.py).

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

# The image. vLLM 0.29 brings torch 2.13 built for CUDA 13.0, so the base image carries the same CUDA (its
# compiler is what just-in-time kernels are built with). Weights come through Hugging Face's Xet transfer.
BASE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"
PYTHON = "3.12"
PIP = ["vllm==0.29.0"]
ENV = {"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_LOG_STATS_INTERVAL": "10",
       "VLLM_USE_FLASHINFER_SAMPLER": "0"}   # PyTorch's sampler, as tested: no kernel compiled at start

# The flags vLLM's recipe gives for Qwen3.5/3.6 (reasoning, XML tool calls), text only, 32k tokens per request
# (a research turn with its sources fits well within it), and no CUDA-graph capture for a faster cold start.
FLAGS = ["--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
         "--language-model-only", "--max-model-len", "32768", "--gpu-memory-utilization", "0.95",
         "--enforce-eager"]

image = (modal.Image.from_registry(BASE, add_python=PYTHON).entrypoint([]).pip_install(*PIP).env(ENV)
         .add_local_file(Path(__file__).with_name("vllm_server.py"), "/root/vllm_server.py"))
weights = modal.Volume.from_name("peresearch-hf-cache", create_if_missing=True)
app = modal.App("peresearch-llm")


@app.function(image=image, volumes={"/root/.cache/huggingface": weights}, timeout=60 * 60)
def download():
    """Fetch the weights into the Volume on a CPU container, so no GPU time is spent downloading."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, revision=REVISION)
    weights.commit()


@app.server(image=image, gpu=GPU, port=PORT, volumes={"/root/.cache/huggingface": weights},
            scaledown_window=SCALEDOWN, startup_timeout=STARTUP, min_containers=0, max_containers=1,
            target_concurrency=8)
class Server:
    @modal.enter()
    def start(self):
        import sys

        sys.path.insert(0, "/root")
        import vllm_server

        cmd = ["vllm", "serve", MODEL, "--revision", REVISION, "--served-model-name", "llm",
               "--host", "0.0.0.0", "--port", str(PORT), *FLAGS]
        print(*cmd, flush=True)
        self.process = vllm_server.start(cmd, PORT, timeout=STARTUP - 60)   # fail before Modal's own timeout

    @modal.exit()
    def stop(self):
        self.process.terminate()
