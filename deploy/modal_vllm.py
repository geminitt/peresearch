"""The agent's model on Modal: vLLM serving Qwen3.6-35B-A3B-FP8 on one L40S, behind Modal's proxy auth.

    modal run deploy/modal_vllm.py::download    # once: weights into a Volume, on a CPU container (no GPU billed)
    modal deploy deploy/modal_vllm.py           # the server; it costs nothing until a request wakes it
    modal app stop peresearch-llm               # take it down

Cost controls, all in this file: one container at most, scaled to zero after SCALEDOWN of silence, and no
public access — a request needs a Modal proxy token (`wk-…`/`ws-…`), which the agent sends as its API key.
The same flags are tested on the laptop with a small model of the same family first (deploy/local_vllm.py).

Client settings ($PERESEARCH_HOME/settings.env):
    PERESEARCH_LLM_URL=<the URL `modal deploy` prints>/v1
    PERESEARCH_LLM_KEY=wk-<id>.ws-<secret>        (a Modal proxy token: `modal workspace proxy-tokens create`)
"""

import modal

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"     # pinned: a moved revision is a different model
VLLM = "vllm==0.29.0"
GPU = "L40S"                 # 48 GB: 37.5 GB of FP8 weights, the rest for the KV cache
SCALEDOWN = 5 * 60           # seconds without requests before the container stops (and billing with it)
PORT = 8000

# The flags vLLM's recipe gives for Qwen3.5/3.6 (reasoning, XML tool calls), text only, 32k tokens per request
# (a research turn with its sources fits well within it), and no CUDA-graph capture for a faster cold start.
# Kept a plain literal: deploy/local_vllm.py reads it to run the same flags on the laptop.
FLAGS = ["--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
         "--language-model-only", "--max-model-len", "32768", "--gpu-memory-utilization", "0.95",
         "--enforce-eager"]

image = (modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
         .entrypoint([])
         .pip_install(VLLM, "huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_LOG_STATS_INTERVAL": "10"}))
weights = modal.Volume.from_name("peresearch-hf-cache", create_if_missing=True)
app = modal.App("peresearch-llm")


@app.function(image=image, volumes={"/root/.cache/huggingface": weights}, timeout=60 * 60)
def download():
    """Fetch the weights into the Volume on a CPU container, so no GPU time is spent downloading."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, revision=REVISION)
    weights.commit()


@app.server(image=image, gpu=GPU, port=PORT, volumes={"/root/.cache/huggingface": weights},
            scaledown_window=SCALEDOWN, startup_timeout=20 * 60, min_containers=0, max_containers=1,
            target_concurrency=8)
class Server:
    @modal.enter()
    def start(self):
        import subprocess

        cmd = ["vllm", "serve", MODEL, "--revision", REVISION, "--served-model-name", "llm",
               "--host", "0.0.0.0", "--port", str(PORT), *FLAGS]
        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
