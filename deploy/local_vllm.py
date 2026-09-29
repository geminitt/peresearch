"""The Modal server's vLLM flags, run on this machine with a small model of the same family (Qwen3.5-0.8B: same
chat template, reasoning and tool-call format), so the serving setup and the agent are tested before any GPU
time is paid for.

    pixi run -e serve python deploy/local_vllm.py         # serves http://localhost:8000/v1 as model "llm"
    PERESEARCH_LLM_URL=http://localhost:8000/v1 pixi run peresearch ask "…"

Only the model, the context length and the memory share differ from deploy/modal_vllm.py (a 6 GB laptop GPU).
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

MODEL, REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
LOCAL = {"--max-model-len": "16384", "--gpu-memory-utilization": "0.80"}


def modal_constant(name: str):
    tree = ast.parse((Path(__file__).parent / "modal_vllm.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return ast.literal_eval(node.value)
    raise SystemExit(f"{name} not found in deploy/modal_vllm.py")


def modal_flags() -> list[str]:
    return modal_constant("FLAGS")


def command() -> list[str]:
    flags = modal_flags()
    for k, v in LOCAL.items():
        flags[flags.index(k) + 1] = v
    return ["vllm", "serve", MODEL, "--revision", REVISION, "--served-model-name", "llm", "--port", "8000", *flags]


if __name__ == "__main__":
    cmd = command()
    print(*cmd, flush=True)
    # The Modal image's environment, plus one laptop-only switch: vLLM turns pinned memory off under WSL2 unless
    # asked, and its GPU model runner needs it (UVA buffers).
    env = {**os.environ, **modal_constant("ENV"), "VLLM_WSL2_ENABLE_PIN_MEMORY": "1"}
    sys.exit(subprocess.call(cmd, env=env))
