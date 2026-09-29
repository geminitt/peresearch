"""The Modal server's image, rebuilt with Docker on this machine and run with a small model of the same family, so
that what would fail on Modal fails here first, for free.

    pixi run python deploy/replica.py                    # build, check, serve, one tool-call request, stop
    pixi run python deploy/replica.py --spec other.py    # the same checks for another spec file

The image (BASE, PYTHON, PIP, ENV), the flags (FLAGS) and the port are read from deploy/modal_vllm.py as literals;
only the model (Qwen3.5-0.8B: same chat template, reasoning and tool-call format) and the context and memory
sizes (a 6 GB laptop GPU) differ. Checks, each reported PASS or FAIL:
  environment  the CUDA the image's compiler targets matches the CUDA torch was built for
  download     a Hugging Face download works with the image's settings (what `download` does on Modal)
  serve        vLLM starts through deploy/vllm_server.py (the Modal start path) and answers within the timeout
  sampling     chat requests sent as the agent sends them (its temperature and top-p, streamed) are answered
  tool call    at least one of three such requests with a tool comes back as a tool call (a small model
               sampled at temperature 0.6 does not call the tool every time)
vLLM's whole log is kept in runs/replica/, with the lines about kernels compiled at run time summarized.
Under WSL, vLLM needs VLLM_WSL2_ENABLE_PIN_MEMORY=1; that is added here only, as Modal does not run WSL.
"""

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON_BUILDS = {"3.12": "https://github.com/astral-sh/python-build-standalone/releases/download/20260924/"
                         "cpython-3.12.14%2B20260924-x86_64-unknown-linux-gnu-install_only.tar.gz"}
SMALL = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
LAPTOP = {"--max-model-len": "16384", "--gpu-memory-utilization": "0.80"}
HOST_PORT = 18000


def spec(path: Path) -> dict:
    """The literal constants of a Modal spec file, without importing Modal."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    missing = {"BASE", "PYTHON", "PIP", "ENV", "FLAGS", "PORT"} - set(out)
    if missing:
        raise SystemExit(f"{path}: missing literal constants {sorted(missing)}")
    return out


def dockerfile(s: dict) -> str:
    env = "".join(f"ENV {k}={json.dumps(v)}\n" for k, v in s["ENV"].items())
    return (f"FROM {s['BASE']}\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates "
            "&& rm -rf /var/lib/apt/lists/*\n"
            f"RUN curl -fsSL {PYTHON_BUILDS[s['PYTHON']]} | tar -xz -C /opt\n"
            "ENV PATH=/opt/python/bin:$PATH\n"
            f"RUN python -m pip install --no-cache-dir {' '.join(json.dumps(p) for p in s['PIP'])}\n"
            f"{env}"
            "COPY vllm_server.py /root/vllm_server.py\n")


def run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def build(s: dict) -> str:
    text = dockerfile(s)
    tag = "peresearch-modal-replica:" + hashlib.sha256(text.encode()).hexdigest()[:12]
    if run(["docker", "image", "inspect", tag]).returncode == 0:
        return tag
    with tempfile.TemporaryDirectory() as ctx:
        (Path(ctx) / "Dockerfile").write_text(text)
        (Path(ctx) / "vllm_server.py").write_text((HERE / "vllm_server.py").read_text())
        print(f"building {tag} (first time only)…", flush=True)
        r = subprocess.run(["docker", "build", "-t", tag, ctx], text=True)
        if r.returncode:
            raise SystemExit("the image does not build: this spec would fail on Modal too")
    return tag


def in_image(tag: str, script: str, extra=()) -> subprocess.CompletedProcess:
    return run(["docker", "run", "--rm", "--gpus", "all", "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
                *extra, tag, "python", "-c", script])


ENV_CHECK = r"""
import json, os, re, subprocess, torch
nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
m = re.search(r"release (\d+\.\d+)", nvcc)
print(json.dumps({"torch_cuda": torch.version.cuda, "nvcc": m.group(1) if m else None,
                  "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))
"""
DOWNLOAD_CHECK = r"""
import os, tempfile, warnings
from huggingface_hub import hf_hub_download
p = hf_hub_download("%s", "config.json", revision="%s", cache_dir=tempfile.mkdtemp())
print("downloaded", os.path.getsize(p), "bytes")
"""


def tool_call(port: int) -> str:
    """One request as the agent sends it: its sampling, and streamed (llm.py streams every reply so that Esc can
    stop the model), so the check goes through vLLM's streaming tool-call parser."""
    body = {"model": "llm", "max_tokens": 512, "temperature": 0.6, "top_p": 0.95, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
        {"role": "user", "content": "List the files in the folder notes/. Use the tool."}],
        "tools": [{"type": "function", "function": {
            "name": "glob", "description": "List files matching a pattern",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}}],
        "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    calls = {}
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for choice in json.loads(line[6:]).get("choices") or []:
                for piece in choice.get("delta", {}).get("tool_calls") or []:
                    call = calls.setdefault(piece.get("index", 0), {"name": "", "arguments": ""})
                    call["name"] += (piece.get("function") or {}).get("name") or ""
                    call["arguments"] += (piece.get("function") or {}).get("arguments") or ""
    first = calls.get(min(calls)) if calls else None
    return f"{first['name']} {first['arguments']}" if first else ""


def main(args: list[str]) -> int:
    s = spec(Path(args[args.index("--spec") + 1]) if "--spec" in args else HERE / "modal_vllm.py")
    results = {}
    tag = build(s)

    r = in_image(tag, ENV_CHECK)
    info = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
    results["environment"] = (bool(info) and info["torch_cuda"] == info["nvcc"],
                              f"torch built for CUDA {info.get('torch_cuda')}, image compiler CUDA {info.get('nvcc')}"
                              if info else r.stderr[-500:])

    r = in_image(tag, DOWNLOAD_CHECK % SMALL)
    results["download"] = (r.returncode == 0, (r.stdout.strip() or r.stderr.strip()).splitlines()[-1][:300])

    flags = list(s["FLAGS"])
    for k, v in LAPTOP.items():
        flags[flags.index(k) + 1] = v
    serve = ["vllm", "serve", SMALL[0], "--revision", SMALL[1], "--served-model-name", "llm", "--host", "0.0.0.0",
             "--port", str(s["PORT"]), *flags]
    name = f"peresearch-replica-{int(time.time())}"
    start = time.time()
    proc = subprocess.Popen(["docker", "run", "--rm", "--name", name, "--gpus", "all", "-p", f"{HOST_PORT}:{s['PORT']}",
                             "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
                             "-e", "VLLM_WSL2_ENABLE_PIN_MEMORY=1", tag,
                             "python", "/root/vllm_server.py", str(s["PORT"]), "900", "--", *serve],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = []
    keep = HERE.parent / "runs" / "replica" / f"{name}.log"
    keep.parent.mkdir(parents=True, exist_ok=True)
    sink = open(keep, "w")

    def drain():                                  # the whole log, also after READY
        for line in proc.stdout:
            sink.write(line)
            sink.flush()
            log.append(line.rstrip())
    import threading
    threading.Thread(target=drain, daemon=True).start()
    try:
        while not any(x.startswith("READY") for x in log) and proc.poll() is None:
            time.sleep(1)
        ready = any(x.startswith("READY") for x in log)
        results["serve"] = (ready, f"serving after {time.time() - start:.0f} s" if ready else
                            "\n".join(x for x in log[-12:] if x.strip()))
        if ready:
            got, errors = [], []
            for _ in range(3):
                try:
                    got.append(tool_call(HOST_PORT))
                except Exception as e:
                    errors.append(f"{type(e).__name__}: {e}")
            results["sampling"] = (not errors, f"{len(got)}/3 requests answered" + (f"; {errors[0]}" if errors else ""))
            calls = [g for g in got if g]
            results["tool call"] = (bool(calls), f"{len(calls)}/{len(got)} replies were tool calls"
                                    + (f", e.g. {calls[0]}" if calls else ""))
    finally:
        run(["docker", "stop", "-t", "20", name])
        proc.wait()
    time.sleep(1)
    jit = [x for x in log if any(k in x.lower() for k in ("flashinfer", "nvcc", "jit", "ninja", "compil"))]
    print(f"\nvLLM log: {keep.relative_to(HERE.parent)}; lines about run-time compilation: {len(jit)}")
    for x in jit[:6]:
        print("   ", x[:200])
    print(f"\nreplica of {s['BASE']} + {' '.join(s['PIP'])}:")
    for check, (ok, detail) in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {check:12s} {detail}")
    return 0 if all(ok for ok, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
