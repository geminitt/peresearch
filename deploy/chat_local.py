"""`pixi run chat-local`: the laptop model server, the TUI against it, and the server stopped when you quit.

A server already answering on localhost:8000 is used as it is and left running. The server's log goes to
runs/serve/local_vllm.log. The local models of ZetokRAG move to the CPU by themselves while vLLM holds the GPU.
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

URL = "http://localhost:8000/v1"
ROOT = Path(__file__).resolve().parent.parent


def ready() -> bool:
    try:
        with urllib.request.urlopen(URL + "/models", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def main() -> int:
    server = None
    if not ready():
        log = ROOT / "runs" / "serve" / "local_vllm.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"starting the local model server (log: {log.relative_to(ROOT)})", flush=True)
        server = subprocess.Popen(["pixi", "run", "-e", "serve", "python", "deploy/local_vllm.py"], cwd=ROOT,
                                  stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        start = time.time()
        while not ready():
            if server.poll() is not None:
                print(f"the server stopped (exit {server.returncode}); last lines of the log:\n"
                      + "\n".join(log.read_text().splitlines()[-15:]), file=sys.stderr)
                return 1
            print(f"\r  waiting for the model… {time.time() - start:.0f} s", end="", flush=True)
            time.sleep(2)
        print(f"\r  model ready after {time.time() - start:.0f} s", flush=True)
    try:
        return subprocess.call(["peresearch", "chat", *sys.argv[1:]], cwd=ROOT,
                               env={**os.environ, "PERESEARCH_LLM_URL": URL})
    finally:
        if server is not None:
            os.killpg(server.pid, signal.SIGTERM)        # the whole group: pixi, the launcher and vLLM's workers
            try:
                server.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
            print("local model server stopped", flush=True)


if __name__ == "__main__":
    sys.exit(main())
