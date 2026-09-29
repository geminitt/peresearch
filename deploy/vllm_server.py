"""Start vLLM and wait until it answers, or fail fast. Used by the Modal server and by its laptop replica.

A container whose vLLM crashes at start would otherwise sit, GPU billed, until Modal's startup timeout (20
minutes) runs out, and may be started again. `start` returns only once /v1/models answers; if the process exits
first, or the deadline passes, it stops the process and raises with the end of vLLM's log, so the container dies
within seconds of the failure. No import of Modal here: the replica runs it in a plain Docker container.
"""

import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque


def _ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def start(cmd: list[str], port: int, timeout: float = 20 * 60, env=None) -> subprocess.Popen:
    tail = deque(maxlen=40)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    def relay():                          # vLLM's log still reaches the container's output, and its end is kept
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            tail.append(line.rstrip())
    threading.Thread(target=relay, daemon=True).start()
    deadline = time.time() + timeout
    while not _ready(port):
        if proc.poll() is not None:
            time.sleep(0.5)               # let the relay catch the last lines
            raise RuntimeError(f"vLLM exited with code {proc.returncode} before serving:\n" + "\n".join(tail))
        if time.time() > deadline:
            proc.terminate()
            raise RuntimeError(f"vLLM did not serve within {timeout:.0f} s:\n" + "\n".join(tail))
        time.sleep(2)
    return proc


if __name__ == "__main__":                # python vllm_server.py <port> <timeout> -- vllm serve …   (the replica)
    port, timeout = int(sys.argv[1]), float(sys.argv[2])
    p = start(sys.argv[sys.argv.index("--") + 1:], port, timeout)
    print(f"READY on port {port}", flush=True)
    sys.exit(p.wait())
