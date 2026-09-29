"""`pixi run chat-local`: the laptop model server, the TUI against it, and the server stopped when you quit.

The server is stopped however the TUI ends: quitting, closing the terminal window (SIGHUP) or a SIGTERM. A server
this script started is recorded (runs/serve/local_vllm.pgid); if a run was killed outright and left it behind, the
next run takes it back and stops it on quit. A server answering on localhost:8000 that this script did not start is
used as it is and left running. The server's log goes to runs/serve/local_vllm.log. The local models of ZetokRAG
move to the CPU by themselves while vLLM holds the GPU.
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
SERVER = ["pixi", "run", "-e", "serve", "python", "deploy/local_vllm.py"]
CHAT = ["peresearch", "chat"]


def ready() -> bool:
    try:
        with urllib.request.urlopen(URL + "/models", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def record() -> Path:
    return ROOT / "runs" / "serve" / "local_vllm.pgid"


def left_behind() -> int | None:
    """The process group of a server an earlier run started and never stopped, if it is still alive."""
    try:
        pgid = int(record().read_text())
        os.killpg(pgid, 0)
        return pgid
    except (OSError, ValueError):
        record().unlink(missing_ok=True)
        return None


def stop(pgid: int) -> None:
    """The whole group — pixi, the launcher and vLLM's workers — then forget it."""
    try:
        os.killpg(pgid, signal.SIGTERM)
        end = time.time() + 30
        while time.time() < end:
            try:
                while os.waitpid(-pgid, os.WNOHANG)[0]:   # reap our own dead children: a zombie still counts
                    pass
            except ChildProcessError:                  # not our children (a server an earlier run left)
                pass
            os.killpg(pgid, 0)
            time.sleep(0.2)
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    record().unlink(missing_ok=True)
    try:
        print("local model server stopped", flush=True)
    except OSError:                                    # the terminal is gone (the window was closed)
        pass


def main() -> int:
    # Closing the window sends SIGHUP, which would end this process before its `finally`; vLLM runs in a session
    # of its own and would go on holding the GPU (an orphan did for 3 h 53 min on 2026-09-29).
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, lambda number, frame: sys.exit(128 + number))
    ours = left_behind()
    if ours and not ready():                          # ours but not answering (hung or half-started): start afresh
        stop(ours)
        ours = None
    chat = None
    try:
        if ours:
            print("using the model server an earlier run left behind; it stops when you quit", flush=True)
        elif not ready():
            log = ROOT / "runs" / "serve" / "local_vllm.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"starting the local model server (log: {log.relative_to(ROOT)})", flush=True)
            server = subprocess.Popen(SERVER, cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                      start_new_session=True)
            ours = server.pid                         # its own session: the group id is its pid
            record().write_text(str(ours))
            start = time.time()
            while not ready():
                if server.poll() is not None:
                    print(f"the server stopped (exit {server.returncode}); last lines of the log:\n"
                          + "\n".join(log.read_text().splitlines()[-15:]), file=sys.stderr)
                    return 1
                print(f"\r  waiting for the model… {time.time() - start:.0f} s", end="", flush=True)
                time.sleep(2)
            print(f"\r  model ready after {time.time() - start:.0f} s", flush=True)
        chat = subprocess.Popen([*CHAT, *sys.argv[1:]], cwd=ROOT, env={**os.environ, "PERESEARCH_LLM_URL": URL})
        return chat.wait()
    finally:
        if chat is not None and chat.poll() is None:  # ended by a signal: the TUI goes too
            chat.terminate()
            try:
                chat.wait(5)
            except subprocess.TimeoutExpired:
                chat.kill()
        if ours:
            stop(ours)


if __name__ == "__main__":
    sys.exit(main())
