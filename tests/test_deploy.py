"""The laptop test of the model server runs the Modal server's own flags, and those flags keep the cost controls."""
import ast
import importlib.util
import sys
from pathlib import Path

import pytest

DEPLOY = Path(__file__).parent.parent / "deploy"


def load_local():
    spec = importlib.util.spec_from_file_location("local_vllm", DEPLOY / "local_vllm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_local_server_uses_the_modal_flags_with_only_size_changed():
    local = load_local()
    flags, cmd = local.modal_flags(), local.command()
    for f in ("--reasoning-parser", "--tool-call-parser", "--enable-auto-tool-choice", "--language-model-only"):
        assert f in flags and f in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "qwen3_coder" == flags[flags.index("--tool-call-parser") + 1]
    assert cmd[cmd.index("--max-model-len") + 1] == "16384" and "--served-model-name" in cmd and "llm" in cmd


def test_the_modal_server_is_private_single_and_scales_to_zero():
    src = (DEPLOY / "modal_vllm.py").read_text()
    server = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef) and n.name == "Server")
    kw = {k.arg: k.value for k in server.decorator_list[0].keywords}
    assert "unauthenticated" not in kw                         # Modal servers require a proxy token by default
    assert ast.literal_eval(kw["max_containers"]) == 1 and ast.literal_eval(kw["min_containers"]) == 0
    assert "scaledown_window" in kw and "REVISION = \"" in src and "vllm==" in src


def load(name):
    spec = importlib.util.spec_from_file_location(name, DEPLOY / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_image_spec_avoids_the_problems_found_in_review():
    s = load("replica").spec(DEPLOY / "modal_vllm.py")
    assert s["BASE"].startswith("nvidia/cuda:13.0")                   # vLLM 0.29's torch is built for CUDA 13.0
    assert s["ENV"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"              # nothing compiled at start, as tested
    assert not any("hf_transfer" in p for p in s["PIP"]) and "HF_HUB_ENABLE_HF_TRANSFER" not in s["ENV"]
    assert "--reasoning-parser" in s["FLAGS"] and s["PORT"] == 8000
    text = load("replica").dockerfile(s)
    assert text.startswith(f"FROM {s['BASE']}") and "vllm==0.29.0" in text and "VLLM_USE_FLASHINFER_SAMPLER" in text


def test_the_laptop_server_gets_the_modal_environment():
    env = load("local_vllm").modal_constant("ENV")
    assert env == load("replica").spec(DEPLOY / "modal_vllm.py")["ENV"]


def test_the_modal_file_defines_its_app_when_modal_is_there():
    """Imports the file with Modal's own library (no network), which checks every decorator argument."""
    import shutil
    import subprocess
    python = None
    try:
        import modal  # noqa: F401
        python = sys.executable
    except ImportError:
        candidate = Path.home() / ".pixi/envs/modal-client/bin/python"
        python = str(candidate) if candidate.exists() else shutil.which("python-modal")
    if not python:
        pytest.skip("Modal's library is not installed here")
    r = subprocess.run([python, "-W", "ignore", "-c",
                        "import sys; sys.path.insert(0, 'deploy'); import modal_vllm as m; "
                        "print(sorted(m.app._functions) if hasattr(m.app, '_functions') else 'ok')"],
                       cwd=DEPLOY.parent, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]


def test_a_vllm_that_dies_at_start_fails_fast():
    import time
    server = load("vllm_server")
    t = time.time()
    with pytest.raises(RuntimeError, match="exited with code 3"):
        server.start([sys.executable, "-c", "print('CUDA error: no kernel image'); import sys; sys.exit(3)"], 18911, 60)
    assert time.time() - t < 10


def test_a_vllm_that_never_serves_is_stopped_at_its_deadline():
    server = load("vllm_server")
    with pytest.raises(RuntimeError, match="did not serve"):
        server.start([sys.executable, "-c", "import time; time.sleep(60)"], 18912, 3)


def test_the_server_is_returned_once_it_answers(tmp_path):
    server = load("vllm_server")
    fake = tmp_path / "fake.py"
    fake.write_text("import http.server as h\n"
                    "class H(h.BaseHTTPRequestHandler):\n"
                    "    def do_GET(self):\n"
                    "        self.send_response(200 if self.path == '/v1/models' else 404); self.end_headers()\n"
                    "h.HTTPServer(('127.0.0.1', 18913), H).serve_forever()\n")
    p = server.start([sys.executable, str(fake)], 18913, 30)
    assert p.poll() is None
    p.terminate()


FAKE_SERVER = """
import http.server, os, sys
open(sys.argv[2], "w").write(str(os.getpid()))
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/v1/models" else 404); self.end_headers()
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""
RUNNER = """
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("chat_local", {deploy!r} + "/chat_local.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.ROOT = Path({root!r}); m.URL = "http://127.0.0.1:{port}/v1"
m.SERVER = [sys.executable, {fake!r}, "{port}", {pidfile!r}]
m.CHAT = [sys.executable, "-c", "import time; time.sleep({chat_seconds})"]   # the TUI, open for a while
sys.exit(m.main())
"""


def alive(pid: int) -> bool:
    import os
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:                      # a zombie is dead too
        return f.read().split(") ")[1][0] != "Z"


@pytest.fixture
def chat_local(tmp_path):
    """Starts chat_local.py with a fake model server, in a terminal of its own (a pty that is its controlling
    terminal, as in a window); returns (start, pid of the fake server)."""
    import os
    import socket
    import subprocess
    import time

    pty = pytest.importorskip("pty")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    (tmp_path / "fake.py").write_text(FAKE_SERVER)
    started = []

    def start(name="run", chat_seconds=120):
        pidfile = tmp_path / f"{name}.server-pid"
        runner = tmp_path / f"{name}.py"
        runner.write_text(RUNNER.format(deploy=str(DEPLOY), root=str(tmp_path), port=port,
                                        fake=str(tmp_path / "fake.py"), pidfile=str(pidfile),
                                        chat_seconds=chat_seconds))
        master, slave = pty.openpty()
        # a new session whose controlling terminal is the pty: closing its master sends SIGHUP, as a window does
        proc = subprocess.Popen([sys.executable, "-c", "import fcntl, os, sys, termios; os.setsid(); "
                                 "fcntl.ioctl(0, termios.TIOCSCTTY, 0); os.execv(sys.executable, sys.argv[1:])",
                                 sys.executable, str(runner)], stdin=slave, stdout=slave, stderr=slave)
        os.close(slave)
        started.append((proc, master))
        return proc, master, pidfile
    start.tmp = tmp_path
    yield start, port
    for proc, master in started:
        proc.kill()
        try:
            os.close(master)
        except OSError:
            pass


def wait_for(check, seconds=20):
    import time
    end = time.time() + seconds
    while time.time() < end:
        if check():
            return True
        time.sleep(0.1)
    return False


def test_closing_the_terminal_stops_the_model_server_chat_local_started(chat_local):
    """Closing the window used to kill chat_local.py before its clean-up, leaving vLLM on the GPU (an orphan held
    the laptop's GPU for 3 h 53 min on 2026-09-29)."""
    import os

    start, _ = chat_local
    proc, master, pidfile = start()
    assert wait_for(pidfile.exists) and wait_for(lambda: pidfile.read_text().strip() != "")
    server = int(pidfile.read_text())
    assert wait_for(lambda: alive(server))
    import time
    time.sleep(1.5)                                              # chat_local has seen the server ready
    os.close(master)                                             # the window is closed
    assert wait_for(lambda: proc.poll() is not None, 10)
    assert wait_for(lambda: not alive(server), 10), "the model server outlived its terminal"


def test_a_server_left_by_a_killed_chat_local_is_taken_back_and_stopped(chat_local):
    """A chat-local killed outright (no clean-up at all) leaves its server; the next one adopts it and stops it on
    quit, instead of reusing it and leaving it running for good as before."""
    start, _ = chat_local
    first, _, pidfile = start("first")
    assert wait_for(lambda: pidfile.exists() and pidfile.read_text().strip() != "")
    server = int(pidfile.read_text())
    import time
    time.sleep(1.5)
    first.kill()                                                  # SIGKILL: no clean-up runs
    first.wait()
    assert alive(server)                                          # the orphan, as seen on the laptop
    second, _, _ = start("second", chat_seconds=1)
    assert wait_for(lambda: second.poll() is not None, 20)
    assert wait_for(lambda: not alive(server), 10), "the orphaned server was reused and left running"
    assert not (start.tmp / "runs" / "serve" / "local_vllm.pgid").exists()


def test_a_server_chat_local_did_not_start_is_left_running(chat_local):
    import subprocess
    start, port = chat_local
    own = start.tmp / "own.server-pid"
    mine = subprocess.Popen([sys.executable, str(start.tmp / "fake.py"), str(port), str(own)])
    try:
        assert wait_for(lambda: own.exists() and own.read_text().strip() != "")
        chat, _, _ = start("uses-it", chat_seconds=1)
        assert wait_for(lambda: chat.poll() is not None, 20)
        assert mine.poll() is None                                # someone else's server: untouched
    finally:
        mine.kill()


def test_quitting_stops_the_server_chat_local_started_at_once(chat_local):
    """The server is chat_local's own child: once stopped it is a zombie until reaped, and a wait that did not reap
    it ran its full 30 s before a pointless SIGKILL."""
    import time
    start, _ = chat_local
    proc, _, pidfile = start("quits", chat_seconds=2)
    assert wait_for(lambda: pidfile.exists() and pidfile.read_text().strip() != "")
    server = int(pidfile.read_text())
    assert wait_for(lambda: proc.poll() is not None, 20)
    t = time.time()
    assert wait_for(lambda: not alive(server), 5) and time.time() - t < 5
    assert not (start.tmp / "runs" / "serve" / "local_vllm.pgid").exists()
