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
