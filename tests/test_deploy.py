"""The laptop test of the model server runs the Modal server's own flags, and those flags keep the cost controls."""
import ast
import importlib.util
from pathlib import Path

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
