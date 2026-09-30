"""Pin the commit and push a kernel with the kaggle CLI.

    python kaggle/push.py retrieval --job full-shards-0-1 --set 'SHARDS=[0, 1]' --commit <sha> --input <earlier kernel>

The kernel clones this repository at the pinned commit (it must be on GitHub), so the code that runs is exactly
a commit of the repository.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("kernel")
ap.add_argument("--commit", default=None, help="defaults to HEAD, which must be pushed")
ap.add_argument("--set", nargs="*", default=[], help="NAME=VALUE overrides of top-level constants")
ap.add_argument("--job", default=None, help="suffix of a separate kernel, so jobs run in parallel")
ap.add_argument("--input", nargs="*", default=[], help="earlier kernels whose outputs this one reads")
ap.add_argument("--dataset", nargs="*", default=[], help="datasets this kernel reads (owner/slug)")
args = ap.parse_args()

commit = args.commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
remote = subprocess.run(["git", "branch", "-r", "--contains", commit], capture_output=True, text=True).stdout
assert remote.strip(), f"commit {commit} is not on the remote yet: push it first"

src = (HERE / args.kernel / "run.py").read_text()
src = re.sub(r'^COMMIT = .*$', f'COMMIT = "{commit}"', src, flags=re.M)
for kv in args.set:
    k, v = kv.split("=", 1)
    src, n = re.subn(rf'^{k} = .*$', f"{k} = {v}", src, flags=re.M)
    assert n == 1, f"no top-level constant {k}"
meta = json.loads((HERE / args.kernel / "kernel-metadata.json").read_text())
meta["kernel_sources"] = args.input
meta["dataset_sources"] = args.dataset
build = HERE / args.kernel / "build"
if args.job:
    meta["id"] += f"-{args.job}"
    meta["title"] += f" {args.job}"
    build = HERE / args.kernel / f"build-{args.job}"
build.mkdir(exist_ok=True)
(build / "run.py").write_text(src)
(build / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
subprocess.run(["kaggle", "kernels", "push", "-p", str(build)], check=True)
