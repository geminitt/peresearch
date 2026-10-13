"""The sandbox every test runs in, and small stand-ins for the GPU models, so every test runs on CPU (as in CI).

Sandbox: tests (and mutation runs, which execute deliberately broken code) once wrote into the owner's real data
folder — a broken home path fell back to ~/.local/share/peresearch and replaced the real index with a test one.
Every test therefore runs with HOME, PERESEARCH_HOME and the eval output folders inside a temporary folder, set
here before anything is imported, and the session fails if the owner's real places changed while it ran."""
import hashlib
import os
import sys
import tempfile
import faulthandler
import threading
from pathlib import Path

from hypothesis import settings

# Properties check behavior, not speed: Hypothesis's 200 ms per example failed a layout test on a slower container
# (321 ms for a first example that takes 6-13 ms here, 2026-10-03), and would on any slow machine. Speed is checked
# by its own tests where it matters (the denylist, the walk).
settings.register_profile("peresearch", deadline=None)
settings.load_profile("peresearch")

REAL_HOME = Path(os.path.expanduser("~"))
REAL_PLACES = [Path(os.environ.get("PERESEARCH_HOME", REAL_HOME / ".local/share/peresearch")),
               Path(__file__).resolve().parent.parent / "results"]


def _state(places):
    """Every file under the places, with its size and modification time."""
    out = {}
    for place in places:
        for dirpath, _, names in os.walk(place):
            for name in names:
                p = Path(dirpath) / name
                try:
                    st = p.stat()
                except OSError:
                    continue
                out[str(p)] = (st.st_size, st.st_mtime_ns)
    return out


BEFORE = _state(REAL_PLACES)

SANDBOX = Path(tempfile.mkdtemp(prefix="peresearch-tests-"))
for name in list(os.environ):
    if name.startswith("PERESEARCH_") or name in ("JINA_API_KEY", "OPENALEX_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"):
        os.environ.pop(name)
os.environ.setdefault("HF_HOME", str(REAL_HOME / ".cache" / "huggingface"))   # models stay where they were fetched
for name, sub in (("HOME", "user"), ("PERESEARCH_HOME", "data"), ("PERESEARCH_RUNS", "runs"),
                  ("PERESEARCH_RESULTS", "results")):
    (SANDBOX / sub).mkdir()
    os.environ[name] = str(SANDBOX / sub)
os.environ.pop("PERESEARCH_RERANK_DEPTH", None)    # the owner's shell setting would change what searches return

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from peresearch.zetokrag.core import tokenize


def pytest_addoption(parser):
    parser.addoption("--test-timeout", type=float, default=300, help="Hard deadline per test in seconds (default: 300)")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    seconds = item.config.getoption("--test-timeout")
    if seconds <= 0:
        raise ValueError("--test-timeout must be positive")
    def expired():
        capture = item.config.pluginmanager.getplugin("capturemanager")
        if capture is not None:
            capture.suspend_global_capture(in_=True)
        print(f"\nTEST TIMEOUT: {item.nodeid} exceeded {seconds} seconds", file=sys.stderr, flush=True)
        faulthandler.dump_traceback(file=sys.stderr)
        os._exit(124)
    timer = threading.Timer(seconds, expired)
    timer.daemon = True
    timer.start()
    try:
        return (yield)
    finally:
        timer.cancel()


class FakeEmbedder:
    """Hashed bag of (diacritic-folded) words: similar wording gives similar vectors."""

    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in tokenize(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).astype(np.float16)

    def documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float16)

    def queries(self, texts):                  # the same vectors; not a document (a spy on documents counts none)
        return FakeEmbedder.documents(self, texts)


class FakeReranker:
    """Share of query words found in the text, in [0, 1]."""

    def scores(self, query, texts):
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(t))) / max(len(q), 1) for t in texts], dtype=np.float32)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("PERESEARCH_HOME", str(h))
    return h


def pytest_sessionfinish(session, exitstatus):
    after = _state(REAL_PLACES)
    changed = sorted(p for p in set(BEFORE) | set(after) if BEFORE.get(p) != after.get(p))
    if changed:
        print("\nTESTS CHANGED THE OWNER'S REAL FILES:\n  " + "\n  ".join(changed))
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
