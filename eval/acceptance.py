"""The 10 acceptance prompts of the agent (approved by the owner 2026-10-06), run through the real agent in order and
checkpointed per prompt: a rerun of the same label and model goes on where it stopped.

    pixi run python eval/acceptance.py run <label>       # the model in the settings, over the declared folders
    pixi run python eval/acceptance.py report <label>    # the table again, from the records

Each prompt is checked two ways:
- HARNESS: what peresearch guarantees whatever the model behind it — an answer comes back (no error, not
  interrupted), no web tool runs under /local, the per-question limits the user set hold, the follow-up is sent with
  its earlier turn. A failure here is a defect of peresearch. (Files before the web is the model's instruction since
  the `gaps` gate was removed, 2026-10-14: a rule below, not a guarantee.)
- RULES: the owner's pass rule for the prompt — the tools a good model picks, citations, language. With a small
  stand-in model they are recorded, not required (owner 2026-09-29: the small model checks mechanics only); the
  acceptance itself is the owner's reading of the full answers, which the records keep.

Records (one JSON line a prompt): the answer, every tool call that passed the budget with the first line of its result,
the calls the budget refused (repeats, limits), sources, the citation check, language, model calls, tokens, time,
empty answers asked again, and the requests that left the machine when the caller counts them.
"""

import collections
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("PERESEARCH_RUNS", ROOT / "runs")) / "acceptance"

# (id, prompt, how it is asked): "new" a fresh conversation, "after:<id>" in that prompt's conversation, "local" a fresh
# conversation with the web off (/local). The prompts are the owner's, verbatim (Vietnamese ones are test inputs).
PROMPTS = [
    ("greeting-vi", "Xin chào", "new"),
    ("greeting-en", "Hello, what can you do?", "new"),
    ("bpe-notes", "Tôi đã viết gì về BPE và SuperBPE trong các project của tôi?", "new"),
    ("where-follow-up", "Đoạn đó nằm ở file nào, dòng mấy?", "after:bpe-notes"),
    ("tokenization-files", "Which of my files cover tokenization?", "new"),
    ("superbpe-exact", "Chỗ nào trong ghi chú của tôi nhắc đúng cụm 'SuperBPE'?", "new"),
    ("nanovitok-folders", "Project nanovitok của tôi gồm những thư mục nào?", "new"),
    ("superbpe-paper", "SuperBPE là gì, ai viết bài báo và công bố khi nào?", "new"),
    ("vllm-latest", "Phiên bản mới nhất của vLLM là bao nhiêu?", "new"),
    ("vllm-latest-local", "Phiên bản mới nhất của vLLM là bao nhiêu?", "local"),
]
WEB = ("web_search", "search_papers", "search_wikipedia", "fetch")
FILES = ("search_notes", "grep", "glob", "read", "tree", "overview")
_VI = re.compile(r"[ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]", re.I)
_PLACE = re.compile(r"[\w./~-]+\.\w+\b.{0,80}?(?:\b(?:line|lines|dòng)\s*\d+|\bL\d+|:\d+)", re.I | re.S)


def language(text: str) -> str:
    """'vi' when a fifth of the words carry Vietnamese letters, 'en' when most are plain ASCII words, else 'other'.
    Citations, code and paths are left out first (they are English in any answer)."""
    text = re.sub(r"`[^`]*`|\[[NW]\d+(?:\s*,\s*[NW]\d+)*\]|\S*[/\\]\S*", " ", text)
    words = re.findall(r"[^\W\d_]+", text)
    if not words:
        return "other"
    if sum(bool(_VI.search(w)) for w in words) >= 0.2 * len(words):
        return "vi"
    return "en" if sum(w.isascii() for w in words) >= 0.8 * len(words) else "other"


def ran(record: dict) -> list[str]:
    """The tools that ran (reached the tools and were not refused), in order."""
    return [c["tool"] for c in record["calls"] if not c["result"].startswith("refused:")]


def harness(record: dict, limits: dict) -> list[str]:
    """What peresearch must guarantee for this prompt, whatever the model: each failure as a sentence."""
    bad = []
    if record.get("error"):
        bad.append(f"the question failed: {record['error']}")
        return bad
    if record["stopped"] == "cancelled" or not record["answer"].strip():
        bad.append(f"no answer came back (stopped {record['stopped']!r})")
    if record["mode"] == "local":
        if any(t in WEB for t in ran(record)):
            bad.append("a web tool ran under /local")
        if record.get("outside"):
            bad.append(f"{record['outside']} requests left the machine under /local")
    searches = sum(t in WEB[:3] for t in ran(record))
    if limits["web_searches"] is not None and searches > limits["web_searches"]:
        bad.append(f"{searches} web searches for a limit of {limits['web_searches']}")
    if limits["steps"] is not None and record["steps"] > limits["steps"] + 2:     # the limit's own call, one re-ask
        bad.append(f"{record['steps']} model calls for a limit of {limits['steps']}")
    if record["mode"].startswith("after:") and record["earlier_messages"] < 2:     # its question and answer
        bad.append("the follow-up was sent without its earlier turn")
    return bad


def rules(record: dict) -> dict[str, bool]:
    """The owner's pass rule for the prompt, as checks a script can make (the reading of the answer is the owner's)."""
    if record.get("error"):
        return {"answered": False}
    tools, done, text = [c["tool"] for c in record["calls"]], ran(record), record["answer"]
    cited = bool(re.search(r"\[[NW]\d+", text))
    web_ran = any(t in WEB for t in done)
    first_web = next((i for i, t in enumerate(tools) if t in WEB), None)
    before_web = tools if first_web is None else tools[:first_web]
    return {
        "greeting-vi": lambda: {"no tool": not tools, "in Vietnamese": record["language"] == "vi"},
        "greeting-en": lambda: {"no tool": not tools, "in English": record["language"] == "en"},
        "bpe-notes": lambda: {"search_notes ran": "search_notes" in done, "cites [N…]": "[N" in text,
                              "no web": not web_ran},
        "where-follow-up": lambda: {"names a file and a line": bool(_PLACE.search(text)),
                                    "no web": not web_ran},
        "tokenization-files": lambda: {"overview ran": "overview" in done, "in English": record["language"] == "en"},
        "superbpe-exact": lambda: {"grep ran": "grep" in done, "names a file and a line": bool(_PLACE.search(text))},
        "nanovitok-folders": lambda: {"tree ran": "tree" in done},
        "superbpe-paper": lambda: {"files first": any(t in FILES for t in before_web),
                                   "web or papers ran": web_ran, "cites": cited},
        "vllm-latest": lambda: {"files first": any(t in FILES for t in before_web),
                                "web ran": web_ran, "cites [W…]": "[W" in text},
        "vllm-latest-local": lambda: {"no web tool called": not any(t in WEB for t in tools),
                                      "no web cited": "[W" not in text},
    }[record["id"]]()


def checkpoint(label: str, config: dict) -> Path:
    sig = hashlib.sha256(json.dumps({"label": label, **config}, sort_keys=True).encode()).hexdigest()[:10]
    return RUNS / f"{label}-{sig}.jsonl"


def done(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
                out[rec["id"]] = rec
            except (ValueError, KeyError, TypeError):
                pass                                    # a line torn by an interruption
    return out


def append(path: Path, record: dict) -> None:
    """One record a line, on disk before the next prompt; a line torn by a power cut is ended first, so the record
    after it stays readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        torn = f.tell() > 0 and path.read_bytes()[-1:] != b"\n"
        f.write((b"\n" if torn else b"") + (json.dumps(record, ensure_ascii=False) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())


def run(label: str, make_agent, config: dict, outside=None, say=print) -> Path:
    """Ask the prompts in order with `make_agent(on_event)` (one agent, a fresh conversation per prompt unless the
    prompt follows another); `outside()` counts the requests that have left the machine so far, when given."""
    path = checkpoint(label, config)
    records = done(path)
    agent, events, reached = None, [], []
    by_id = {pid: (pid, prompt, mode) for pid, prompt, mode in PROMPTS}

    def ask(pid, prompt, mode):
        if mode.startswith("after:"):
            agent.resume(records[mode[6:]]["conversation"])
            if not agent.history():                     # its conversation is not in this home (another one): asked again
                ask(*by_id[mode[6:]])
                agent.resume(records[mode[6:]]["conversation"])
        else:
            agent.new_session()
        agent.toolbox.local_only = mode == "local"
        reached.clear()
        events.clear()
        before, earlier, t0 = outside() if outside else 0, len(agent.history()), time.time()
        rec = {"id": pid, "prompt": prompt, "mode": mode, "conversation": agent.conversation, "earlier_messages": earlier}
        try:
            a = agent.ask(prompt)
            refused = collections.Counter(n for n, _ in a.calls) - collections.Counter(c["tool"] for c in reached)
            rec.update(answer=a.text, stopped=a.stopped, steps=a.steps, tokens=a.prompt_tokens + a.completion_tokens,
                       calls=list(reached), loop_refused=dict(refused),
                       sources={k: s.where for k, s in a.sources.items()},
                       check={"unknown_ids": a.check.unknown_ids, "unsupported_quotes": len(a.check.unsupported_quotes),
                              "uncited": a.check.uncited},
                       asked_again=sum(k == "think" and d.startswith("empty answer") for k, d in events),
                       language=language(a.text))
        except Exception as e:                          # recorded and the run goes on: one failure never hides the rest
            rec.update(error=f"{type(e).__name__}: {str(e)[:300]}", calls=list(reached))
        finally:
            agent.toolbox.local_only = False            # /local is its prompt's only
        rec.update(seconds=round(time.time() - t0, 1), outside=(outside() - before) if outside else None)
        append(path, rec)
        records[pid] = rec
        say(f"{pid}: {'failed' if rec.get('error') else 'answered'} in {rec['seconds']} s; tools "
            f"{[c['tool'] for c in rec['calls']]}")

    for pid, prompt, mode in PROMPTS:
        if pid in records:
            continue
        if agent is None:
            agent = make_agent(lambda kind, detail: events.append((kind, detail)))
            made = agent._registry

            def registry():                                 # every call the budget let through, with its result
                r = made()
                real = r.run

                def run(name, args, budget=None, on_run=None):
                    out = real(name, args, budget, on_run)
                    if out.refused != "budget":
                        reached.append({"tool": name, "args": args, "result": out.text.strip().split("\n")[0][:200]})
                    return out
                r.run = run
                return r
            agent._registry = registry
        ask(pid, prompt, mode)
    return path


def report(path: Path, limits: dict) -> str:
    """One row a prompt: the harness's verdict (with its failures) and the owner's rules met."""
    records = done(path)
    rows = ["| prompt | harness | owner's rules | tools that ran | model calls | seconds |", "|---|---|---|---|---|---|"]
    for pid, _, _ in PROMPTS:
        rec = records.get(pid)
        if rec is None:
            rows.append(f"| {pid} | not run | | | | |")
            continue
        bad, met = harness(rec, limits), rules(rec)
        rows.append(f"| {pid} | {'ok' if not bad else 'FAILED: ' + '; '.join(bad)} | "
                    f"{sum(met.values())}/{len(met)}" + ("" if all(met.values()) else
                                                         " (missed: " + ", ".join(k for k, v in met.items() if not v) + ")")
                    + f" | {', '.join(ran(rec)) or '-'} | {rec.get('steps', '-')} | {rec['seconds']} |")
    return "\n".join(rows)


def main(args: list[str]) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from peresearch import cli, settings
    from peresearch.agent import Limits

    limits = Limits.from_settings()
    config = {"model": settings.get("PERESEARCH_LLM_MODEL"), "url": settings.get("PERESEARCH_LLM_URL"),
              "limits": [limits.steps, limits.web_searches, limits.fetches, limits.seconds]}
    path = checkpoint(args[1], config)
    if args[0] == "run":
        run(args[1], lambda on_event: cli.make_agent(f"acceptance-{args[1]}", on_event), config)
    table = report(path, {"steps": limits.steps, "web_searches": limits.web_searches})
    print(table)
    return 0 if "FAILED" not in table else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
