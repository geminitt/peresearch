"""Every assertion a change to the tests removed or rewrote, for a reviewer to read: a test made green by weakening
it shows here, whoever made the change.

    pixi run python scripts/assert_diff.py [BASE]      # BASE: a commit, default HEAD

Prints, for each test file, the assert lines that left (-) and the ones that came (+) in each hunk, and how many
assert statements the file held before and after. Exit status 1 when any assert line left, so a script can stop on
it; a reviewer then checks that each one kept its meaning (only how its inputs are built may change)."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSERT = re.compile(r"^\s*assert\b|pytest\.raises|\bassert ")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def count(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s*assert\b", line))


def main(base: str = "HEAD") -> int:
    files = [f for f in git("diff", "--name-only", base, "--", "tests").split() if f.endswith(".py")]
    files += [f for f in git("ls-files", "--others", "--exclude-standard", "tests").split() if f.endswith(".py")]
    removed_any = False
    for name in sorted(set(files)):
        try:
            before = git("show", f"{base}:{name}")
        except subprocess.CalledProcessError:
            before = ""                                      # a new file: nothing it held before
        path = ROOT / name
        after = path.read_text() if path.exists() else ""
        hunks, current = [], None
        for line in git("diff", "-U0", base, "--", name).splitlines():
            if line.startswith("@@"):
                current = [line, [], []]
                hunks.append(current)
            elif current is not None and line[:1] in "-+" and not line.startswith(("---", "+++")):
                if ASSERT.search(line[1:]):
                    current[1 if line[0] == "-" else 2].append(line)
        shown = [h for h in hunks if h[1] or h[2]]
        if not shown and count(before) <= count(after):
            continue
        print(f"== {name}: {count(before)} assert statements before, {count(after)} after")
        for head, gone, came in shown:
            print(head)
            for line in gone + came:
                print("  " + line)
        removed_any = removed_any or any(h[1] for h in shown) or count(after) < count(before)
    return 1 if removed_any else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
