"""Contract of the mascot (peresearch.mascot), the owner's arctic fox of docs/mascot/fox.svg (at rest) and
fox_thinking.svg (at work, its bubble's dots animated): every frame is the sketch's pixels, two pixel rows to a line
(half blocks), its colors exactly the sketch's, its timing the sketch's own."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from rich.color import Color
from rich.console import Console

from peresearch import mascot

CONSOLE = Console()
SKETCHES = Path(__file__).resolve().parents[1] / "docs" / "mascot"
NS = "{http://www.w3.org/2000/svg}"


def sketch(name: str):
    """(pixel rows of RGB tuples or None, size, the animated rects: [(x, y, rgb, values, keyTimes, dur)])."""
    root = ET.parse(SKETCHES / name).getroot()
    _, _, w, h = (int(v) for v in root.get("viewBox").split())
    grid = [[None] * w for _ in range(h)]
    animated = []
    for r in root.iter(NS + "rect"):
        x, y, rw, rh = (int(r.get(k)) for k in ("x", "y", "width", "height"))
        rgb = Color.parse(r.get("fill")).get_truecolor()
        anim = r.find(NS + "animate")
        if anim is not None:
            animated.append((x, y, tuple(rgb), anim.get("values").split(";"), anim.get("keyTimes").split(";"),
                             float(re.sub("s$", "", anim.get("dur")))))
            continue
        for yy in range(y, y + rh):
            for xx in range(x, x + rw):
                grid[yy][xx] = tuple(rgb)
    return grid, (w, h), animated


def pixels(lines):
    """The pixel grid a frame draws, read back from its characters and colors (None: the terminal's background)."""
    rgb = lambda color: tuple(color.get_truecolor()) if color else None        # noqa: E731
    grid = []
    for line in lines:
        top, bottom = [], []
        assert len(line.plain) == mascot.WIDTH, line.plain
        for char, style in [(line.plain[i], line.get_style_at_offset(CONSOLE, i)) for i in range(mascot.WIDTH)]:
            fg, bg = rgb(style.color), rgb(style.bgcolor)
            assert char in " ▀▄█", char
            t, b = {" ": (None, None), "▀": (fg, bg), "▄": (bg, fg), "█": (fg, fg)}[char]
            top.append(t)
            bottom.append(b)
        grid += [top, bottom]
    return grid


def test_at_rest_it_is_the_sketch():
    want, (w, h), animated = sketch("fox.svg")
    assert (w, h) == (21, 10) and not animated
    lines = mascot.frame(None)
    assert len(lines) == h // 2 == mascot.REST_LINES and pixels(lines) == want


THINKING, (TW, TH), DOTS = sketch("fox_thinking.svg")


@pytest.mark.parametrize("k", range(4))
def test_at_work_each_frame_is_the_sketch_with_its_dots(k):
    """The bubble's dots as the sketch animates them (each dot shown where its value at keyTime k is 1); one empty
    pixel row above makes the 15 rows two to a line; the fox itself lies where it lies at rest."""
    want = [row[:] for row in THINKING]
    for x, y, rgb, values, times, _ in DOTS:
        if values[k] == "1":
            want[y][x] = rgb
    lines = mascot.frame(k)
    assert len(lines) == (TH + 1) // 2 == mascot.WORK_LINES and pixels(lines) == [[None] * TW] + want
    rest, drawn = sketch("fox.svg")[0], pixels(lines)[-10:]              # the same fox (the bubble's tail beside
    assert all(drawn[y][x] == c for y, row in enumerate(rest) for x, c in enumerate(row) if c)   # its head)


def test_the_timing_and_the_colors_are_the_sketchs():
    _, _, (dot, *_) = sketch("fox_thinking.svg")
    _, _, _, values, times, dur = dot
    assert mascot.FRAMES == len(times) == 4 and mascot.STEP == dur / len(times) == 0.5    # 2 s, four beats
    used = {c for row in sketch("fox.svg")[0] + THINKING for c in row if c} | {d[2] for d in DOTS}
    assert {tuple(c.get_truecolor()) for c in mascot.COLORS.values()} == used
    assert mascot.WIDTH == 21 and mascot.TAIL == 4                     # the tail: the last 4 columns, hanging down
