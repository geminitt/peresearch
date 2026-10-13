"""The mascot: the owner's arctic fox (docs/mascot/fox.svg at rest, fox_thinking.svg at work), drawn with half
blocks, two pixel rows to a line. It lies on the top border of the box the user types in, its tail hanging over the
box's right edge (the interface places it).

Its colors are the sketch's own, the one exception to the interface's terminal-palette rule (the owner's choice,
2026-10-05, for the mascot); a terminal without 24-bit color gets the nearest colors it has.
"""
from rich.color import Color
from rich.style import Style
from rich.text import Text

COLORS = {"O": Color.parse("#4f6a8a"), "W": Color.parse("#f5f8fc"), "S": Color.parse("#d3e2f1"),
          "E": Color.parse("#16243b"), "T": Color.parse("#7cc4ee"), "B": Color.parse("#ffffff")}

FOX = [".OO..OO..............",          # fox.svg, 21 x 10: lying, its belly on row 5, the tail hanging from x 17
       "OWSOOSWO.............",
       "OWWWWWWOOOOOOOOO.....",
       "OWEWWEWWWWWWWWWWOOO..",
       "OSSESSWSSSSSSSSWOOWO.",
       ".OOOOOOOOOOOOOOOOOWWO",
       ".................OWWO",
       ".................OWWO",
       ".................OTTO",
       "..................OO."]
BUBBLE = ["...........OOOOOOOOO.",       # fox_thinking.svg, rows 0-6: the bubble, its tail down to the fox's head
          "..........OBBBBBBBBBO",
          "..........OBBBBBBBBBO",
          "..........OBBBBBBBBBO",
          "...........OOOOOOOOO.",
          "...........O.........",
          "..........O.........."]
DOTS = [(13, 2), (15, 2), (17, 2)]      # the bubble's dots, on at the beats where the sketch's values are 1
SHOWN = [[1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0]]
WIDTH = len(FOX[0])
TAIL = 4                                # the last columns: the tail, over the box's right edge
FRAMES = len(SHOWN[0])                  # beats of the animation
STEP = 2.0 / FRAMES                     # the sketch's 2 s, in four beats
REST_LINES = len(FOX) // 2
WORK_LINES = (len(FOX) + 5 + 1) // 2    # the bubble's 5 rows above the fox, and one empty row for an even count


def _grid(beat: int | None) -> list[list[str]]:
    if beat is None:
        return [list(row) for row in FOX]
    grid = [["."] * WIDTH] + [list(row) for row in BUBBLE[:5]] + [list(row) for row in FOX]
    for (x, y), shown in zip(DOTS, SHOWN):
        if shown[beat]:
            grid[1 + y][x] = "E"
    for y, row in enumerate(BUBBLE[5:], start=6):      # the bubble's tail beside the fox's head
        for x, c in enumerate(row):
            if c != ".":
                grid[y][x] = c
    return grid


def frame(beat: int | None) -> list[Text]:
    """The fox at rest (None), or at work on beat 0-3 of its bubble; a line per two pixel rows."""
    grid = _grid(beat)
    lines = []
    for y in range(0, len(grid), 2):
        line = Text()
        for top, bottom in zip(grid[y], grid[y + 1]):
            t, b = COLORS.get(top), COLORS.get(bottom)
            if t is None and b is None:
                line.append(" ")
            elif b is None:
                line.append("▀", Style(color=t))
            elif t is None:
                line.append("▄", Style(color=b))
            elif top == bottom:
                line.append("█", Style(color=t))
            else:
                line.append("▀", Style(color=t, bgcolor=b))
        lines.append(line)
    return lines
