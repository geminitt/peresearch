"""Architecture diagrams of peresearch, written as SVG from the descriptions below.

    pixi run python docs/architecture/draw.py            # rewrites docs/architecture/*.svg
    pixi run python docs/architecture/draw.py --png DIR  # also renders PNG previews into DIR

Every box names the code it stands for (file and function), so a failure can be traced to its place. Numbered red
marks point at open items; their state is kept in the project's STATUS.md, not here. Text is measured with the
DejaVu Sans metrics and a label that does not fit its box stops the script, so a diagram never ships cut off.
"""

import html
import sys
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent
AS_OF = "working tree on 5ca550a, 2026-10-12"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONTS = {name: pymupdf.Font(fontfile=str(FONT_DIR / f)) if (FONT_DIR / f).exists() else pymupdf.Font("helv")
         for name, f in (("sans", "DejaVuSans.ttf"), ("bold", "DejaVuSans-Bold.ttf"), ("mono", "DejaVuSansMono.ttf"))}
SANS = "'DejaVu Sans', 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'DejaVu Sans Mono', Consolas, Menlo, monospace"

# fill, stroke, text
KINDS = {
    "ui": ("#E8F0FE", "#3B6FD8", "#1A3D8F"),
    "agent": ("#F1ECFB", "#7A55C9", "#45288A"),
    "llm": ("#FDF0E2", "#C9741A", "#7A4205"),
    "retrieval": ("#E3F4EC", "#2E9467", "#155A3B"),
    "web": ("#E4F4F7", "#2A8FA6", "#12505E"),
    "guard": ("#FCE8E8", "#C94343", "#7D1F1F"),
    "store": ("#F0EFEA", "#8A877C", "#3D3B35"),
    "external": ("#FFFFFF", "#9A9A94", "#3D3B35"),
    "decision": ("#FFF6D6", "#B8960B", "#5E4B00"),
    "end": ("#EDEDED", "#6B6B6B", "#2B2B2B"),
    "ok": ("#DDF2E3", "#2E9467", "#155A3B"),
}
LEGEND = [("ui", "interface"), ("agent", "agent and tools"), ("llm", "model client"), ("retrieval", "ZetokRAG"),
          ("web", "web"), ("guard", "guard: access, redaction, checks"), ("store", "files on disk"),
          ("external", "outside this machine"), ("decision", "decision")]
ARROWS = {"flow": ("#55554F", ""), "back": ("#8A877C", "6 4"), "error": ("#C94343", "6 4"), "exit": ("#C94343", "")}

# Open items marked on the diagrams (number -> short label); their state lives in CLAUDE.md.
OPEN = {
    5: "the embedding and rerank endpoints are the user's (no model runs in peresearch, 2026-10-08): their speed and "
       "quality are theirs; checked on Ollama, llama.cpp and vLLM (runs/serve/endpoints-2026-10-08.json); in RAM: "
       "vectors 4 KiB per chunk (1,024 dimensions); a corpus far larger than the owner's not measured",
    8: "grep walks the folders on every call (files.walk: 1.1–1.4 s of 2.2 s for 13,402 files on the owner's "
       "machine); speed on a large corpus and on macOS or Windows not measured",
    9: "the endpoint is the user's, through its provider's protocol (Chat Completions, Responses, Anthropic Messages, "
       "Gemini Interactions); checked with small models on Ollama 0.32.15, llama.cpp and vLLM 0.29 (2026-10-06, recorded "
       "in tests/data/servers); LM Studio and the cloud APIs only against recorded or SDK-shaped fixtures",
    12: "encoding guess: a text under 64 bytes whose language is not recognized is read with replacement "
        "characters (said in the note); cp1252 and cp1250 are confused (measured)",
    13: "PDF reading order measured on 10 LaTeX arXiv papers only (prose of 69 two-column pages right); Word, "
        "scanned and vertical CJK layouts not measured; no OCR in the first version (needs_ocr is reported)",
    14: "Word, PowerPoint, Excel and HTML have text readers; complex visual layouts remain unverified; "
        ".csv, .json, .log are read and grepped by the tools but not ranked",
    16: "level B checked on Linux only (clean Docker image, CPU, not root, POSIX locale); macOS, Windows "
        "(junctions, cloud-synced placeholders) not checked",
    17: "pixi.toml lists linux-64 only: peresearch, and the ripgrep it ships, do not install on macOS or Windows yet",
    18: "Semantic Scholar without a key answered 429 to 3 of 3 requests (2026-10-03), and its key needs a school "
        "or work email: it is asked every search and named when it does not answer; OpenAlex and arXiv answer",
}


def width(text: str, size: float, font: str = "sans") -> float:
    return FONTS[font].text_length(text, fontsize=size)


class Box:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def n(self, t=0.5):
        return (self.x + self.w * t, self.y)

    def s(self, t=0.5):
        return (self.x + self.w * t, self.y + self.h)

    def w_(self, t=0.5):
        return (self.x, self.y + self.h * t)

    def e(self, t=0.5):
        return (self.x + self.w, self.y + self.h * t)


class Diagram:
    PAD = 12
    TITLE, CODE, LINE = 14.5, 11.5, 12.5

    def __init__(self, name: str, title: str, subtitle: str, w: int, h: int):
        self.name, self.w, self.h = name, w, h
        self.parts: list[str] = []
        self.top: list[str] = []          # labels and marks drawn over the arrows
        self.boxes: dict[str, Box] = {}
        self.marks: set[int] = set()
        self.text(28, 42, title, 22, "bold", "#1F1F1F")
        self.text(28, 68, subtitle, 13, "sans", "#5A5A55")

    # --- primitives ---

    def text(self, x, y, s, size, font="sans", color="#1F1F1F", anchor="start", layer=None):
        family = MONO if font == "mono" else SANS
        weight = ' font-weight="700"' if font == "bold" else ""
        (self.parts if layer is None else layer).append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}"{weight} fill="{color}" '
            f'text-anchor="{anchor}">{html.escape(s)}</text>')

    def group(self, x, y, w, h, label, color="#B5B3AA"):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="none" stroke="{color}" '
                          f'stroke-width="1.4" stroke-dasharray="9 6"/>')
        self.text(x + 16, y + 24, label, 14, "bold", "#6B6963")

    def box(self, key, x, y, w, h, kind, title, code=None, lines=(), mark=None):
        fill, stroke, ink = KINDS[kind]
        dash = ' stroke-dasharray="7 5"' if kind == "external" else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" '
                          f'stroke-width="1.4"{dash}/>')
        room = w - 2 * self.PAD
        checks = [(title, self.TITLE, "bold")] + ([(code, self.CODE, "mono")] if code else []) + \
                 [(line, self.LINE, "sans") for line in lines]
        for s, size, font in checks:
            if width(s, size, font) > room:
                raise SystemExit(f"{self.name}: '{s}' is {width(s, size, font):.0f}px, box '{key}' has {room}px")
        yy = y + 23
        self.text(x + self.PAD, yy, title, self.TITLE, "bold", ink)
        if code:
            yy += 18
            self.text(x + self.PAD, yy, code, self.CODE, "mono", stroke)
        yy += 6 if lines else 0
        for line in lines:
            yy += 17
            self.text(x + self.PAD, yy, line, self.LINE, "sans", "#2B2B28")
        if yy + 10 > y + h:
            raise SystemExit(f"{self.name}: box '{key}' needs {yy + 10 - y:.0f}px of height, has {h}")
        if mark:
            self.mark(x + w - 14, y + 14, mark)
        self.boxes[key] = Box(x, y, w, h)
        return self.boxes[key]

    def mark(self, x, y, n: int):
        self.marks.add(n)
        self.top.append(f'<circle cx="{x}" cy="{y}" r="10" fill="#C94343"/>')
        self.text(x, y + 4.5, str(n), 12, "bold", "#FFFFFF", "middle", layer=self.top)

    def arrow(self, points, label=None, style="flow", at=0.5, seg=None, both=False, dx=0, dy=-6):
        color, dash = ARROWS[style]
        d = " ".join(f"{'M' if i == 0 else 'L'} {x:.1f} {y:.1f}" for i, (x, y) in enumerate(points))
        start = ' marker-start="url(#arrow-%s)"' % style if both else ""
        dash = f' stroke-dasharray="{dash}"' if dash else ""
        width_ = 2.2 if style == "exit" else 1.5
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width_}"{dash} '
                          f'marker-end="url(#arrow-{style})"{start}/>')
        if label:
            i = (len(points) - 2) // 2 if seg is None else seg
            (x1, y1), (x2, y2) = points[i], points[i + 1]
            lx, ly = x1 + (x2 - x1) * at + dx, y1 + (y2 - y1) * at + dy
            ink = "#B23A3A" if style in ("error", "exit") else "#3D3B35"
            tw = width(label, 11.5)
            anchor = "middle"
            self.top.append(f'<rect x="{lx - tw / 2 - 4:.1f}" y="{ly - 12:.1f}" width="{tw + 8:.1f}" height="17" '
                            f'rx="4" fill="#FCFCFA" fill-opacity="0.92"/>')
            self.text(lx, ly, label, 11.5, "sans", ink, anchor, layer=self.top)

    def legend(self, x, y, kinds=None, cols=1, colw=230):
        items = [("box", k, v) for k, v in LEGEND if kinds is None or k in kinds]
        items += [("line", s, v) for s, v in (("flow", "call or data"), ("back", "returns, next round"),
                                              ("error", "error path"), ("exit", "leaves the machine"))]
        self.text(x, y, "Legend", 13, "bold", "#3D3B35")
        rows = -(-len(items) // cols)
        for i, (what, key, label) in enumerate(items):
            cx, yy = x + colw * (i // rows), y + 12 + 21 * (i % rows)
            if what == "box":
                fill, stroke, _ = KINDS[key]
                dash = ' stroke-dasharray="4 3"' if key == "external" else ""
                self.parts.append(f'<rect x="{cx}" y="{yy}" width="22" height="14" rx="3" fill="{fill}" '
                                  f'stroke="{stroke}"{dash}/>')
            else:
                color, dash = ARROWS[key]
                dash = f' stroke-dasharray="{dash}"' if dash else ""
                self.parts.append(f'<path d="M {cx} {yy + 7} L {cx + 22} {yy + 7}" stroke="{color}" '
                                  f'stroke-width="1.8"{dash} marker-end="url(#arrow-{key})"/>')
            self.text(cx + 30, yy + 11.5, label, 12, "sans", "#3D3B35")
        return y + 12 + 21 * rows

    def footnotes(self, x, y, w):
        """The open items marked on this diagram, wrapped to `w` pixels."""
        if not self.marks:
            return
        self.text(x, y, "Open items (numbers on the boxes)", 13, "bold", "#7D1F1F")
        self.text(x, y + 17, f"{AS_OF}; their state is kept in STATUS.md", 12, "sans", "#7D1F1F")
        yy = y + 25
        for n in sorted(self.marks):
            words, line, out = OPEN[n].split(), "", []
            for word in words:
                trial = f"{line} {word}".strip()
                if width(trial, 12) > w - 30:
                    out.append(line)
                    line = word
                else:
                    line = trial
            out.append(line)
            yy += 20
            self.parts.append(f'<circle cx="{x + 9}" cy="{yy - 4}" r="9" fill="#C94343"/>')
            self.text(x + 9, yy, str(n), 11, "bold", "#FFFFFF", "middle")
            for i, line in enumerate(out):
                self.text(x + 26, yy + 17 * i, line, 12, "sans", "#3D3B35")
            yy += 17 * (len(out) - 1)
        return yy

    def svg(self) -> str:
        markers = "".join(
            f'<marker id="arrow-{s}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>'
            for s, (c, _) in ARROWS.items())
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">\n<defs>{markers}</defs>\n'
                f'<rect width="{self.w}" height="{self.h}" fill="#FCFCFA"/>\n'
                + "\n".join(self.parts + self.top) + "\n</svg>\n")

    def save(self):
        path = OUT / f"{self.name}.svg"
        path.write_text(self.svg())
        return path


# ----------------------------------------------------------------------------------------------------------------

def overview():
    d = Diagram("overview", "peresearch: system overview",
                f"One flow: interface → agent loop → model + tools, with guard on every way in and out. "
                f"Details in the other diagrams. {AS_OF}.", 1660, 1430)
    d.group(20, 92, 1255, 1035, "This machine")
    user = d.box("user", 45, 130, 300, 62, "external", "User", lines=["works in a terminal: TUI or CLI"])
    tui = d.box("tui", 45, 228, 300, 148, "ui", "TUI", "tui.py · windows.py", [
        "Textual; grouped floating windows", "folders, models, settings, sessions", "daemon agent/index; permission dialog",
        "untrusted text: guard.sanitize"])
    cli = d.box("cli", 45, 400, 300, 96, "ui", "CLI", "cli.py: main", [
        "no subcommand: opens the TUI", "optional CLI: setup, find, ask, index"])
    ws = d.box("ws", 45, 520, 300, 96, "agent", "Declared folders", "workspace.py: Workspace", [
        "TUI /folders: add, remove, index", "CLI add, remove: run index after"])
    d.box("store", 45, 690, 330, 190, "store", "$PERESEARCH_HOME", "~/.local/share/peresearch", [
        "config.json      declared folders", "settings.env     active keys (0600)",
        "models.json      saved APIs and IDs",
        "index.sqlite     files, chunks, vectors", "sessions/<p>/*.jsonl  a file per conversation",
        "web/             page cache", "audit.jsonl      crossings (counts only)"])
    agent = d.box("agent", 410, 200, 370, 180, "agent", "Agent loop", "assembly.py → agent.py: Agent.ask", [
        "product/eval: shared assembly and limits", "tool results wrapped as untrusted",
        "citations and quotes checked", "writes the conversation's file"])
    tools = d.box("tools", 410, 440, 370, 182, "agent", "Tools", "registry.py: Registry.run · tools.py", [
        "tree · search_notes · overview · grep · glob · read", "web_search · search_papers · wikipedia · fetch",
        "one pipeline: arguments, permission, budget, run", "sources N1…, W1… for citations"], mark=8)
    rule = d.box("rule", 410, 685, 370, 96, "guard", "Reading rule", "guard.allowed · Toolbox._path", [
        "declared folders: read; protected: refused", "elsewhere under ~: ask the user"])
    folders = d.box("folders", 410, 840, 370, 80, "store", "Declared folders on disk", None, [
        "the user's notes, projects, courses"])
    llm = d.box("llm", 870, 200, 290, 162, "llm", "Model client", "llm.py: LLM.chat", [
        "streamed; stop() within 0.1 s", "retries a cold or dropped server", "tool calls recovered from text",
        "model profile; tools via the prompt"])
    search = d.box("search", 870, 420, 290, 120, "retrieval", "ZetokRAG search", "zetokrag/search.py", [
        "BM25 + dense → fuse → rerank", "copies folded, all ranked", "reads a snapshot of index.sqlite"])
    web = d.box("web", 870, 580, 290, 130, "web", "Web client", "web.py: Web", [
        "search: the user's MCP server (mcp.py)", "fetch: httpx + trafilatura", "Jina Reader: with the user's key"])
    index = d.box("index", 870, 770, 290, 140, "retrieval", "ZetokRAG index", "zetokrag/index.py: Index", [
        "walk, filter data, chunk, embed", "secret paragraphs withheld", "commits every 60 s: resumable"])
    models = d.box("models", 870, 960, 290, 112, "retrieval", "Search endpoints", "zetokrag/embed.py", [
        "the user's /embeddings, /rerank", "texts redacted, audited", "none set: words alone (BM25)"], mark=5)
    gm = d.box("gm", 1195, 240, 160, 96, "guard", "Exit: model", "guard.outbound", ["secrets redacted,", "crossing audited"])
    gw = d.box("gw", 1195, 600, 160, 96, "guard", "Exit: web", "outbound_query", ["a query or URL with", "a secret: refused"])
    em = d.box("em", 1390, 205, 250, 168, "external", "Model endpoint", "its provider's protocol", [
        "a provider's, or the user's", "own server: Ollama, llama.cpp,", "LM Studio, vLLM…"], mark=9)
    ew = d.box("ew", 1390, 585, 250, 125, "external", "Web", None, [
        "the user's search MCP server", "web pages over HTTPS", "r.jina.ai (the user's key)"])
    hf = d.box("hf", 1390, 985, 250, 80, "external", "The user's servers", None, ["Ollama, llama.cpp, vLLM, a provider"])

    d.arrow([user.s(), tui.n()], both=True)
    d.arrow([tui.e(0.3), (agent.x, tui.y + tui.h * 0.3)], "question")
    d.arrow([(agent.x, tui.y + tui.h * 0.62), tui.e(0.62)], "events", style="back")
    d.arrow([cli.e(0.4), (385, cli.y + cli.h * 0.4), (385, 360), (agent.x, 360)], "ask", seg=0, at=0.45)
    d.arrow([ws.e(0.5), (390, ws.y + 48), (390, 640), (835, 640), (835, index.y + 40), index.w_(40 / 140)],
            "Index.update", seg=3, at=0.55, dy=0)
    d.arrow([agent.e(0.3), (llm.x, agent.y + agent.h * 0.3)], "request", dy=-7)
    d.arrow([(llm.x, agent.y + agent.h * 0.65), agent.e(0.65)], "reply", style="back")
    d.arrow([llm.e(0.4), (gm.x, llm.y + llm.h * 0.4)])
    d.arrow([gm.e(0.4), (em.x, gm.y + gm.h * 0.4)], "HTTPS", style="exit")
    d.arrow([agent.s(0.35), tools.n(0.35)], "tool calls", dx=-34)
    d.arrow([tools.n(0.7), agent.s(0.7)], "results", style="back", dx=30)
    d.arrow([tools.e(0.22), (search.x, tools.y + tools.h * 0.22)], "search_notes", dy=-7)
    d.arrow([tools.e(0.85), (web.x, tools.y + tools.h * 0.85)], "web tools", dy=-7)
    d.arrow([tools.s(), rule.n()], "tree, grep, glob, read", dx=70, dy=12)
    d.arrow([rule.s(), folders.n()], "allowed paths", dx=50)
    d.arrow([web.e(0.5), (gw.x, web.y + web.h * 0.5 + 3)])
    d.arrow([gw.e(0.45), (ew.x, gw.y + gw.h * 0.45)], "HTTPS", style="exit")
    d.arrow([search.e(0.65), (1177, search.y + search.h * 0.65), (1177, index.y + 110), index.e(110 / 140)])
    d.arrow([folders.e(0.5), (index.x, folders.y + 40)], "read", dy=-6)
    d.arrow([index.s(), models.n()], "embed", dx=28)
    d.arrow([models.e(0.6), (hf.x, models.y + models.h * 0.6)], "HTTP(S)", style="exit")
    d.legend(40, 1165, cols=3, colw=235)
    d.footnotes(780, 1165, 860)
    return d


def agent_loop():
    d = Diagram("agent-loop", "Agent loop: one question",
                f"agent.py: Agent.ask. The model reads the message and decides; tools until it answers or a limit "
                f"is hit. {AS_OF}.", 1540, 1420)
    start = d.box("start", 480, 100, 440, 148, "agent", "Agent.ask(question)", "agent.py", [
        "retain source IDs; a new Budget; the question's URLs", "system prompt; NO_WEB note without a provider",
        "history: this conversation's turns, whole;", "the oldest summarized by the model when they",
        "no longer fit; question marked current"])
    cancel = d.box("cancel", 550, 270, 300, 56, "decision", "cancelled (Esc)?")
    e_cancel = d.box("e_cancel", 980, 270, 300, 56, "end", "answer '(interrupted)'")
    limit = d.box("limit", 550, 366, 300, 56, "decision", "step limit set and hit, or > 900 s?")
    n_limit = d.box("n_limit", 980, 356, 330, 80, "agent", "Limit reached", None, [
        "message: answer now with what you have", "no tools offered on this call"])
    call = d.box("call", 480, 470, 440, 112, "llm", "Agent._chat → LLM.chat(Request, stop)", "agent._fit · _fold", [
        "oldest results trimmed, then the oldest turns", "summarized by the model, to fit the context; a 400",
        "for length: trimmed to the server's count, again"])
    calls = d.box("calls", 550, 614, 300, 56, "decision", "tool calls, no limit?")
    each = d.box("each", 30, 614, 400, 44, "agent", "for each tool call")
    before = d.box("before", 30, 698, 400, 130, "agent", "Before running it", "registry.py: Registry.run", [
        "arguments unreadable → the reason, not run", "schema, then permission (web off, not offered)",
        "budget: repeated → 'already called'; a limit set", "by the user and reached → limit message"])
    run = d.box("run", 30, 858, 400, 96, "agent", "The tool's function", "tools.py · recall (Kept)", [
        "refusals and errors returned as text; 429 waited", "events: tool, result"])
    wrap = d.box("wrap", 30, 994, 400, 96, "guard", "Wrap as untrusted data", "agent._wrap", [
        "<tool_output trust=\"untrusted data\">", "+ warning if it reads like an instruction"])
    empty = d.box("empty", 550, 714, 300, 56, "decision", "answer text empty?")
    reask = d.box("reask", 980, 704, 400, 96, "llm", "Ask once more", None, [
        "'Write the final answer now'", "reasoning off (thinking = False)"])
    length = d.box("length", 550, 840, 300, 56, "decision", "finish_reason = length?")
    n_len = d.box("n_len", 980, 840, 400, 56, "end", "stopped = 'length' (shown as cut)")
    check = d.box("check", 480, 940, 440, 112, "guard", "check(text, sources)", "agent.check", [
        "cited IDs identify known conversation excerpts", "a quote just before a citation is verbatim",
        "sources but no citation → flagged"])
    save = d.box("save", 480, 1092, 440, 80, "store", "Save the turn", "agent._remember", [
        "the conversation's file: turn, then its messages"])
    end = d.box("end", 550, 1212, 300, 50, "end", "return Answer; event 'answer'")

    d.arrow([start.s(), cancel.n()])
    d.arrow([cancel.e(), e_cancel.w_()], "yes")
    d.arrow([e_cancel.e(), (1505, 298), (1505, 925), (940, 925), (940, 962), (check.x + check.w, 962)],
            "still checked and saved", seg=2, at=0.75)
    d.arrow([cancel.s(), limit.n()], "no", dx=16)
    d.arrow([limit.e(), (n_limit.x, limit.y + 28)], "yes")
    d.arrow([n_limit.s(), (n_limit.x + 165, call.y + 52), call.e()])
    d.arrow([limit.s(), call.n()], "no", dx=16)
    d.arrow([call.s(), calls.n()])
    d.arrow([calls.w_(), (each.x + each.w, calls.y + 28)], "yes")
    d.arrow([each.s(), before.n()])
    d.arrow([before.s(), run.n()])
    d.arrow([run.s(), wrap.n()])
    d.arrow([wrap.e(), (455, wrap.y + 48), (455, cancel.y + 28), cancel.w_()], "next round", style="back",
            seg=1, at=0.45, dx=12)
    d.arrow([calls.s(), empty.n()], "no", dx=16)
    d.arrow([empty.e(), (reask.x, empty.y + 28)], "yes")
    d.arrow([reask.w_(0.75), (900, reask.y + 72), (900, length.y + 17), (length.x + length.w, length.y + 17)])
    d.arrow([empty.s(), length.n()], "no", dx=16)
    d.arrow([(length.x + length.w, length.y + 40), (n_len.x, length.y + 40)], "yes", dx=30)
    d.arrow([n_len.s(0.1), (1020, 911), (956, 911), (956, 995), (check.x + check.w, 995)])
    d.arrow([length.s(), check.n()], "no", dx=16)
    d.arrow([check.s(), save.n()])
    d.arrow([save.s(), end.n()])
    d.box("errors", 980, 960, 480, 148, "guard", "What leaves Agent.ask as an error", None, [
        "BudgetExceeded; Interrupted → cancelled", "connection errors after 5 tries (llm-client.svg)",
        "a 400 (e.g. over the context length); a bad setting", "a failed internal check (assert): a bug in peresearch",
        "the interface shows explain(error), with file:line"])
    d.box("events", 980, 1128, 480, 80, "ui", "Events to the interface (on_event)", None, [
        "think · tokens · tool · result · wait · answer"])
    d.legend(30, 1140, {"agent", "llm", "guard", "store", "decision", "ui"})
    d.footnotes(980, 1250, 520)
    return d


def llm_client():
    d = Diagram("llm-client", "Model client: one LLM.chat call",
                f"llm.py. Streamed so that stop() ends a reply at once; retried while a cold server wakes. {AS_OF}.",
                1500, 1240)
    out = d.box("out", 470, 100, 460, 96, "guard", "Outbound filter", "llm._outbound → guard.outbound", [
        "every message and tool-call argument:", "secrets redacted, crossing audited"])
    att = d.box("att", 470, 236, 460, 62, "llm", "Try n of 5", "LLM._send")
    bud = d.box("bud", 470, 338, 460, 112, "llm", "Profile and tool mode", "LLM.profile · prompt_tools", [
        "sampling only from the profile or the caller", "reply-budget field selected by capability",
        "prompt mode: tools described in the system prompt"])
    create = d.box("create", 470, 474, 460, 130, "llm", "_create(stop): streamed request", "chat.completions.create",
                   ["observe submitted messages for evaluation", "caller polls every 0.1 s: stop() → Interrupted",
                    "no first chunk after 5 s → 'not started yet'"])
    rec = d.box("rec", 470, 644, 460, 62, "llm", "Reasoning left in the text moved out", "llm.split_reasoning")
    calls = d.box("calls", 470, 746, 460, 146, "llm", "Tool calls", "llm._arguments", [
        "arguments: one JSON object, else {} + the reason", "(the last call of a reply that used every token:",
        "said cut); none → recovered from text or", "reasoning: Qwen XML, Hermes JSON (vllm#39056)",
        "refusal examples are never executed"])
    reply = d.box("reply", 470, 916, 460, 56, "end", "Reply: text, calls, reasoning, tokens, refusal")
    ep = d.box("ep", 990, 236, 480, 96, "external", "Model endpoint", None, [
        "a provider's API, in its own protocol, or", "the user's server (Ollama, llama.cpp, vLLM…)"], mark=9)
    ctx = d.box("ctx", 990, 100, 480, 112, "llm", "Context the server runs with", "LLM.context_length (once)", [
        "setting > server: vLLM max_model_len, llama.cpp", "meta.n_ctx, Ollama /api/ps > the model table"])
    reader = d.box("reader", 990, 450, 480, 182, "llm", "_read_stream (reader thread)", "llm._read_stream", [
        "joins text, refusal, reasoning and call pieces", "checks stop() between chunks",
        "closes the stream itself: a close from another", "thread left vLLM generating for 55.7 s",
        "no finish_reason → StreamCut"])
    stop = d.box("stop", 990, 672, 480, 96, "end", "Interrupted", None, [
        "stop() was true: the agent ends the question", "as cancelled; the reader closes the stream"])
    other = d.box("other", 30, 338, 400, 96, "guard", "Raised to Agent.ask", None, [
        "non-retryable errors, a 400 left,", "the last failed try → UI explains"])
    bad = d.box("bad", 30, 474, 400, 166, "guard", "Refusals and parser failures", None, [
        "explicit tool refusal → prompt mode, kept;", "parser failure → this request via the prompt;",
        "only unsupported fields fall back; unrelated", "400s keep capabilities; one alternate token",
        "field; context overflow is raised"])
    retry = d.box("retry", 30, 664, 400, 130, "guard", "Retryable errors", None, [
        "connection, timeout, 5xx, rate limit, StreamCut:", "Retry-After ≤ 60 s, else wait 10·2ⁿ s",
        "local server refused: 2 tries, 2 s apart", "last try: raised"])

    d.arrow([out.s(), att.n()])
    d.arrow([ctx.s(0.5), ep.n(0.5)], "asked once", style="exit", dx=46)
    d.arrow([att.s(), bud.n()])
    d.arrow([bud.s(), create.n()])
    d.arrow([create.s(), rec.n()])
    d.arrow([rec.s(), calls.n()], "reply received", dx=56)
    d.arrow([calls.s(), reply.n()])
    d.arrow([create.e(0.25), (950, create.y + 32.5), (950, ep.y + 70), (ep.x, ep.y + 70)], "request", seg=1)
    d.arrow([ep.s(0.5), reader.n(0.5)], "chunks", style="exit", dx=34)
    d.arrow([reader.w_(0.6), (create.x + create.w, reader.y + reader.h * 0.6)], "reply", style="back")
    d.arrow([create.e(0.92), (962, create.y + create.h * 0.92), (962, stop.y + 48), stop.w_()], "stop()",
            style="error", seg=1)
    d.arrow([bad.n(0.5), other.s(0.5)], "raised", style="error", dx=36)
    d.arrow([create.w_(0.3), (bad.x + bad.w, create.y + create.h * 0.3)], style="error")
    d.arrow([create.w_(0.75), (448, create.y + create.h * 0.75), (448, retry.y + 30), (retry.x + retry.w, retry.y + 30)],
            style="error")
    d.arrow([bad.w_(0.3), (22, bad.y + bad.h * 0.3), (22, 279), (att.x, 279)], style="back")
    d.arrow([retry.w_(0.7), (12, retry.y + retry.h * 0.7), (12, 255), (att.x, 255)], "next try", style="back",
            seg=2, at=0.5)
    d.legend(1000, 810, {"llm", "guard", "store", "external", "end"})
    d.footnotes(990, 1030, 480)
    return d


def model_frame():
    d = Diagram("model-frame", "Model frame: one gateway, any provider",
                f"What peresearch does for whatever model the user chooses, through its provider's protocol; the author's "
                f"test tools are kept apart (bottom). {AS_OF}.", 1500, 1120)
    st = d.box("st", 30, 100, 400, 130, "store", "Settings", "connections.py · TUI /model", [
        "chat / embedding / reranker tabs; hidden API keys", "saved IDs or explicit GET /models; no inference",
        "select: active settings.env; catalog: models.json"])
    chk = d.box("chk", 30, 270, 400, 166, "ui", "Model check", "llm.check_model · TUI /check", [
        "explicit /check: calls a tool (native, else via", "the prompt, or the other way when the table is",
        "wrong), where reasoning comes back, the", "context and its source, tokens per second"])
    prof = d.box("prof", 520, 100, 440, 166, "llm", "Model profile", "profiles.py: profile", [
        "a trimmed models.dev table (2,336 names), found", "however a server spells the name (Ollama tags,",
        "HF paths, -instruct rows), then the family's", "settings, then the user's (tools, temperature, top_p)"])
    shape = d.box("shape", 520, 306, 440, 130, "llm", "The request", "LLM._send · prompt_tools", [
        "sampling only when the profile has it", "the reasoning switch for families that have one",
        "tools native, or described in the system prompt", "reply-budget field from the model capability"])
    read = d.box("read", 520, 476, 440, 148, "llm", "The reply", "_read_stream · split_reasoning · _arguments", [
        "streamed; Esc stops it within 0.1 s", "reasoning apart from the answer (field or tags)",
        "calls native or read from the text; arguments", "one JSON object, read by each tool's types",
        "structured refusal reason preserved separately"])
    err = d.box("err", 520, 664, 440, 166, "guard", "When the server says no", "LLM._send · llm.too_long", [
        "tools refused → through the prompt from then on", "its parser failed → this request through the prompt",
        "too long → the agent trims and sends again", "cold, dropped, rate-limited → retried",
        "(Retry-After ≤ 60 s honoured)"])
    ctx = d.box("ctx", 1050, 100, 420, 166, "llm", "The context it runs with", "LLM.context_length", [
        "PERESEARCH_LLM_CONTEXT, else the server's own", "(vLLM max_model_len, llama.cpp meta.n_ctx,",
        "Ollama /api/ps for the loaded model), else the", "model table's maximum (often far larger)"])
    ep = d.box("ep", 1050, 340, 420, 148, "external", "The provider's API", None, [
        "a provider's API (models ready to use)", "or the user's own server, anywhere: Ollama,",
        "llama.cpp, LM Studio, vLLM, a rented GPU…", "(how it is hosted is the user's)"], mark=9)
    d.group(20, 880, 1460, 200, "Test tools (the author's, not part of peresearch)")
    mx = d.box("mx", 40, 920, 460, 130, "external", "Compatibility matrix", "runs/serve/compat-matrix-*.py", [
        "Ollama, llama.cpp, vLLM with small models", "(pixi env `servers`), each way a user may run them",
        "every exchange recorded"])
    rec = d.box("rec", 530, 920, 440, 130, "store", "Recorded exchanges", "tests/data/servers/*.json.gz", [
        "replayed offline through the real OpenAI client", "by the tests: a change that misreads a real",
        "server's stream, error or listing fails"])
    md = d.box("md", 1000, 920, 460, 130, "external", "Modal serving", "deploy/ (serving.svg)", [
        "Qwen3.6-35B-A3B on vLLM for the agent's", "measurements; chat-local: the laptop 0.8B"])
    d.arrow([st.e(0.4), (prof.x, st.y + st.h * 0.4)], "model name")
    d.arrow([prof.s(), shape.n()])
    d.arrow([shape.e(0.4), (1010, shape.y + shape.h * 0.4), (1010, ep.y + 60), (ep.x, ep.y + 60)], "request",
            style="exit", seg=1)
    d.arrow([(ep.x, ep.y + 110), (1025, ep.y + 110), (1025, read.y + 50), read.e(50 / 148)], "streamed chunks",
            style="back", seg=1)
    d.arrow([read.s(), err.n()], "an error instead", style="error", dx=60)
    d.arrow([ctx.s(), ep.n()], "asked once", style="exit", dx=46)
    d.arrow([st.s(), chk.n()], "explicit /check", dx=50)
    d.arrow([mx.e(0.5), (rec.x, mx.y + 65)], "kept")
    d.legend(40, 480, {"store", "ui", "llm", "guard", "external"}, cols=1)
    d.footnotes(1050, 540, 420)
    return d


def storage_frame():
    d = Diagram("storage-frame", "Storage frame: what peresearch keeps, and for whom",
                f"Everything lives in $PERESEARCH_HOME (default ~/.local/share/peresearch), a folder only the user may "
                f"enter (guard.ensure_home, at every command). {AS_OF}.", 1500, 980)
    d.group(20, 92, 1460, 640, "$PERESEARCH_HOME  (0700)")
    ses = d.box("ses", 40, 130, 460, 166, "store", "Conversations", "sessions/<project>/<conv>.jsonl (0600)", [
        "a header; each turn: answer, sources, scope, usage,", "then its messages whole (calls, provider data,",
        "results as notes); summaries; names (last wins);", "/sessions: resume, new, rename; /status: usage"])
    res = d.box("res", 40, 336, 460, 148, "store", "Tool results (same file)", "agent.ResultStore · InMemoryResults", [
        "every result redacted, with accessed path scope;", "R refs; excerpt ranges + hashes verify resume;",
        "restorable compression; originals retained"])
    idx = d.box("idx", 540, 130, 440, 148, "retrieval", "Index", "index.sqlite (+ text/: 0700, 0600)", [
        "files, chunks, vectors, what was left out;", "the extracted text of PDFs, notebooks, documents"])
    web = d.box("web", 540, 318, 440, 96, "web", "Web cache", "web/", ["pages read, by address"])
    cfg = d.box("cfg", 1020, 130, 440, 148, "guard", "Configuration and keys", "config.json · settings.env · models.json", [
        "declared folders, exclude / include;", "active settings and saved model connections",
        "catalog: atomic replace, mode 0600"])
    aud = d.box("aud", 1020, 318, 440, 96, "guard", "Audit log", "audit.jsonl", ["what left the machine: counts only"])
    out = d.box("out", 540, 454, 440, 96, "ui", "Exports", "/sessions → Export (user destination)", [
        "a conversation as Markdown, outside this folder"])
    note = d.box("note", 1020, 454, 440, 130, "end", "Writes that survive a crash", "agent._append_private", [
        "one line appended at a time; a line a crash", "cut (mid-character too) is ended first and",
        "skipped when read: that line alone is lost"])
    agent = d.box("agent", 40, 780, 460, 96, "agent", "Agent", "agent.py", ["writes turns, names, tool results"])
    tui = d.box("tui", 540, 780, 440, 96, "ui", "Interface", "tui.py · windows.py · cli.py", ["reads them for /sessions, /status"])
    d.arrow([agent.w_(0.5), (28, agent.y + 48), (28, ses.y + 120), ses.w_(120 / 166)])
    d.text(36, 600, "turns, names", 11, color="#55554F")
    d.arrow([agent.n(0.75), res.s(0.75)], "results", dx=40)
    d.arrow([tui.n(0.2), (tui.x + tui.w * 0.2, 620), (ses.x + ses.w + 20, 620), (ses.x + ses.w + 20, ses.y + 120),
             ses.e(120 / 166)], "reads", style="back", seg=1)
    d.arrow([tui.n(0.6), out.s(0.6)], "Export", dx=36)
    d.legend(1020, 760, {"store", "retrieval", "web", "guard", "ui", "agent", "end"}, cols=2, colw=220)
    return d


def harness_frame():
    d = Diagram("harness-frame", "Harness frame: around the model, the question kept on its way",
                f"What Harness wave 1 built: settings and reports for the user, the rules and the date in the prompt, a loop "
                f"that ends a refused run, and the conversation kept within the model's context. {AS_OF}.", 1570, 1190)
    ui = d.box("ui", 30, 100, 420, 148, "ui", "Interface", "tui.py: Chat.command · config · context_text", [
        "/config: the limits, reasoning, context, rerank", "depth (shown with their source, set, reset)",
        "/context: what the next request holds, in tokens"])
    st = d.box("st", 1050, 100, 420, 130, "store", "Settings", "settings.py: CONFIG · row · check", [
        "settings.env (0600): checked before kept;", "the environment wins (named when it does)",
        "assembly and workspace index use them"])
    sysp = d.box("sysp", 520, 100, 440, 148, "agent", "The system prompt", "Agent._system", [
        "today's date; the user's files first, then the", "web; search_notes first, grep to complete;",
        "note what the files cover, then the web"])
    loop = d.box("loop", 520, 300, 440, 166, "agent", "The loop", "Agent.ask · Limits.from_settings", [
        "product/eval share tools and settings", "new eval agent per question; repeat not run;",
        "three rounds of refused calls in a row:", "told to answer with what it has"])
    tools = d.box("tools", 1050, 300, 420, 130, "retrieval", "Tools", "tools.py (Capabilities frame)", [
        "files, then the web (instructed); /local refuses", "the web ones by name; at most 1 MiB, then said"])
    ctx = d.box("ctx", 520, 520, 440, 184, "agent", "Within the model's context", "Agent._chat · _fit · _context", [
        "a page (50 KiB, 2,000 lines) or the room left:", "head + [R4: … recall(\"R4\", from=n)]; the oldest",
        "results become [R3: read(…) returned …]; then the", "oldest turns summarized by the model (kept);",
        "latest round cut; access checked on replay", "history dependencies inherit file scope"])
    store = d.box("store", 30, 540, 420, 148, "store", "Tool-result store", "agent.ResultStore (Storage frame)", [
        "every result whole, by its ref; recall(ref, from)", "scope rechecked; revoked evidence withheld",
        "in memory when no history is recorded"])
    srv = d.box("srv", 520, 760, 440, 166, "guard", "When the server says no, or cuts", "Agent._chat · llm.too_long", [
        "too long: trimmed to what it counted, resent", "cut without a word (Ollama): its count far",
        "below what was sent; taken as its context,", "trimmed, resent; a context a server showed",
        "wins over a larger one known"])
    mdl = d.box("mdl", 1050, 760, 420, 130, "external", "The model", "llm.py (Model frame)", [
        "through its provider's protocol"])
    d.group(20, 1010, 1460, 150, "Test tools (the author's, not part of peresearch)")
    acc = d.box("acc", 40, 1045, 700, 96, "external", "Acceptance prompts", "eval/acceptance.py", [
        "the 10 approved prompts through the real agent: the harness's guarantees checked, the rules recorded"])
    srvs = d.box("srvs", 770, 1045, 690, 96, "external", "Real servers", "runs/e2e/e2e-silent-cut-*.py", [
        "Ollama cutting silently; the local vLLM through the interface"])
    d.arrow([ui.s(0.5), (ui.x + ui.w * 0.5, loop.y + 60), loop.w_(60 / 166)], "question", seg=1)
    d.arrow([ui.n(0.5), (ui.x + ui.w * 0.5, 91), (st.x + st.w * 0.5, 91), st.n(0.5)], "/config saves", seg=1)
    d.arrow([st.e(0.7), (1488, st.y + st.h * 0.7), (1488, mdl.y + 30), mdl.e(30 / 130)], "reasoning, context",
            seg=1)
    d.arrow([st.s(0.5), (st.x + st.w * 0.5, 280), (loop.x + loop.w * 0.8, 280), loop.n(0.8)], "limits", seg=1)
    d.arrow([sysp.s(0.3), loop.n(0.3)])
    d.arrow([loop.e(0.4), (tools.x, loop.y + loop.h * 0.4)], "calls")
    d.arrow([loop.s(0.5), ctx.n(0.5)], "each call")
    d.arrow([ctx.w_(0.4), (store.x + store.w, ctx.y + ctx.h * 0.4)], "kept / recall", style="back")
    d.arrow([ctx.e(0.6), (1010, ctx.y + ctx.h * 0.6), (1010, mdl.y + 40), (mdl.x, mdl.y + 40)], "request",
            style="exit", seg=1)
    d.arrow([(mdl.x, mdl.y + 95), srv.e(95 / 166)], "refused, or a cut count", style="error")
    d.arrow([srv.n(0.3), ctx.s(0.3)], "trimmed, resent", style="back", dx=60)
    d.legend(30, 740, {"store", "ui", "agent", "guard", "retrieval", "external"}, cols=1)
    return d


def local_documents():
    d = Diagram("local-documents", "Local documents: which files are read, and how",
                f"files.py decides which files; zetokrag/parse.py turns one into text; the index, the tools and the "
                f"extracted-text store share both. {AS_OF}.", 1500, 1370)
    d.group(20, 92, 480, 930, "files.py: which files")
    d.group(520, 92, 500, 930, "zetokrag/parse.py: the document converter")
    d.group(1040, 92, 440, 930, "Who reads the result")
    roots = d.box("roots", 40, 130, 440, 78, "store", "Declared folders", "guard.roots() · config.json", [
        "the outermost only: a nested folder walked once"])
    folders = d.box("folders", 40, 248, 440, 200, "retrieval", "Each folder, in this order", "files.walk", [
        "1 protected (guard.DENY): counted, never named", "2 a symlinked folder: not followed",
        "3 peresearch's own data ($PERESEARCH_HOME)", "4 the user's patterns: config.json exclude/include,",
        "   .peresearchignore (the last match wins)", "5 made by a tool: pyvenv.cfg, conda-meta/,",
        "   CACHEDIR.TAG, .git, node_modules, *.dist-info…", "6 a hidden name: left out unless included"])
    large = d.box("large", 40, 488, 440, 96, "decision", "More than 500 files side by side?", None, [
        "yes → its files left out (likely a dataset),", "unless the folder is declared or included"])
    each = d.box("each", 40, 624, 440, 130, "retrieval", "Each file", "files.walk", [
        "protected, or a link leading outside: counted", "then the user's patterns, a hidden name,",
        "an Office owner file (~$…) → tool_made", "a type parse does not read → 'unsupported'"], mark=14)
    left = d.box("left", 40, 776, 440, 130, "store", "Left out, with the reason", "index table `skipped`", [
        "tool_made · hidden · user_excluded · unreadable", "large_folder · symlinked_folder · unsupported",
        "a folder once, not each file in it", "Index.why_not(path) answers from it"])
    kind = d.box("kind", 540, 130, 460, 112, "decision", "parse.kind(extension)", None, [
        ".md .txt .rst .tex .bib and code → lines", ".pdf .pptx → pages · .ipynb → cells",
        ".docx .xlsx .html → Markdown lines · else unsupported"])
    text = d.box("text", 540, 266, 460, 180, "retrieval", "A text file", "parse.decode", [
        "over 20 MiB → too_large · a NUL byte → binary", "a byte order mark → UTF-8, 16 or 32",
        "valid UTF-8 → as it is", "else the likeliest legacy encoding (charset-", "normalizer), kept if its language is recognized",
        "or ≥ 64 bytes without rare characters; NFC", "else UTF-8 with replacement; the note says which"], mark=12)
    pdf = d.box("pdf", 540, 486, 460, 164, "retrieval", "A PDF", "parse._pdf (PyMuPDF)", [
        "no text on any page → needs_ocr (no OCR yet)", "PyMuPDF's own order: columns read one by one",
        "lines in the top or bottom 7.5% found on half", "the pages, digits aside → dropped (heads, numbers)",
        "a glyph without a character → U+FFFD", "limit: 20 M characters of text, not bytes"], mark=13)
    nb = d.box("nb", 540, 690, 460, 96, "retrieval", "A notebook", "parse._notebook", [
        "markdown and code cells, numbered", "a code cell's text outputs, ≤ 2,000 characters"])
    doc = d.box("doc", 540, 826, 460, 112, "end", "Doc(status, blocks, note)", None, [
        "a block: text, unit (line, page, cell), start", "lines end at \\n only (parse.lines), as editors,",
        "git and ripgrep number them"])
    idx = d.box("idx", 1060, 130, 400, 78, "retrieval", "Index.update", "zetokrag/index.py", [
        "chunks and vectors: zetokrag-index.svg"])
    store = d.box("store", 1060, 248, 400, 164, "store", "Extracted-text store", "text/<sha>.txt + table texts", [
        "PDFs, notebooks, text not in UTF-8, as UTF-8", "one file per content, copies share it; 0600",
        "each block from a new line; units map a line", "back to its page, cell or line (locate)",
        "every update: missing written, the rest removed"])
    tools = d.box("tools", 1060, 452, 400, 112, "agent", "Tools: read, grep, glob, tree", "tools.py", [
        "read: parse.read, same text and lines", "grep: ripgrep over the files and the store,",
        "the converter for what the index has not seen"], mark=8)
    find = d.box("find", 1060, 604, 400, 96, "retrieval", "Search's freshness check", "Searcher.find", [
        "a file changed since indexing is decoded", "as the index decoded it"])
    d.box("level", 1060, 740, 400, 96, "ok", "Checked on a clean machine", None, [
        "Docker, CPU only, not root, POSIX locale", "(level B, Linux only)"], mark=16)
    d.arrow([roots.s(), folders.n()])
    d.arrow([folders.s(), large.n()])
    d.arrow([large.s(), each.n()], "no", dx=16)
    d.arrow([each.s(), left.n()], "left out", dx=34)
    d.arrow([large.e(), (490, large.y + 48), (490, left.y + 40), left.e(0.3)], "yes", style="back", seg=1, dx=-14)
    d.arrow([each.e(0.3), (510, each.y + 34), (510, kind.y + 48), kind.w_()])
    d.arrow([kind.s(0.2), (kind.x + 92, text.y)])
    d.arrow([text.s(0.2), (text.x + 92, pdf.y)])
    d.arrow([pdf.s(0.2), (pdf.x + 92, nb.y)])
    d.arrow([nb.s(0.2), (nb.x + 92, doc.y)])
    d.arrow([doc.e(0.2), (1030, doc.y + 22), (1030, idx.y + 39), idx.w_()], seg=1)
    d.arrow([(1030, store.y + 82), store.w_()])
    d.arrow([(1030, tools.y + 56), tools.w_()])
    d.arrow([(1030, find.y + 48), find.w_()])
    d.text(552, 976, "The extension picks one of the three; each ends in a Doc.", 12, "sans", "#5A5A55")
    d.legend(40, 1065, {"retrieval", "agent", "store", "decision"}, cols=2, colw=230)
    d.footnotes(540, 1065, 940)
    return d


def tools_diagram():
    d = Diagram("tools", "Tools: grep, glob, tree, search_notes and overview",
                f"tools.py: what the model calls to look through the user's files. Every result is text with source ids. "
                f"{AS_OF}.", 1500, 1400)
    call = d.box("call", 470, 100, 460, 78, "agent", "grep(pattern, glob)", "tools.py: Toolbox.grep",
                 ["the pattern checked with Python's re first"])
    scope = d.box("scope", 470, 218, 460, 96, "guard", "Where to look", "Toolbox._scope", [
        "declared folders, and folders allowed this session,", "or the folder a ~ or absolute pattern names"])
    walk = d.box("walk", 470, 354, 460, 130, "retrieval", "Walk", "tools._walk → files.walk", [
        "the index's rules (local-documents.svg)", "a folder named in the pattern is searched,",
        "unless the user excluded it; a file once", "left out → counted by reason"])
    plan = d.box("plan", 470, 524, 460, 164, "decision", "Each file: how to read it?", "Toolbox._matches", [
        "a type parse does not read → skipped", "text over 20 MiB → counted, not searched",
        "current hash matches converted text → store", "PDF, notebook → the converter",
        "else valid UTF-8 → ripgrep, other text → converter"])
    rg = d.box("rg", 40, 740, 400, 130, "retrieval", "ripgrep", "tools._rg", [
        "--json --ignore-case --crlf --engine auto", "400 paths a call, every matching line", "a NUL byte → the file dropped (binary)"])
    store = d.box("store", 470, 740, 460, 130, "store", "Extracted-text store", "text/<sha>.txt (by ripgrep)", [
        "a store line → its page, cell or line", "(extracted.locate); copies share the text",
        "written by the index update"])
    conv = d.box("conv", 960, 740, 500, 130, "retrieval", "Converter", "parse.read + Python re", [
        "what the index has not seen as it is;", "everything when ripgrep is missing or refuses a",
        "pattern Python takes"])
    share = d.box("share", 470, 920, 460, 96, "agent", "In turns", None, [
        "every match, whole: each file's first in walk order,", "then each file's second…; redacted"])
    notes = d.box("notes", 470, 1056, 460, 148, "end", "Notes for the model", None, [
        "the whole result: the agent shows what fits", "files over 20 MiB not searched",
        "files the user may not read: counted", "stopped after 60 s", "not searched: counts by reason"])
    d.arrow([call.s(), scope.n()])
    d.arrow([scope.s(), walk.n()])
    d.arrow([walk.s(), plan.n()])
    d.arrow([plan.s(0.1), (plan.x + 46, 714), (rg.x + 200, 714), rg.n()])
    d.arrow([plan.s(), store.n()])
    d.arrow([plan.s(0.9), (plan.x + 414, 714), (conv.x + 250, 714), conv.n()])
    d.arrow([rg.s(), (rg.x + 200, 895), (share.x + 46, 895), share.n(0.1)])
    d.arrow([store.s(), share.n()])
    d.arrow([conv.s(), (conv.x + 250, 895), (share.x + 414, 895), share.n(0.9)])
    d.arrow([share.s(), notes.n()])
    d.box("sn", 40, 100, 400, 166, "agent", "search_notes(query)", "Toolbox.search_notes", [
        "Searcher.find: zetokrag-search.svg", "every candidate, best first: (before) the hit", "(after), each whole",
        "not enough → what the index does not hold,", "counted by reason (Index.coverage)"])
    d.box("tree", 40, 306, 400, 112, "agent", "tree(path, depth)", "Toolbox.tree", [
        "every entry; ≤ 4 levels (bounds the walk's time)", "a folder grep would skip: named with its reason",
        "(files.check), not opened; hidden, protected: not shown"])
    d.box("glob", 40, 458, 400, 96, "agent", "glob(pattern)", "Toolbox.glob", [
        "the same walk; every file listed;", "what was not listed, counted by reason"])
    d.box("ov", 40, 590, 400, 112, "agent", "overview(query, folder)", "Toolbox.overview", [
        "Searcher.documents: every file once, by its best", "passage (whole, one line); headings, first line;",
        "no query: the folder's catalog by path"])
    d.box("measured", 960, 100, 500, 130, "ok", "Measured on the owner's folders", None, [
        "13,402 files: ripgrep 2.2 s, converter only 6.1 s,", "the same output (old Python grep: 43.8 s);",
        "walking takes 1.1–1.4 s of it"], mark=8)
    d.box("levelb", 960, 270, 500, 96, "ok", "Checked on a clean machine", None, [
        "Docker: rg from pixi, none on the system;", "every test passed (Linux only)"], mark=17)
    d.legend(40, 1250, {"agent", "retrieval", "store", "decision", "guard"}, cols=2, colw=230)
    d.footnotes(540, 1250, 920)
    return d


def web_diagram():
    d = Diagram("web", "Web: searching pages and papers, reading them",
                f"web.py and papers.py, called by tools.web_search, tools.search_papers and tools.fetch. {AS_OF}.",
                1500, 1310)
    q = d.box("q", 40, 100, 420, 96, "agent", "web_search(query, n)", "tools.py", [
        "one search budget with papers and Wikipedia", "(a limit only when set); outbound_query"])
    qp = d.box("qp", 520, 100, 460, 96, "agent", "search_papers(query, n)", "tools.py → papers.Papers", [
        "counts as a web search", "query through guard.outbound_query"])
    prov = d.box("prov", 40, 236, 420, 130, "web", "The user's search MCP server", "web.McpSearch · mcp.Server", [
        "a command (stdio) or a URL (Streamable HTTP);", "its search tool called through its own schema;",
        "links read from its text; none set → said", "retry completes initialize + initialized first"])
    src = d.box("src", 520, 236, 460, 148, "web", "Every scholarly source, merged", "Papers.search", [
        "OpenAlex (with a key) · Semantic Scholar · arXiv", "one from each in turn; a paper once (DOI, arXiv",
        "id or title); arXiv ≥ 3 s apart; a source that", "fails is named in the result"], mark=18)
    lab = d.box("lab", 40, 406, 420, 96, "retrieval", "Its text, each link a source", "tools.web_search", [
        "[W#] per link (its paragraph), the kind of site:", "university, preprint server, blog… (never a filter)"])
    plab = d.box("plab", 520, 424, 460, 112, "retrieval", "Each paper says what it is", "tools.search_papers", [
        "RETRACTED · journal article / conference paper /", "preprint (published as …) · venue · citations",
        "authors · DOI page · open PDF (fetch may read it)"])
    f = d.box("f", 280, 576, 460, 96, "agent", "fetch(url)", "tools.py → Web.fetch", [
        "a URL a search returned or the user wrote, whole;", "pages limited when set; a page read again free"])
    pub = d.box("pub", 280, 712, 460, 130, "guard", "Public address? Connect to it", "web._checked · web._pinned", [
        "localhost, 10.x, 192.168.x, 169.254.x, ::1, .local:", "refused; the name resolved once, the request",
        "sent to that address (Host and TLS name kept);", "again at every redirect (≤ 5)"])
    get = d.box("get", 280, 882, 460, 112, "web", "GET, streamed", "Web._get", [
        "a page ≤ 5 MiB (cut), a PDF ≤ 64 MiB (else refused)", "the PDF limit chosen on the first bytes",
        "(type, .pdf, or %PDF-)"])
    ext = d.box("ext", 280, 1034, 460, 148, "retrieval", "extract", "web.extract", [
        "PDF → the document converter: [p. N] pages,", "running heads dropped",
        "HTML → trafilatura, in the declared or guessed", "encoding; other text as is",
        "not text (image, archive, NUL) → not read"])
    jina = d.box("jina", 800, 1034, 400, 96, "external", "Jina Reader", "r.jina.ai", [
        "only with the user's JINA_API_KEY, when a page", "yields nothing (25 of 30 read locally)"])
    out = d.box("out", 280, 1222, 460, 56, "end", "Page text, untrusted, cached (UTF-8, 0600)")
    for a_, b_ in ((q, prov), (prov, lab), (qp, src), (src, plab), (f, pub), (pub, get), (get, ext), (ext, out)):
        d.arrow([a_.s(), b_.n()])
    d.arrow([lab.s(), (lab.x + 210, 556), (f.x + 115, 556), f.n(0.25)], style="back")
    d.arrow([plab.s(), (plab.x + 230, 556), (f.x + 345, 556), f.n(0.75)], style="back")
    d.arrow([ext.e(0.3), (jina.x, ext.y + 44)], "empty")
    d.arrow([pub.w_(), (240, pub.y + 65), (240, out.y + 28), out.w_()], "refused → \"\"", style="error", seg=1, dx=-46)
    d.box("ext2", 1040, 100, 420, 130, "external", "Search services", "the user's MCP server · OpenAlex", [
        "Semantic Scholar · arXiv · Wikipedia: the query", "leaves the machine (secrets refused first)"])
    d.box("wk", 1040, 250, 420, 150, "web", "search_wikipedia(query, languages)", "tools.py → wiki.Wikipedia", [
        "≤ 3 editions (codes checked); lead summary,", "address, the title in en vi zh ja ko hi;",
        "an encyclopedia: a starting point, not", "evidence; counts as a web search"])
    d.arrow([qp.e(), (1040, qp.y + 48)], style="exit")
    d.legend(1040, 430, {"agent", "web", "guard", "retrieval", "external"})
    d.footnotes(1040, 690, 420)
    return d


def zetokrag_index():
    d = Diagram("zetokrag-index", "ZetokRAG: building the index",
                f"zetokrag/index.py: Index.update. Run when the TUI opens (folders declared), on /index, /add, /remove, "
                f"by `peresearch index` and `setup`. {AS_OF}.", 1500, 2170)
    st = d.box("st", 470, 100, 460, 78, "retrieval", "Index.update(roots)", "zetokrag/index.py", [
        "_write_lock: one update at a time, own writer connection"])
    pro = d.box("pro", 470, 218, 460, 130, "decision", "First commit: what the index was built with", None, [
        "another embedder → its vectors deleted (chunks kept)", "another chunking or context → every file marked",
        "to be read again; old chunks kept, searchable", "meta: embedder, chunking, context written"])
    walk = d.box("walk", 470, 388, 460, 130, "retrieval", "Walk and hash (no transaction)", "Index._files → files.walk", [
        "guard.outermost: a nested root walked once", "rules of local-documents.svg; left out → `skipped`",
        "hash current bytes even with unchanged metadata"])
    dup = d.box("dup", 470, 558, 460, 78, "decision", "Same sha256 as an earlier path?", None, [
        "yes → status 'duplicate', its chunks dropped"])
    unch = d.box("unch", 470, 676, 460, 96, "decision", "sha256 unchanged (and verdict kept)?", None, [
        "yes → 'unchanged'; if its declared folder changed,", "its place is updated and embedded again"])
    parse_ = d.box("parse", 470, 812, 460, 112, "retrieval", "parse.read(path)", "zetokrag/parse.py", [
        "md, text, code → lines · PDF → pages · ipynb → cells", "too_large: text > 20 MiB, PDF > 20 M characters",
        "binary · needs_ocr · error → reported, no chunks"])
    data = d.box("data", 470, 964, 460, 112, "decision", "is_data(doc): datafilter.reason", "zetokrag/datafilter.py", [
        "plain text: URLs/numbers, JSON, table, log, word list", "code: generated or minified; .sql: a dump",
        "yes → status 'data' + reason: not chunked"])
    chunk = d.box("chunk", 470, 1116, 460, 112, "retrieval", "chunk.chunk(doc, keep = no secret, whole)",
                  "zetokrag/chunk.py: DEFAULT blocks+code-defs", ["tables, code blocks, lists kept whole (≤ 6,000)",
                                                                  ".py cut at top-level definitions; else ≤ 1,500",
                                                                  "checked: each chunk verbatim at its lines"])
    write = d.box("write", 470, 1268, 460, 78, "store", "Replace the file's chunks and row", None, [
        "chunks carry their place (declared folder + path)"])
    emb = d.box("emb", 470, 1386, 460, 112, "retrieval", "Index.embed, after every file is committed",
                "zetokrag/embed.py: Embedder (the user's endpoint)", [
        "64 chunks a request, each batch committed, fp16;", "stoppable, resumable; endpoint down → said, words",
        "meanwhile; a text too long → embedded by its head"], mark=5)
    com = d.box("com", 470, 1538, 460, 112, "store", "60 s since the last commit? → COMMIT", None, [
        "checked: no vector without its chunk;", "extracted-text store synced (_sync_texts)",
        "stopped → this batch rolled back, the rest kept"])
    gone = d.box("gone", 470, 1690, 460, 96, "store", "After the last file (last commit)", None, [
        "files not found dropped; `skipped` rewritten;", "meta.datafilter written (all files judged)"])
    texts = d.box("texts", 470, 1826, 460, 96, "store", "_sync_texts + COMMIT", "text/<sha>.txt + table texts", [
        "the extracted-text store made to match the files:", "missing texts written, anything else removed"])
    rep = d.box("rep", 470, 1962, 460, 78, "end", "Report + guard.audit('index', counts)", None, [
        "the largest files (≥ 100 chunks) named"])
    for a, b in ((st, pro), (pro, walk), (walk, dup), (dup, unch), (unch, parse_), (parse_, data), (data, chunk),
                 (chunk, write), (write, emb), (emb, com), (com, gone), (gone, texts), (texts, rep)):
        d.arrow([a.s(), b.n()])
    rail = 445
    d.arrow([dup.w_(), (rail, dup.y + 39)], style="back")
    d.arrow([unch.w_(), (rail, unch.y + 48)], style="back")
    d.arrow([data.w_(), (rail, data.y + 56)], style="back")
    d.arrow([com.w_(), (rail, com.y + 56), (rail, dup.y + 39), dup.w_()], "next file", style="back", seg=1, at=0.5,
            dx=-40)
    d.box("schema", 990, 218, 480, 236, "store", "index.sqlite", "PRAGMA user_version = 2, WAL", [
        "files(path, size, mtime, sha, status, note,", "      withheld, reason)",
        "chunks(id, path, section, unit, start, end,", "       text, sha, place)   — ids never reused",
        "vectors(id = chunks.id, vec fp16 BLOB)", "meta(embedder, chunking, context, secrets, datafilter)",
        "skipped(path, reason, note) — rewritten each update", "texts(sha, units) — the extracted-text store"])
    d.box("status", 990, 494, 480, 130, "store", "files.status", None, [
        "ok · duplicate · data (reason) · too_large", "binary · needs_ocr · error",
        "older layout (user_version < 2): emptied,", "rebuilt by the next update"])
    d.box("readers", 990, 664, 480, 112, "retrieval", "Readers", None, [
        "see only committed data (WAL, separate connection):", "during a long update, each committed file;",
        "Searcher's snapshot: zetokrag-search.svg"])
    d.box("conn", 990, 816, 480, 96, "store", "_connect", None, [
        "lock wait 30 s set first; WAL switch retried ≤ 30 s,", "then a clear error (a drive without WAL)"])
    d.box("report", 990, 952, 480, 130, "end", "What the report names", None, [
        "added, changed, unchanged, removed, denied, largest",
        "problems: duplicate, data_file, too_large, binary,", "needs_ocr, error; withheld; tool_made, hidden,",
        "user_excluded, large_folder, unreadable, special_file,", "symlinked_folder, unsupported, data_type"])
    d.legend(1000, 1130, {"retrieval", "store", "decision", "end"})
    d.footnotes(990, 1400, 480)
    return d


def zetokrag_search():
    d = Diagram("zetokrag-search", "ZetokRAG: answering search_notes",
                f"zetokrag/search.py: Searcher.find, called by tools.search_notes. No step calls an LLM, and no score "
                f"decides for the model. {AS_OF}.", 1500, 1640)
    tool = d.box("tool", 470, 100, 460, 62, "agent", "search_notes(query)", "tools.py")
    find = d.box("find", 470, 202, 460, 62, "retrieval", "Searcher.find(query, k = None)", "zetokrag/search.py")
    snap = d.box("snap", 470, 304, 460, 130, "retrieval", "Index.snapshot()", "zetokrag/index.py", [
        "PRAGMA data_version changed → rebuild in one read:", "ids, BM25 over place + section + text,",
        "vectors fp16 → float32; live scope before fusion"])
    bm = d.box("bm", 160, 474, 440, 112, "retrieval", "BM25.scores(query)", "zetokrag/core.py: BM25", [
        "words as typed, with combining marks", "Han, Kana, Thai… cut into character pairs",
        "k1 = 1.5, b = 0.75 (16 settings: within 0.006)"])
    dense = d.box("dense", 800, 474, 440, 112, "retrieval", "Dense similarity", "embed.Embedder.queries", [
        "the user's endpoint, the model card's query prefix;", "cosine = vectors · q; no vector yet → the lowest;",
        "none, or the endpoint down → words alone, said"])
    fuse = d.box("fuse", 470, 626, 460, 112, "retrieval", "fuse_minmax(ρ = 0.5, top 100 each)", "core.fuse_minmax", [
        "min-max per view, then ρ·BM25 + (1 − ρ)·dense", "a chunk missing from a view gets 0 there",
        "BM25 best ≤ 0 → BM25 left out"])
    rr = d.box("rr", 470, 778, 460, 130, "retrieval", "Rerank the first `depth` of 30", "embed.Reranker", [
        "the user's /rerank on place + section + text;", "score in [0, 1] (a raw logit → logistic);",
        "depth: setting 0–30 (30); the rest, fused order;", "none set or down → fused order, no score, said"])
    fold = d.box("fold", 470, 948, 460, 96, "retrieval", "Searcher.candidates: fold near-copies", None, [
        "8-character shingles, Jaccard ≥ 0.8, any script:", "a copy joins the better-ranked one (also in …)"])
    page = d.box("page", 470, 1084, 460, 78, "retrieval", "Every candidate, best first", None, [
        "no threshold, no verdict (`peresearch find`: pages of k)"])
    stale = d.box("stale", 470, 1202, 460, 112, "guard", "Re-read the source file", None, [
        "gone or ambiguous → dropped; copies checked too", "changed text → current lines and warning",
        "old neighbors omitted; unique ordered span only"])
    nb = d.box("nb", 470, 1338, 460, 62, "retrieval", "Neighbors", None, ["unchanged file: id ± 1, same section"])
    res = d.box("res", 470, 1440, 460, 166, "agent", "Result for the model", "tools.search_notes", [
        "'n ranked candidates, best first (relevance…)'", "[N#] span (relevance) § section, (before), the",
        "hit, (after), each whole, (also in …); redacted;", "what the index does not hold, counted by reason;",
        "the agent shows what fits (harness frame)"])
    d.arrow([tool.s(), find.n()])
    d.arrow([find.s(), snap.n()])
    d.arrow([snap.s(0.25), (snap.x + 115, 455), (bm.x + 220, 455), bm.n()])
    d.arrow([snap.s(0.75), (snap.x + 345, 455), (dense.x + 220, 455), dense.n()])
    d.arrow([bm.s(), (bm.x + 220, 606), (fuse.x + 115, 606), fuse.n(0.25)])
    d.arrow([dense.s(), (dense.x + 220, 606), (fuse.x + 345, 606), fuse.n(0.75)])
    for a, b in ((fuse, rr), (rr, fold), (fold, page), (page, stale), (stale, nb), (nb, res)):
        d.arrow([a.s(), b.n()])
    docs = d.box("docs", 30, 778, 410, 166, "retrieval", "Searcher.documents: overview", "zetokrag/search.py", [
        "the same fusion, inside a folder when asked", "(kept before fusing); each file once, by its",
        "best chunk; the first `depth` reranked on it;", "overview: every file, headings, first line",
        "(gold: +0.11 relevant files in 20 lines)"])
    d.arrow([fuse.w_(), (docs.x + 205, fuse.y + 56), docs.n()], seg=0)
    d.box("locks", 990, 202, 480, 96, "store", "Locks", "Index._lock (RLock)", [
        "snapshot and every read on the one reading", "connection, one thread at a time"])
    d.box("lazy", 990, 778, 480, 96, "retrieval", "No model runs in peresearch", "zetokrag/embed.py", [
        "the endpoints from the settings (none by default)", "zetokrag/models.py: the benchmarks' test tools"])
    d.box("why", 990, 948, 480, 150, "decision", "Why no verdict (gold dev, 2026-10-04)", None, [
        "best score: answerable vs not, AUROC 0.56", "[0.37, 0.75]; the old cut (top 5, 0.25) showed",
        "0.60 of the answering passages, the top 10 0.71,", "the top 30 0.78"])
    d.legend(30, 1130, {"agent", "retrieval", "guard", "store", "decision"})
    d.footnotes(990, 1180, 480)
    return d


def guard_paths():
    d = Diagram("guard", "Guard: what may be read, and what may leave",
                f"guard.py with tools.py (Toolbox._path, _outside, _scope). Refusals reach the model as text. {AS_OF}.",
                1500, 1330)
    d.group(20, 92, 870, 1210, "Reading a path the model names")
    d.group(910, 92, 570, 1000, "Leaving the machine, and redaction")
    g0 = d.box("g0", 200, 130, 480, 78, "agent", "A path from the model", "Toolbox._path", [
        "read, tree; grep and glob use _scope (same rules)"])
    g1 = d.box("g1", 200, 248, 480, 78, "decision", "Relative path?", None, [
        "→ the first declared or granted folder holding it"])
    g2 = d.box("g2", 200, 366, 480, 96, "decision", "guard.allowed: inside a declared folder?", None, [
        "real path (symlinks resolved) inside a root,", "not denylisted or inside application storage"])
    g3 = d.box("g3", 200, 502, 480, 56, "decision", "real path outside the home folder?")
    g4 = d.box("g4", 200, 598, 480, 78, "decision", "hidden part, denylist, or own data?", None, [
        "own data = $PERESEARCH_HOME"])
    g5 = d.box("g5", 200, 716, 480, 56, "decision", "does not exist?")
    g6 = d.box("g6", 200, 812, 480, 56, "decision", "inside a folder granted this session?")
    g7 = d.box("g7", 200, 908, 480, 56, "decision", "nobody to ask (not interactive)?")
    g8 = d.box("g8", 200, 1004, 480, 96, "ui", "Ask the user", "TUI dialog · cli.ask_in_terminal", [
        "once → read · session → folder granted, read", "no → refused: 'do not ask for it again'"])
    rd = d.box("rd", 40, 366, 130, 734, "ok", "READ")
    rf = d.box("rf", 720, 248, 150, 852, "guard", "REFUSED")
    d.box("walk", 200, 1150, 480, 112, "guard", "Walking folders", "files.walk: index, tree, grep, glob", [
        "protected and peresearch's own: never; links to", "a hidden target skipped in allowed folders;",
        "rules: local-documents.svg"])
    for a, b in ((g0, g1), (g1, g2), (g2, g3), (g3, g4), (g4, g5), (g5, g6), (g6, g7), (g7, g8)):
        d.arrow([a.s(), b.n()], "no" if a not in (g0, g1) else None, dx=16)
    d.arrow([g2.w_(), (rd.x + rd.w, g2.y + 48)], "yes")
    d.arrow([g6.w_(), (rd.x + rd.w, g6.y + 28)], "yes")
    d.arrow([g8.w_(), (rd.x + rd.w, g8.y + 48)])
    d.arrow([g1.e(), (rf.x, g1.y + 39)], "none found", style="error")
    for g in (g3, g4, g5, g7):
        d.arrow([g.e(), (rf.x, g.y + g.h / 2)], "yes", style="error")
    d.arrow([g8.e(), (rf.x, g8.y + 48)], "no", style="error")
    d.text(rf.x + 12, rf.y + 50, "returned to the", 12, "sans", "#7D1F1F")
    d.text(rf.x + 12, rf.y + 67, "model as text", 12, "sans", "#7D1F1F")
    d.box("xm", 930, 130, 530, 130, "guard", "To the model", "Agent._filter_access · llm._outbound", [
        "live evidence scope checked before requests;", "secrets redacted by own + gitleaks rules;",
        "revocation cancels retries; no secret remains", "audit: channel, characters, count; no content"])
    d.box("xw", 930, 282, 530, 96, "guard", "To the web", "guard.outbound_query", [
        "a query or URL holding a secret → refused", "fetch opens only URLs a search returned"])
    d.box("xi", 930, 418, 530, 96, "guard", "Into the index", "Index._update: keep=", [
        "a paragraph with a secret is never stored; new rules", "(guard.fingerprint) → every file judged again"])
    d.box("xt", 930, 554, 530, 78, "guard", "Into tool results", "guard.redact", [
        "grep lines, read text and excerpts redacted"])
    d.box("xs", 930, 672, 530, 78, "guard", "To the terminal", "guard.sanitize", [
        "ANSI escapes and control characters removed"])
    d.box("xa", 930, 790, 530, 78, "store", "audit.jsonl", "guard.audit", [
        "time, event, channel, counts; never the text"])
    d.box("xd", 930, 908, 530, 170, "guard", "Never readable", "guard.DENY", [
        ".ssh .gnupg .aws .azure .kube .docker .kaggle", ".modal.toml .netrc .git-credentials .pypirc .npmrc",
        ".env .env.* *.env *.pem *.key *.p12 *.pfx *.kdbx", "id_rsa* id_ecdsa* id_ed25519* .password-store",
        "token *.token token.json credentials secrets.json …", "any part of the path; tokenizer.md stays readable"])
    d.footnotes(930, 1150, 540)
    return d


def threads():
    d = Diagram("threads", "Threads in the TUI",
                f"tui.py, agent.py, llm.py, zetokrag/index.py, cli.py. Who runs where, how they wait on each other, and "
                f"how quitting stops them. {AS_OF}.", 1500, 1230)
    d.group(20, 92, 360, 840, "Textual main thread")
    d.group(400, 92, 360, 840, "Thread 'agent' (daemon)")
    d.group(780, 92, 340, 840, "Model reader thread (daemon)")
    d.group(1140, 92, 340, 300, "CLI process (no TUI)")
    d.group(1140, 412, 340, 520, "Thread 'index' (daemon)")
    sub = d.box("sub", 40, 130, 320, 78, "ui", "Prompt submitted", "App.on_prompt_submitted", ["→ run_agent(question)"])
    ev = d.box("ev", 40, 260, 320, 78, "ui", "on_agent_event", None, ["status line, spinner, tokens, answer"])
    dlg = d.box("dlg", 40, 390, 320, 96, "ui", "Permission dialog", "App.permission → show()", [
        "once · session · no; Esc refuses and stops"])
    esc = d.box("esc", 40, 532, 320, 78, "ui", "Esc", "App.action_escape", ["agent.cancel(), if a question runs"])
    chk = d.box("chk", 40, 660, 320, 96, "ui", "Folder check", "App.reindex", [
        "on open, /index, /add, /remove;", "refused while a question runs"])
    quit_ = d.box("quit_", 40, 812, 320, 96, "ui", "Ctrl+Q, /exit", "App.action_quit", [
        "stopping.set(), agent.cancel();", "waits ≤ QUIT_GRACE 2 s, then exits"])
    ask = d.box("ask", 420, 130, 320, 96, "agent", "Agent.ask(question)", "App.run_agent", [
        "agent built on the first question", "events through _ui (none once quitting)"])
    wait = d.box("wait", 420, 390, 240, 96, "agent", "Toolbox.ask = permission", None, [
        "waits on a threading.Event", "(0.1 s polls); 'no' on quit"])
    flag = d.box("flag", 420, 532, 240, 78, "agent", "Agent.cancelled", None, [
        "read by the loop each round", "and by the reader's stop()"])
    srch = d.box("srch", 420, 650, 320, 78, "retrieval", "Searcher.find", None, ["Index.read / snapshot, under the RLock"])
    rd = d.box("rd", 800, 260, 300, 130, "llm", "_read_stream", "LLM._create helper", [
        "reads chunks, checks stop()", "closes the stream at the next chunk", "the caller is freed within 0.1 s"])
    up = d.box("up", 1160, 520, 300, 150, "retrieval", "Index.update(stop)", "Workspace.update", [
        "shared lock across embedding changes", "a question asked meanwhile runs", "stop() checked before each file:",
        "commits what is done, raises Stopped"])
    cli_ = d.box("cli_", 1160, 130, 300, 130, "ui", "peresearch index, setup", "cli.main → cli._update", [
        "Ctrl+C: SIGINT, then once more", "from pixi: the first stops, the rest", "are ignored; exit status 130"])
    db = d.box("db", 400, 980, 1080, 96, "store", "index.sqlite (WAL)", "zetokrag/index.py", [
        "reading connection `db` + RLock for searches · writer connection for updates · a search sees only committed",
        "data · the snapshot is rebuilt when PRAGMA data_version changes · an unfinished batch rolls back"])
    d.arrow([sub.e(), (ask.x, sub.y + 39)], "thread")
    d.arrow([ask.s(0.08), (ask.x + 25.6, ev.y + 39), ev.e()], "events", style="back", seg=1)
    d.arrow([ask.s(0.35), (ask.x + 112, wait.y)], "outside path", dx=-2)
    d.arrow([wait.w_(0.35), (dlg.x + dlg.w, wait.y + wait.h * 0.35)], "show", dy=-6)
    d.arrow([(dlg.x + dlg.w, wait.y + wait.h * 0.75), wait.w_(0.75)], "answer", style="back", dy=-6)
    d.arrow([esc.e(), flag.w_()], "set")
    d.arrow([flag.e(), (770, flag.y + 39), (770, rd.y + 90), (rd.x, rd.y + 90)], "stop()", seg=1, at=0.3)
    d.arrow([ask.e(0.5), (790, ask.y + 48), (790, rd.y + 30), (rd.x, rd.y + 30)], "model call", seg=0, at=0.5)
    d.arrow([ask.s(0.85), (ask.x + 272, srch.y)], "search_notes", dx=8, at=0.62)
    d.arrow([chk.s(), (chk.x + 160, 786), (1130, 786), (1130, up.y + 50), (up.x, up.y + 50)], "thread", seg=1, at=0.6)
    d.arrow([quit_.e(0.3), (392, quit_.y + 29), (392, flag.y + 62), flag.w_(0.8)], "cancel()", seg=1, at=0.12, dx=-34)
    d.arrow([quit_.e(0.7), (1148, quit_.y + 67), (1148, up.y + 120), (up.x, up.y + 120)], "stopping.is_set",
            seg=0, at=0.55)
    d.arrow([cli_.s(0.8), (cli_.x + 240, up.y)], "update()", dx=30)
    d.arrow([srch.s(), (srch.x + 160, db.y)], "read", dx=24)
    d.arrow([up.s(), (up.x + 150, db.y)], "write", dx=24)
    d.legend(30, 1000, {"ui", "agent", "llm", "retrieval", "store"})
    return d


def serving():
    d = Diagram("serving", "Test tools: serving a model for the author's measurements (gate 3)",
                f"deploy/, not part of peresearch (any provider the gateway speaks to is). The laptop runs a small model of the "
                f"same family with the Modal flags; Docker rebuilds the Modal image before any GPU time is paid for. "
                f"{AS_OF}.", 1500, 1010)
    d.group(20, 92, 440, 760, "Laptop: pixi run chat-local")
    d.group(480, 92, 480, 760, "Laptop: pixi run replica (Docker)")
    d.group(980, 92, 500, 760, "Modal: pixi run modal-deploy")
    cl = d.box("cl", 40, 130, 400, 130, "ui", "chat_local.py", "deploy/chat_local.py", [
        "starts the server, then peresearch chat", "stops it on quit, SIGHUP, SIGTERM",
        "pgid in runs/serve/local_vllm.pgid"])
    lv = d.box("lv", 40, 300, 400, 150, "llm", "local_vllm.py", "pixi env 'serve': vLLM 0.29.0", [
        "Qwen3.5-0.8B (pinned revision)", "FLAGS and ENV read from modal_vllm.py (ast)",
        "max-model-len 16384, GPU memory 0.80", "WSL: VLLM_WSL2_ENABLE_PIN_MEMORY=1"])
    ag = d.box("ag", 40, 490, 400, 96, "agent", "peresearch (TUI)", None, [
        "PERESEARCH_LLM_URL=http://localhost:8000/v1", "ZetokRAG models move to the CPU"])
    lg = d.box("lg", 40, 626, 400, 78, "store", "runs/serve/local_vllm.log", None, ["server log, kept"])
    sp = d.box("sp", 500, 130, 440, 130, "llm", "Image spec", "deploy/modal_vllm.py (literals)", [
        "BASE nvidia/cuda:13.0.3-devel-ubuntu22.04", "PYTHON 3.12 · PIP vllm==0.29.0",
        "ENV · FLAGS (qwen3 parsers, 32k, CUDA graphs)"])
    rp = d.box("rp", 500, 300, 440, 200, "llm", "replica.py", "docker build + run", [
        "environment: CUDA of nvcc = CUDA of torch", "download: HF with the image's settings",
        "serve: through vllm_server.py, as on Modal", "sampling: requests sent as the agent sends",
        "tool call: 1 of 3 requests calls the tool", "each check PASS or FAIL"])
    rl = d.box("rl", 500, 540, 440, 78, "store", "runs/replica/", None, ["vLLM's whole log, compiled kernels noted"])
    dep = d.box("dep", 1000, 130, 460, 96, "end", "pixi run modal-deploy", "pixi.toml: depends-on", [
        "test and replica must pass first"])
    dl = d.box("dl", 1000, 266, 460, 96, "llm", "download()", "CPU container", [
        "weights into Volume peresearch-hf-cache", "no GPU billed for the download"])
    sv = d.box("sv", 1000, 402, 460, 182, "llm", "Server", "modal_vllm.Server (one L40S)", [
        "Qwen3.6-35B-A3B-FP8, pinned revision", "max 1 container, scale to zero after 300 s",
        "startup timeout 1,200 s", "proxy auth (Modal's default) as the API key",
        "vllm_server.start: dies within seconds"], mark=9)
    cli_ = d.box("cli_", 1000, 624, 460, 112, "agent", "Client settings", "settings.env", [
        "PERESEARCH_LLM_URL = <deployed URL>/v1", "PERESEARCH_LLM_KEY = wk-….ws-… (proxy token)",
        "PERESEARCH_LLM_MODEL = llm"])
    d.arrow([cl.s(), lv.n()], "start, wait until ready", dx=80)
    d.arrow([lv.s(), ag.n()], "localhost:8000", dx=60)
    d.arrow([lv.e(0.85), (455, lv.y + 128), (455, lg.y + 39), lg.e()], style="back")
    d.arrow([sp.s(), rp.n()], "read with ast", dx=56)
    d.arrow([rp.s(), rl.n()])
    d.arrow([lv.e(0.25), (sp.x, lv.y + 37)], "FLAGS, ENV", style="back", dy=-6)
    d.arrow([dep.s(), dl.n()], "once", dx=30)
    d.arrow([dl.s(), sv.n()], "Volume", dx=34)
    d.arrow([cli_.n(), sv.s()], "HTTPS", style="exit", dx=30)
    d.arrow([sp.e(0.5), (dep.x - 0, sp.y + 65)], "same spec", style="back")
    d.legend(40, 880, {"ui", "llm", "agent", "store", "end"}, cols=3, colw=235)
    d.footnotes(500, 700, 460)
    return d


def main(argv):
    drawn = [f().save() for f in (overview, agent_loop, llm_client, model_frame, storage_frame, harness_frame, local_documents, tools_diagram, web_diagram,
                                  zetokrag_index, zetokrag_search, guard_paths, threads, serving)]
    if "--png" in argv:
        target = Path(argv[argv.index("--png") + 1])
        target.mkdir(parents=True, exist_ok=True)
        for path in drawn:
            with pymupdf.open(path) as doc:
                doc[0].get_pixmap(dpi=72).save(target / f"{path.stem}.png")
    print("\n".join(str(p) for p in drawn))


if __name__ == "__main__":
    main(sys.argv[1:])
