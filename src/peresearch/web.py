"""The web: the user's search server, and a page reader. Nothing here touches the personal index.

Search goes to the MCP server the user adds (PERESEARCH_SEARCH_MCP: Tavily's, Exa's, Brave's, a DuckDuckGo or SearXNG
one…; peresearch/mcp.py): its search tool is called through its own schema — no adapter per provider — and the text
it returns is given to the agent as it is, each link in it a source. None set: no web search, and the agent says so.
Every query and URL passes `guard.outbound_query`: one holding a secret is refused, not sent. A page is fetched with httpx — never from
this machine or a private network (localhost, 10.x, 169.254.x…), checked again at every redirect, connected to
the address that was checked (no second lookup for DNS rebinding to answer differently) — and reduced to
its main text on this machine: HTML by trafilatura, in the encoding the server or the page declares (else guessed
as for local files), a PDF by the document converter (running heads dropped, "[p. N]" before each page). A file
that is not text (an image, an archive) is not read. The Jina Reader (r.jina.ai), a third party, is asked only when
the user set JINA_API_KEY and a text page yields nothing (pages that need JavaScript): on 30 sample pages the local
reader read 25 and Jina 2 more (runs/e2e/page-reader-2026-10-08.json). Pages are cached as UTF-8 under $PERESEARCH_HOME/web, apart from the
index of personal files, and the text is untrusted data for the agent.
"""

import codecs
import hashlib
import ipaddress
import os
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

from peresearch import guard, settings
from peresearch.zetokrag import parse

MAX_PAGE_BYTES = 5 * 2**20
MAX_PDF_BYTES = 64 * 2**20        # papers run to tens of MB (Llama 2: 13.6 MB, the largest of 10 measured)
MAX_REDIRECTS = 5
_TEXT_TYPES = ("text/", "html", "xml", "json", "javascript")
_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?([A-Za-z0-9_.:-]+)""", re.I)
USER_AGENT = "peresearch/0.1 (+https://github.com/geminitt/peresearch)"


@dataclass
class Found:
    """What a web search returned: who answered, its text as it came, and the links in it (title, url)."""
    provider: str
    text: str
    links: list[tuple[str, str]]


class Unavailable(Exception):
    """No web search can answer: none is set, or its server failed."""


QUERY_NAMES = ("query", "q", "search_query", "searchquery", "search_term", "searchterm", "keywords", "terms", "term")
COUNT_NAMES = ("max_results", "maxresults", "num_results", "numresults", "count", "limit", "top_k", "topk", "k", "n",
               "results", "size")
_MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_URL = re.compile(r"https?://[^\s<>\"'()\[\]]+")
_LABEL = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s*)?(?:title\s*:\s*)?", re.I)


def links(text: str) -> list[tuple[str, str]]:
    """The links in a search server's text, in order, each once: a Markdown link's own title; a bare URL titled by
    the line before it when that line names it ("Title: …", "1. …"), else none."""
    out: dict[str, str] = {}
    for title, url in _MARKDOWN_LINK.findall(text):
        out.setdefault(url, title.strip())
    lines = text.split("\n")
    for i, line in enumerate(lines):
        for url in _URL.findall(line):
            url = url.rstrip(".,;:!?")
            if url in out:
                continue
            before = lines[i - 1].strip() if i and line.strip().lower().startswith(("url:", "http", "link:")) else ""
            out[url] = "" if not before or _URL.search(before) else _LABEL.sub("", before).strip()[:200]
    order = {u: text.find(u) for u in out}
    return [(out[u], u) for u in sorted(out, key=lambda u: order[u])]


class McpSearch:
    """Web search through the user's MCP server (peresearch.mcp.Server, or any object with tools/call/close)."""

    def __init__(self, server, tool: str | None = None):
        self.server, self.tool = server, tool or None

    @classmethod
    def from_settings(cls) -> "McpSearch | None":
        """PERESEARCH_SEARCH_MCP (a command line or a URL), PERESEARCH_SEARCH_TOOL (default: the tool named for a web
        search), PERESEARCH_SEARCH_MCP_KEY (NAME=value, put in that server's environment alone); None when unset."""
        from peresearch.mcp import Server

        target = settings.get("PERESEARCH_SEARCH_MCP")
        if not target:
            return None
        key, env = settings.get("PERESEARCH_SEARCH_MCP_KEY"), None
        if key:
            name, sep, value = key.partition("=")
            if not sep or not name.strip():
                raise ValueError("PERESEARCH_SEARCH_MCP_KEY must be NAME=value: the variable the search server reads "
                                 "its key from, e.g. TAVILY_API_KEY=tvly-…")
            env = {name.strip(): value.strip()}
        return cls(Server(target, env=env), settings.get("PERESEARCH_SEARCH_TOOL") or None)

    def _pick(self) -> dict:
        tools = self.server.tools()
        if self.tool:
            found = next((t for t in tools if t.get("name") == self.tool), None)
            if found is None:
                raise Unavailable(f"no tool {self.tool} on {self.server.name} (its tools: "
                                  f"{', '.join(t.get('name', '?') for t in tools)})")
            return found
        searching = [t for t in tools if "search" in t.get("name", "").lower()]
        if not searching:
            raise Unavailable(f"{self.server.name} offers no search tool (its tools: "
                              f"{', '.join(t.get('name', '?') for t in tools) or 'none'}); set PERESEARCH_SEARCH_TOOL")
        return next((t for t in searching if "web" in t["name"].lower()), searching[0])

    @staticmethod
    def arguments(schema: dict, query: str, n: int) -> dict:
        """The call's arguments from the tool's inputSchema: the query in the string property named like one (else
        the first required string, else the first string); n in the number property named like a count, if any."""
        props = (schema or {}).get("properties") or {}
        strings = [k for k, v in props.items() if (v or {}).get("type") == "string"]
        q = next((k for k in strings if k.lower() in QUERY_NAMES), None) or next(
            (k for k in (schema or {}).get("required") or [] if k in strings), None) or (strings[0] if strings else "query")
        out = {q: query}
        count = next((k for k, v in props.items() if (v or {}).get("type") in ("integer", "number")
                      and k.lower() in COUNT_NAMES), None)
        if count:
            out[count] = n
        return out

    def search(self, query: str, n: int) -> Found:
        from peresearch.mcp import MCPError

        try:
            tool = self._pick()
            text = self.server.call(tool["name"], self.arguments(tool.get("inputSchema"), query, n))
        except MCPError as e:
            raise Unavailable(f"the web search server did not answer: {e}") from None
        return Found(f"{tool['name']} on {self.server.name}", text, links(text))

    def close(self) -> None:
        self.server.close()


OLD_KEYS = {"TAVILY_API_KEY": ("Tavily", "npx -y tavily-mcp"), "EXA_API_KEY": ("Exa", "npx -y exa-mcp-server")}


def check(searcher: "McpSearch | None") -> list[str]:
    """`peresearch check` and /setup: what ships and what the user adds; with a server, one search ("test") — it
    answers, how fast, how many links — or why not; a key of the removed adapters still in the settings, where it
    goes now."""
    import time

    if searcher is None:
        lines = ["web search: none set — what ships: arXiv, Semantic Scholar and Wikipedia (search_papers, "
                 "search_wikipedia) and reading a page; add a search MCP server in /setup (e.g. npx -y tavily-mcp, "
                 "https://mcp.exa.ai/mcp, a DuckDuckGo or SearXNG server)"]
        for key, (who, command) in OLD_KEYS.items():
            if settings.get(key):
                lines.append(f"your settings hold {key}: {who}'s MCP server takes it — PERESEARCH_SEARCH_MCP={command}, "
                             f"PERESEARCH_SEARCH_MCP_KEY={key}=…")
        return lines
    t0 = time.monotonic()
    try:
        found = searcher.search("test", 3)
    except Unavailable as e:
        return [f"web search: FAILED: {e}"]
    return [f"web search: {found.provider} · answered in {time.monotonic() - t0:.1f} s · {len(found.links)} links"]


SETTINGS = object()      # Web's default search: the user's server from the settings, or none


def _http():
    import httpx

    return httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True)


def _is_pdf(content_type: str, url: str, body: bytes) -> bool:
    return "pdf" in content_type.lower() or urlsplit(url).path.lower().endswith(".pdf") or body[:5] == b"%PDF-"


def _charset(name: str | None) -> str | None:
    """A declared charset Python knows, or None."""
    try:
        return codecs.lookup(name).name if name else None
    except LookupError:
        return None


def _decode(body: bytes, content_type: str, html: bool) -> str:
    """Text in the encoding the server declares, else (HTML) the page's meta tag, else as a local file would be read
    (a byte order mark, UTF-8, or the likeliest legacy encoding)."""
    declared = re.search(r"charset\s*=\s*[\"']?([\w.:-]+)", content_type, re.I)
    enc = _charset(declared.group(1) if declared else None)
    if enc is None and html:
        meta = _CHARSET.search(body[:4096])
        enc = _charset(meta.group(1).decode("ascii") if meta else None)
    if enc is not None and not body.startswith(tuple(b for b, _, _ in parse.BOMS)):
        return body.decode(enc, errors="replace")
    return parse.decode(body)[0]


def extract(body: bytes, content_type: str, url: str) -> str | None:
    """Main text of a page: the document converter for a PDF, trafilatura for HTML, as-is for other text; None for a
    file that is not text (an image, an archive, anything with a NUL byte that is not a PDF)."""
    ct = content_type.lower()
    if _is_pdf(ct, url, body):
        doc = parse.read_pdf(body, url)
        return "\n\n".join(f"[p. {b.start}]\n{b.text}" for b in doc.blocks).strip() if doc.status == "ok" else ""
    if (ct and not any(t in ct for t in _TEXT_TYPES)) or b"\x00" in body[:1024]:
        return None
    html = "html" in ct or body.lstrip()[:15].lower().startswith((b"<!doctype", b"<html"))
    text = _decode(body, ct, html)
    if html:
        import trafilatura

        return (trafilatura.extract(text, url=url, include_comments=False, include_tables=True) or "").strip()
    return text.strip()


# What kind of site a result comes from, by its host name: a fact for the model to weigh, never a filter. Hosts are
# matched by their last labels, so "notedu.com" is not a university. Sites not listed are "web page".
SITE_KINDS = [
    ("preprint server", ("arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "osf.io", "preprints.org",
                         "researchsquare.com", "chemrxiv.org", "hal.science")),
    ("peer review site", ("openreview.net",)),
    ("scholarly publisher", ("doi.org", "dl.acm.org", "ieeexplore.ieee.org", "link.springer.com", "springer.com",
                             "sciencedirect.com", "nature.com", "science.org", "wiley.com", "tandfonline.com",
                             "aclanthology.org", "proceedings.neurips.cc", "proceedings.mlr.press", "jmlr.org",
                             "mdpi.com", "frontiersin.org", "plos.org", "cell.com", "thelancet.com", "nejm.org",
                             "academic.oup.com", "cambridge.org", "sagepub.com", "aaai.org", "ijcai.org",
                             "thecvf.com", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "semanticscholar.org")),
    ("encyclopedia", ("wikipedia.org", "britannica.com")),
    ("code repository", ("github.com", "gitlab.com", "huggingface.co", "bitbucket.org", "pypi.org")),
    ("forum", ("stackoverflow.com", "stackexchange.com", "reddit.com", "quora.com", "news.ycombinator.com")),
    ("blog", ("medium.com", "substack.com", "blogspot.com", "wordpress.com", "dev.to", "towardsdatascience.com",
              "hashnode.dev")),
    ("video", ("youtube.com", "youtu.be", "vimeo.com", "bilibili.com")),
]
_UNIVERSITY = re.compile(r"(^|\.)(edu|ac\.[a-z]{2}|edu\.[a-z]{2})$")
_GOVERNMENT = re.compile(r"(^|\.)(gov|gov\.[a-z]{2}|go\.[a-z]{2}|gouv\.[a-z]{2}|gob\.[a-z]{2}|mil|nic\.in)$")


def site_kind(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    for kind, hosts in SITE_KINDS:
        if any(host == h or host.endswith("." + h) for h in hosts):
            return kind
    if _GOVERNMENT.search(host):
        return "government"
    if _UNIVERSITY.search(host):
        return "university"
    return "web page"


def _checked(url: str) -> tuple[bool, str | None]:
    """(whether `url` points to the public internet, the checked address to connect to). Not public: this machine,
    a private, link-local or reserved address, or a name that resolves to one. A name that does not resolve is let
    through (it fails by itself when fetched); an address written in the URL needs no resolving."""
    host = urlsplit(url).hostname or ""
    if not host or host == "localhost" or host.endswith((".localhost", ".local")):
        return False, None
    try:
        return ipaddress.ip_address(host).is_global, None
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return True, None
    ips = [info[4][0].split("%")[0] for info in infos]
    ok = all(ipaddress.ip_address(ip).is_global for ip in ips)
    return ok, (ips[0] if ok else None)


def public(url: str) -> bool:
    """Whether `url` points to the public internet (see _checked)."""
    return _checked(url)[0]


def _pinned(url: str, ip: str) -> tuple[str, dict, dict]:
    """(URL, headers, extensions) that reach `ip` while the site still sees its own name: the Host header, and for
    HTTPS the TLS name the certificate is checked against. A second lookup could answer another address (DNS
    rebinding); none is made."""
    parts = urlsplit(url)
    netloc = (f"[{ip}]" if ":" in ip else ip) + (f":{parts.port}" if parts.port else "")
    extensions = {"sni_hostname": parts.hostname} if parts.scheme == "https" else {}
    return parts._replace(netloc=netloc).geturl(), {"Host": parts.netloc.rsplit("@", 1)[-1]}, extensions


class Web:
    def __init__(self, search=SETTINGS, http=None, cache=None, pin=None):
        """`search`: the user's search server (McpSearch.from_settings) by default, or one given; None or empty: no web
        search. `pin`: connect to the address that was checked (default: when this client is the real one)."""
        self.pin = http is None if pin is None else pin
        self.http = http or _http()
        self.searcher = McpSearch.from_settings() if search is SETTINGS else (search or None)
        self.cache = cache or guard.home() / "web"

    def search(self, query: str, n: int = 5) -> Found:
        """What the user's search server found for `query`; none set, or failing: Unavailable, said."""
        query = guard.outbound_query(query, "web_search")
        if self.searcher is None:
            raise Unavailable("no web search is set: add a search MCP server in /setup")
        return self.searcher.search(query, n)

    def close(self) -> None:
        """Stop the search server, if this client started one."""
        if self.searcher is not None:
            self.searcher.close()

    def fetch(self, url: str) -> str:
        url = guard.outbound_query(url, "fetch")
        path = self.cache / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".txt")
        if path.exists():
            return path.read_text(encoding="utf-8")
        text: str | None = ""
        try:
            text = self._get(url)
        except Exception as e:
            guard.audit("fetch_failed", error=type(e).__name__)
        if text is None:                        # not text, too large, or not on the public internet
            return ""
        key = settings.get("JINA_API_KEY")
        if not text and key:                    # a page that needs JavaScript: the Jina Reader the user chose renders it
            try:
                r = self.http.get("https://r.jina.ai/" + url, timeout=60, headers={"Authorization": f"Bearer {key}"})
                if r.status_code == 200:
                    text = r.text.strip()
            except Exception as e:
                guard.audit("reader_failed", error=type(e).__name__)
        if text:
            guard.private_folder(self.cache)
            with os.fdopen(guard.open_private(path, os.O_WRONLY | os.O_TRUNC), "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def _get(self, url: str) -> str | None:
        """The page's text; None when it must not or cannot be read as text. Redirects are followed here, each
        target checked like the first URL."""
        for _ in range(MAX_REDIRECTS + 1):
            ok, ip = _checked(url)
            if not ok:
                guard.audit("fetch_refused", reason="not a public address")
                return None
            target, headers, extensions = _pinned(url, ip) if self.pin and ip else (url, {}, {})
            with self.http.stream("GET", target, headers=headers, extensions=extensions, timeout=30,
                                  follow_redirects=False) as r:
                if r.is_redirect:
                    url = urljoin(url, r.headers["location"])
                    continue
                r.raise_for_status()
                ct = r.headers.get("content-type", "")
                body, limit = b"", MAX_PAGE_BYTES
                for part in r.iter_bytes():
                    if not body and _is_pdf(ct, url, part):
                        limit = MAX_PDF_BYTES
                    body += part
                    if len(body) > limit:
                        if limit == MAX_PDF_BYTES:     # a PDF cut short cannot be opened
                            guard.audit("fetch_refused", reason="PDF too large")
                            return None
                        break
                return extract(body, ct, url)
        guard.audit("fetch_refused", reason="too many redirects")
        return None
