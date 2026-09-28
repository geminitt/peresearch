"""The web: free search providers taken in turn, and a page reader. Nothing here touches the personal index.

Search goes to Tavily first and to Exa once Tavily's free credits run out (or its key is missing or refused);
when both are out the agent is told the web is unavailable — there is no paid fallback. Every query and URL
passes `guard.outbound_query`: one holding a secret is refused, not sent. A page is fetched with httpx and
reduced to its main text by trafilatura on this machine; the Jina Reader (r.jina.ai) is asked only when that
yields nothing (pages that need JavaScript). Pages are cached under $PERESEARCH_HOME/web, apart from the index
of personal files, and the text is untrusted data for the agent.
"""

import hashlib
from dataclasses import dataclass

from peresearch import guard, settings

MAX_PAGE_BYTES = 5 * 2**20
USER_AGENT = "peresearch/0.1 (+https://github.com/geminitt/peresearch)"


@dataclass
class Result:
    title: str
    url: str
    snippet: str


class Exhausted(Exception):
    """The provider will not answer until its quota resets (or its key is missing or refused)."""


class Unavailable(Exception):
    """No search provider can answer."""


class Tavily:
    name, url = "tavily", "https://api.tavily.com/search"

    def __init__(self, key: str, http):
        self.key, self.http = key, http

    def search(self, query: str, n: int) -> list[Result]:
        r = self.http.post(self.url, headers={"Authorization": f"Bearer {self.key}"}, timeout=30,
                           json={"query": query, "max_results": n, "search_depth": "basic"})
        if r.status_code in (401, 403, 432, 433):            # bad key, plan limit, pay-as-you-go limit
            raise Exhausted(f"tavily {r.status_code}")
        r.raise_for_status()
        return [Result(x.get("title", ""), x["url"], x.get("content", "")) for x in r.json().get("results", [])]


class Exa:
    name, url = "exa", "https://api.exa.ai/search"

    def __init__(self, key: str, http):
        self.key, self.http = key, http

    def search(self, query: str, n: int) -> list[Result]:
        r = self.http.post(self.url, headers={"x-api-key": self.key}, timeout=30,
                           json={"query": query, "numResults": n, "type": "auto",
                                 "contents": {"highlights": {"maxCharacters": 600}}})
        if r.status_code in (401, 402, 403):                  # bad key, no credits left
            raise Exhausted(f"exa {r.status_code}")
        r.raise_for_status()
        return [Result(x.get("title") or "", x["url"], " … ".join(x.get("highlights") or []) or (x.get("text") or "")[:600])
                for x in r.json().get("results", [])]


def _http():
    import httpx

    return httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True)


def extract(body: bytes, content_type: str, url: str) -> str:
    """Main text of a page: trafilatura for HTML, pymupdf for PDF, as-is for plain text."""
    ct = content_type.lower()
    if "pdf" in ct or url.lower().endswith(".pdf"):
        import pymupdf

        with pymupdf.open(stream=body, filetype="pdf") as pdf:
            return "\n\n".join(page.get_text("text") for page in pdf).strip()
    text = body.decode("utf-8", errors="replace")
    if "html" in ct or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        import trafilatura

        return (trafilatura.extract(text, url=url, include_comments=False, include_tables=True) or "").strip()
    return text.strip()


class Web:
    def __init__(self, providers=None, http=None, cache=None):
        self.http = http or _http()
        if providers is None:
            providers = [cls(settings.get(k), self.http) for cls, k in ((Tavily, "TAVILY_API_KEY"), (Exa, "EXA_API_KEY"))
                         if settings.get(k)]
        self.providers, self.exhausted = providers, set()
        self.cache = cache or guard.home() / "web"

    def search(self, query: str, n: int = 5) -> tuple[str, list[Result]]:
        """(provider, results). A provider out of quota is skipped for the rest of the session."""
        query = guard.outbound_query(query, "web_search")
        errors = []
        for p in self.providers:
            if p.name in self.exhausted:
                continue
            try:
                return p.name, p.search(query, n)[:n]
            except Exhausted as e:
                self.exhausted.add(p.name)
                guard.audit("provider_exhausted", provider=p.name)
                errors.append(str(e))
            except Exception as e:       # rate limit or outage: this query goes to the next provider
                errors.append(f"{p.name}: {type(e).__name__}")
        raise Unavailable("no web search provider can answer (" + ("; ".join(errors) or "none configured") + ")")

    def fetch(self, url: str) -> str:
        url = guard.outbound_query(url, "fetch")
        path = self.cache / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".txt")
        if path.exists():
            return path.read_text()
        text = ""
        try:
            with self.http.stream("GET", url, timeout=30) as r:
                r.raise_for_status()
                body = b""
                for part in r.iter_bytes():
                    body += part
                    if len(body) > MAX_PAGE_BYTES:
                        break
                text = extract(body, r.headers.get("content-type", ""), url)
        except Exception as e:
            guard.audit("fetch_failed", error=type(e).__name__)
        if not text:                            # a page that needs JavaScript: the Jina Reader renders it
            key = settings.get("JINA_API_KEY")
            try:
                r = self.http.get("https://r.jina.ai/" + url, timeout=60,
                                  headers={"Authorization": f"Bearer {key}"} if key else {})
                if r.status_code == 200:
                    text = r.text.strip()
            except Exception as e:
                guard.audit("reader_failed", error=type(e).__name__)
        if text:
            self.cache.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return text
