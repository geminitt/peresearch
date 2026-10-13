"""Wikipedia: overviews and definitions in the user's languages, and a term's name in the other target languages.

Each language edition named (at most MAX_EDITIONS a search) is searched through its MediaWiki API: for each article,
its lead section as plain text, its address, and its title in the other target languages (English, Vietnamese,
Chinese, Japanese, Korean, Hindi) — what a cross-language search needs. An encyclopedia is a starting point and a
map to sources, not primary evidence; the tool says so. No key is needed. A language code is checked before it
becomes a host name; the query passes `guard.outbound_query` (one holding a secret is refused, not sent). An answer
continued over several parts (an article with hundreds of language links) is read up to MAX_PARTS parts.
"""

import re
from dataclasses import dataclass, field

from peresearch import guard

TARGET_LANGS = ("en", "vi", "zh", "ja", "ko", "hi")
MAX_EDITIONS = 3
MAX_RESULTS = 10
MAX_PARTS = 4
_LANG = re.compile(r"^[a-z]{2,3}(-[a-z]{2,8})?$")


@dataclass
class Article:
    title: str
    url: str
    lang: str
    summary: str
    other: dict[str, str] = field(default_factory=dict)       # language -> title, target languages only


class Wikipedia:
    name = "wikipedia"

    def __init__(self, http=None):
        if http is None:
            import httpx

            from peresearch.web import USER_AGENT
            http = httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True)
        self.http = http

    def search(self, query: str, langs: list[str], n: int = 3) -> tuple[list[Article], list[str]]:
        """(articles, editions that did not answer). Articles are grouped by edition, in each in search rank."""
        bad = [x for x in langs if not isinstance(x, str) or not _LANG.match(x)]
        if bad or not langs:
            raise ValueError(f"not a Wikipedia language code: {bad or langs!r} (use codes such as en, vi, zh)")
        query = guard.outbound_query(query, "wikipedia")
        out, missed = [], []
        for lang in list(dict.fromkeys(langs))[:MAX_EDITIONS]:
            try:
                out += self._edition(query, lang, min(max(int(n), 1), MAX_RESULTS))
            except Exception as e:           # an edition down or refusing: the others still answer
                code = getattr(getattr(e, "response", None), "status_code", None)
                missed.append(f"{lang}.wikipedia ({code or type(e).__name__})")
                guard.audit("wikipedia_failed", lang=lang, error=type(e).__name__, status=code)
        return out, missed

    def _edition(self, query: str, lang: str, n: int) -> list[Article]:
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {"action": "query", "format": "json", "formatversion": "2", "generator": "search", "gsrsearch": query,
                  "gsrlimit": n, "prop": "extracts|langlinks|info", "exintro": 1, "explaintext": 1, "exlimit": "max",
                  "lllimit": "max", "inprop": "url"}
        pages: dict[int, dict] = {}
        more: dict = {}
        for _ in range(MAX_PARTS):
            r = self.http.get(url, params={**params, **more}, timeout=30)
            r.raise_for_status()
            data = r.json()
            for pg in (data.get("query") or {}).get("pages") or []:
                seen = pages.setdefault(pg["pageid"], {"langlinks": []})
                seen.update({k: v for k, v in pg.items() if k != "langlinks"})
                seen["langlinks"] += pg.get("langlinks") or []
            more = {k: v for k, v in (data.get("continue") or {}).items()}
            if "llcontinue" not in more:
                break
        ranked = sorted(pages.values(), key=lambda pg: pg.get("index", 1 << 30))
        return [Article(pg.get("title", ""), pg.get("fullurl", ""), lang, (pg.get("extract") or "").strip(),
                        {x["lang"]: x["title"] for x in pg["langlinks"] if x.get("lang") in TARGET_LANGS
                         and x["lang"] != lang})
                for pg in ranked if pg.get("title")]
