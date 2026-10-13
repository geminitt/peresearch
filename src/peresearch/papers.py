"""Scholarly search: papers, with what kind of record each is, from free sources taken together.

Semantic Scholar and arXiv need no key (Semantic Scholar's keyless pool is shared by everyone and often refuses with
429: a free SEMANTIC_SCHOLAR_API_KEY gives a request a second of one's own); OpenAlex joins when OPENALEX_API_KEY is
set (its free key gives a daily allowance; the key travels as a bearer token, never in a URL). Every source is asked
for each query and the answers are merged one from each source in turn, so every source contributes; the same paper
(by DOI, arXiv id or title) is listed once, with every source that found it. A source that fails, throttles or
refuses its key is left out of that answer; when none answers the agent is told. arXiv is asked at most once every 3
seconds (its terms of use).

Each paper says what kind of record it is — journal article, conference paper, review, preprint (and where it was
published, if arXiv knows), RETRACTED — with its venue, year and citation count when the source gives them. These
are facts for the model to weigh, not a filter: nothing is dropped for its kind. Every query passes
`guard.outbound_query` (one holding a secret is refused, not sent). Results are untrusted data for the agent.
"""

import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from peresearch import guard, settings

ARXIV_SPACING = 3.0                       # seconds between two arXiv requests (arXiv API terms of use)
_ARXIV_LAST = [0.0]                       # when arXiv was last asked, for every client in this process
_ARXIV_LOCK = threading.Lock()
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass
class Paper:
    title: str
    url: str                              # the DOI's address when there is one, else the source's page
    pdf: str | None = None                # an open-access PDF, readable with fetch
    year: int | None = None
    venue: str = ""
    kind: str = ""                        # journal article | conference paper | review article | preprint… | ""
    citations: int | None = None
    retracted: bool = False
    doi: str = ""
    arxiv: str = ""                       # without its version, e.g. 1706.03762
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    found_by: list[str] = field(default_factory=list)


class Unavailable(Exception):
    """No scholarly source can answer."""


def _doi(value: str | None) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", (value or "").strip(), flags=re.I).lower()


def _arxiv_id(value: str | None) -> str:
    m = re.search(r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$", (value or "").strip())
    return m.group(1) if m else ""


def _same_title(title: str) -> str:
    return re.sub(r"\W+", " ", title.casefold()).strip()


class SemanticScholar:
    name, url = "semantic scholar", "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,year,venue,publicationTypes,citationCount,externalIds,openAccessPdf,authors,abstract,url"
    KINDS = {"JournalArticle": "journal article", "Conference": "conference paper", "Review": "review article"}

    def __init__(self, key: str, http):
        self.key, self.http = key, http

    def search(self, query: str, n: int) -> list[Paper]:
        r = self.http.get(self.url, params={"query": query, "limit": min(n, 100), "fields": self.FIELDS},
                          headers={"x-api-key": self.key} if self.key else {}, timeout=30)
        r.raise_for_status()
        out = []
        for x in r.json().get("data") or []:
            ids = x.get("externalIds") or {}
            doi, arxiv = _doi(ids.get("DOI")), _arxiv_id(ids.get("ArXiv"))
            types = x.get("publicationTypes") or []
            kind = next((self.KINDS[t] for t in ("Review", "JournalArticle", "Conference") if t in types), "")
            if not kind and (arxiv and (x.get("venue") or "").lower() in ("", "arxiv.org", "arxiv")):
                kind = "preprint"
            url = (f"https://doi.org/{doi}" if doi else f"https://arxiv.org/abs/{arxiv}" if arxiv
                   else x.get("url") or "")
            out.append(Paper(
                title=(x.get("title") or "").strip(), url=url, pdf=((x.get("openAccessPdf") or {}).get("url") or None),
                year=x.get("year"), venue=x.get("venue") or "", kind=kind, citations=x.get("citationCount"),
                doi=doi, arxiv=arxiv, authors=[a.get("name", "") for a in x.get("authors") or []],
                abstract=(x.get("abstract") or "").strip(), found_by=[self.name]))
        return out


class Arxiv:
    name, url = "arxiv", "https://export.arxiv.org/api/query"

    def __init__(self, key: str, http):
        self.http = http

    def search(self, query: str, n: int) -> list[Paper]:
        with _ARXIV_LOCK:                                    # one request at a time, 3 s apart, process-wide
            wait = _ARXIV_LAST[0] + ARXIV_SPACING - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.http.get(self.url, params={"search_query": f"all:{query}", "start": 0, "max_results": n},
                                  timeout=30)
            finally:
                _ARXIV_LAST[0] = time.monotonic()
        r.raise_for_status()
        out = []
        for e in ET.fromstring(r.content).findall("a:entry", _ATOM):
            text = lambda tag: " ".join((e.findtext(tag, "", _ATOM) or "").split())
            page = text("a:id")
            pdf = next((l.get("href") for l in e.findall("a:link", _ATOM) if l.get("title") == "pdf"), None)
            ref = text("arxiv:journal_ref")
            published = text("a:published")
            out.append(Paper(
                title=text("a:title"), url=page.replace("http://", "https://"),
                pdf=pdf.replace("http://", "https://") if pdf else None,
                year=int(published[:4]) if published[:4].isdigit() else None, venue=ref,
                kind=f"preprint; published as {ref}" if ref else "preprint", doi=_doi(text("arxiv:doi")),
                arxiv=_arxiv_id(page), authors=[" ".join((a.findtext("a:name", "", _ATOM)).split())
                                               for a in e.findall("a:author", _ATOM)],
                abstract=text("a:summary"), found_by=[self.name]))
        return out


class OpenAlex:
    name, url = "openalex", "https://api.openalex.org/works"
    KINDS = {"journal": "journal article", "conference": "conference paper", "repository": "preprint"}   # of an article
    WORKS = {"preprint": "preprint", "review": "review article", "conference-paper": "conference paper"}

    def __init__(self, key: str, http):
        self.key, self.http = key, http

    def search(self, query: str, n: int) -> list[Paper]:
        r = self.http.get(self.url, params={"search": query, "per-page": min(n, 200)},
                          headers={"Authorization": f"Bearer {self.key}"}, timeout=30)
        r.raise_for_status()
        out = []
        for x in r.json().get("results") or []:
            where = x.get("primary_location") or {}
            source = where.get("source") or {}
            work = x.get("type") or ""
            kind = self.WORKS.get(work) or (self.KINDS.get(source.get("type"), "") if work == "article" else "") \
                or work.replace("-", " ")
            words = x.get("abstract_inverted_index") or {}
            abstract = " ".join(w for _, w in sorted((i, w) for w, places in words.items() for i in places))
            doi = _doi(x.get("doi"))
            out.append(Paper(
                title=(x.get("display_name") or "").strip(),
                url=f"https://doi.org/{doi}" if doi else where.get("landing_page_url") or x.get("id") or "",
                pdf=(x.get("best_oa_location") or {}).get("pdf_url") or None, year=x.get("publication_year"),
                venue=source.get("display_name") or "", kind=kind, citations=x.get("cited_by_count"),
                retracted=bool(x.get("is_retracted")), doi=doi,
                authors=[(a.get("author") or {}).get("display_name", "") for a in x.get("authorships") or []],
                abstract=abstract, found_by=[self.name]))
        return out


def _http():
    import httpx

    from peresearch.web import USER_AGENT
    return httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True)


class Papers:
    def __init__(self, sources=None, http=None):
        if sources is None:
            http = http or _http()
            key = settings.get("OPENALEX_API_KEY")
            sources = ([OpenAlex(key, http)] if key else []) + [
                SemanticScholar(settings.get("SEMANTIC_SCHOLAR_API_KEY") or "", http), Arxiv("", http)]
        self.sources = sources
        self.missed: list[str] = []          # the sources that did not answer the last search, and why

    def search(self, query: str, n: int = 5) -> list[Paper]:
        """Up to `n` papers, one from each source in turn, each paper once. `missed` names the sources that did not
        answer (measured 2026-10-03: Semantic Scholar without a key answered 429 three times in a row)."""
        query = guard.outbound_query(query, "paper_search")
        answers, self.missed = [], []
        for s in self.sources:
            try:
                answers.append(s.search(query, n))
            except Exception as e:           # down, throttled (429), a refused key: the others still answer
                code = getattr(getattr(e, "response", None), "status_code", None)
                self.missed.append(f"{s.name} ({code}{', throttled' if code == 429 else ''})" if code
                                   else f"{s.name} ({type(e).__name__})")
                guard.audit("paper_source_failed", source=s.name, error=type(e).__name__, status=code)
        if not answers:
            raise Unavailable("no scholarly source can answer (" + "; ".join(self.missed) + ")")
        merged: list[Paper] = []
        keys: dict[str, Paper] = {}
        for rank in range(max((len(a) for a in answers), default=0)):
            for a in answers:
                if rank >= len(a) or not a[rank].title:
                    continue
                p = a[rank]
                found = []                          # the listed records sharing a key with p, in list order
                for m in merged:
                    if any(k in keys and keys[k] is m for k in _keys(p)):
                        found.append(m)
                if not found:
                    merged.append(p)
                    same = p
                else:
                    # A record holding both a DOI and an arXiv id can join two records listed apart (one had the
                    # DOI, the other the arXiv id and another title): all one paper, kept where it first appeared
                    same = found[0]
                    for other in found[1:]:
                        _merge(same, other)
                        merged = [m for m in merged if m is not other]
                    _merge(same, p)
                for k in _keys(same) + _keys(p):
                    keys[k] = same
                for k, m in list(keys.items()):
                    if not any(m is x for x in merged):
                        keys[k] = same              # the keys of a record merged away point to the one kept
        held = [k for m in merged for k in _keys(m)]
        assert len(held) == len(set(held)), "two listed records share a DOI, an arXiv id or a title"
        return merged[:n]


def _keys(p: Paper) -> list[str]:
    """What tells a paper: its DOI, its arXiv id, its title as written whatever the case and punctuation."""
    return [k for k in (f"doi:{p.doi}" if p.doi else "", f"arxiv:{p.arxiv}" if p.arxiv else "",
                        f"title:{_same_title(p.title)}") if k]


def _merge(into: Paper, other: Paper) -> None:
    """One record from two sources' records of the same paper: the published kind over "preprint", the larger
    citation count, retracted if either says so, and whatever the first one lacked."""
    if other.kind and (not into.kind or (into.kind.startswith("preprint") and not other.kind.startswith("preprint"))):
        into.kind = other.kind
        into.venue = other.venue or into.venue
    if other.citations is not None and (into.citations is None or other.citations > into.citations):
        into.citations = other.citations
    into.retracted = into.retracted or other.retracted
    for name in ("pdf", "year", "venue", "doi", "arxiv", "abstract"):
        if not getattr(into, name) and getattr(other, name):
            setattr(into, name, getattr(other, name))
    if not into.authors:
        into.authors = other.authors
    into.found_by += [s for s in other.found_by if s not in into.found_by]
