"""Contracts of the scholarly search (peresearch.papers), with fake HTTP: what each source is asked, how its records
are read (kind, venue, year, citations, retraction, PDF), how sources are merged and deduplicated, that arXiv is
asked at most once every 3 s, and that a query holding a secret is never sent."""

import json
from types import SimpleNamespace as NS

import httpx
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from peresearch import papers
from peresearch.papers import Arxiv, OpenAlex, Papers, SemanticScholar, Unavailable
from tests.test_guard import CANARIES
from tests.test_web import Recorder


def P(**kw):
    base = dict(title="T", url="https://u", pdf=None, year=None, venue="", kind="", citations=None, retracted=False,
                doi="", arxiv="", authors=[], abstract="", found_by=["a"])
    return papers.Paper(**{**base, **kw})

S2 = {"total": 3, "data": [
    {"paperId": "p1", "title": "Attention Is All You Need", "year": 2017, "venue": "Neural Information Processing Systems",
     "publicationTypes": ["Conference"], "citationCount": 120000,
     "externalIds": {"DOI": "10.5555/3295222.3295349", "ArXiv": "1706.03762"},
     "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762"},
     "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}, {"name": "Niki Parmar"}, {"name": "Jakob Uszkoreit"}],
     "abstract": "The dominant sequence transduction models are based on recurrent networks.",
     "url": "https://www.semanticscholar.org/paper/p1"},
    {"paperId": "p2", "title": "A Journal Study", "year": 2020, "venue": "Nature", "publicationTypes": ["JournalArticle"],
     "citationCount": 12, "externalIds": {"DOI": "10.1038/x"}, "openAccessPdf": None, "authors": [{"name": "A. Author"}],
     "abstract": None, "url": "https://www.semanticscholar.org/paper/p2"},
    {"paperId": "p3", "title": "Only On Arxiv", "year": 2026, "venue": "arXiv.org", "publicationTypes": None,
     "citationCount": 0, "externalIds": {"ArXiv": "2601.00001"}, "openAccessPdf": None, "authors": [], "abstract": "x",
     "url": "https://www.semanticscholar.org/paper/p3"}]}

ARXIV = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All
      You Need</title>
    <summary>  The dominant sequence transduction models.  </summary>
    <author><name>Ashish Vaswani</name></author>
    <link href="http://arxiv.org/abs/1706.03762v7" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v7" rel="related" type="application/pdf"/>
    <arxiv:journal_ref>NeurIPS 2017</arxiv:journal_ref>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2601.00002v1</id>
    <published>2026-01-02T00:00:00Z</published>
    <title>A Fresh Preprint</title>
    <summary>New.</summary>
    <author><name>B. Writer</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2601.00002v1" rel="related" type="application/pdf"/>
  </entry>
</feed>"""

OPENALEX = {"meta": {"count": 2}, "results": [
    {"id": "https://openalex.org/W9", "display_name": "Withdrawn Findings", "publication_year": 2019, "type": "article",
     "cited_by_count": 40, "is_retracted": True, "doi": "https://doi.org/10.9/withdrawn",
     "primary_location": {"landing_page_url": "https://doi.org/10.9/withdrawn",
                          "source": {"display_name": "Journal of Results", "type": "journal"}},
     "best_oa_location": None, "authorships": [{"author": {"display_name": "C. Author"}}],
     "abstract_inverted_index": {"We": [0], "report": [1], "results.": [2]}},
    {"id": "https://openalex.org/W1", "display_name": "Attention is all you need", "publication_year": 2017,
     "type": "article", "cited_by_count": 130000, "is_retracted": False, "doi": "https://doi.org/10.5555/3295222.3295349",
     "primary_location": {"landing_page_url": "https://dl.acm.org/x", "source": {"display_name": "NeurIPS",
                                                                               "type": "conference"}},
     "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/1706.03762"}, "authorships": [],
     "abstract_inverted_index": None}]}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """arXiv's spacing is checked by recording sleeps, not by waiting."""
    slept = []
    clock = {"t": 1000.0}
    monkeypatch.setattr(papers.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(papers.time, "sleep", lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s)))
    monkeypatch.setattr(papers, "_ARXIV_LAST", [0.0])
    return slept


def routes(**over):
    base = {SemanticScholar.url: httpx.Response(200, json=S2), Arxiv.url: httpx.Response(200, text=ARXIV),
            OpenAlex.url: httpx.Response(200, json=OPENALEX)}
    base.update(over)
    return Recorder(base)


def test_each_source_is_asked_as_documented_and_read_into_one_record_shape():
    rec = routes()
    http = rec.client()
    s2 = SemanticScholar("", http).search("attention", 3)
    assert s2[0] == papers.Paper(
        title="Attention Is All You Need", url="https://doi.org/10.5555/3295222.3295349",
        pdf="https://arxiv.org/pdf/1706.03762", year=2017, venue="Neural Information Processing Systems",
        kind="conference paper", citations=120000, retracted=False, doi="10.5555/3295222.3295349", arxiv="1706.03762",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        abstract="The dominant sequence transduction models are based on recurrent networks.", found_by=["semantic scholar"])
    assert [(p.kind, p.url) for p in s2[1:]] == [("journal article", "https://doi.org/10.1038/x"),
                                                 ("preprint", "https://arxiv.org/abs/2601.00001")]
    req = rec.requests[0]
    assert req.url.params["query"] == "attention" and req.url.params["limit"] == "3" and "x-api-key" not in req.headers
    assert {"title", "year", "venue", "publicationTypes", "citationCount", "externalIds", "openAccessPdf", "authors",
            "abstract", "url"} <= set(req.url.params["fields"].split(","))
    SemanticScholar("s2-key", http).search("attention", 3)
    assert rec.requests[-1].headers["x-api-key"] == "s2-key"

    ax = Arxiv("", http).search("attention", 2)
    assert [(p.title, p.kind, p.arxiv, p.pdf, p.year, p.url) for p in ax] == [
        ("Attention Is All You Need", "preprint; published as NeurIPS 2017", "1706.03762",
         "https://arxiv.org/pdf/1706.03762v7", 2017, "https://arxiv.org/abs/1706.03762v7"),
        ("A Fresh Preprint", "preprint", "2601.00002", "https://arxiv.org/pdf/2601.00002v1", 2026,
         "https://arxiv.org/abs/2601.00002v1")]
    assert ax[0].abstract == "The dominant sequence transduction models." and ax[0].citations is None
    assert rec.requests[-1].url.params["search_query"] == "all:attention" and rec.requests[-1].url.params["max_results"] == "2"

    oa = OpenAlex("oa-key", http).search("attention", 2)
    assert (oa[0].kind, oa[0].retracted, oa[0].abstract, oa[0].venue) == ("journal article", True, "We report results.",
                                                                          "Journal of Results")
    assert (oa[1].kind, oa[1].pdf, oa[1].doi) == ("conference paper", "https://arxiv.org/pdf/1706.03762",
                                                  "10.5555/3295222.3295349")
    assert rec.requests[-1].headers["authorization"] == "Bearer oa-key" and "oa-key" not in str(rec.requests[-1].url)


def test_sources_are_merged_in_turn_and_the_same_paper_is_listed_once():
    p = Papers([OpenAlex("k", routes().client()), SemanticScholar("", routes().client()), Arxiv("", routes().client())])
    got = p.search("attention", 6)
    titles = [x.title for x in got]
    assert titles == ["Withdrawn Findings", "Attention Is All You Need", "A Journal Study", "A Fresh Preprint",
                      "Only On Arxiv"]                                # one from each source in turn, copies merged
    attention = got[1]
    assert attention.found_by == ["semantic scholar", "arxiv", "openalex"]       # one record, every source named
    assert attention.kind == "conference paper" and attention.citations == 130000  # the published version's kind
    assert attention.abstract == "The dominant sequence transduction models are based on recurrent networks."
    assert len(p.search("attention", 2)) == 2


# Records of five papers as sources return them: each may lack its DOI or arXiv id, and its title may be written
# another way (case, punctuation) or be another title altogether (a preprint renamed when published)
IDENTITIES = [(f"10.1/p{i}", f"2401.0000{i}", f"Paper number {i}") for i in range(5)]
record = st.tuples(st.integers(0, 4), st.booleans(), st.booleans(), st.sampled_from(["same", "styled", "other"])).map(
    lambda t: (t[0], P(doi=IDENTITIES[t[0]][0] if t[1] else "", arxiv=IDENTITIES[t[0]][1] if t[2] else "",
                       title={"same": IDENTITIES[t[0]][2], "styled": IDENTITIES[t[0]][2].upper() + "!",
                              "other": f"Renamed {t[0]} {t[1]}{t[2]}"}[t[3]], found_by=[])))


@given(st.lists(st.lists(record, max_size=5), min_size=1, max_size=3), st.integers(1, 12))
@settings(max_examples=300)
@example([[(0, P(doi="10.1/p0", title="Preprint name", found_by=[]))],            # a DOI, and apart an arXiv id,
          [(0, P(arxiv="2401.00000", title="Published name", found_by=[]))],        # then one record with both:
          [(0, P(doi="10.1/p0", arxiv="2401.00000", title="Third name", found_by=[]))]], 5)   # one paper, not two
def test_each_paper_is_listed_once_and_none_is_lost(answers, n):
    """Whatever the sources return, the list holds each paper once (no two records share a DOI, an arXiv id or a
    title), every paper a source returned is in it (up to n), and records come one from each source in turn."""
    # each record tagged in found_by, which a merge unites: the trail of where every record went
    sources = [NS(name=f"s{i}", search=lambda q, k, i=i, a=[r for _, r in rows]: [
        P(**{**vars(x), "found_by": [f"{i}-{j}"]}) for j, x in enumerate(a)]) for i, rows in enumerate(answers)]
    got = Papers(sources).search("q", n)
    held = [k for x in got for k in papers._keys(x)]
    assert len(held) == len(set(held)), held                      # no two listed records share a DOI, id or title
    assert len(got) <= n
    trail = [t for x in got for t in x.found_by]
    assert len(trail) == len(set(trail))                          # a record is in one listed paper at most
    if len(got) < n:                                              # nothing cut by n: every record is in the list
        assert sorted(trail) == sorted(f"{i}-{j}" for i, rows in enumerate(answers) for j in range(len(rows)))


def test_a_failing_or_throttled_source_leaves_the_others_and_none_means_unavailable(no_waiting):
    rec = routes(**{SemanticScholar.url: httpx.Response(429), OpenAlex.url: httpx.Response(401)})
    http = rec.client()
    p = Papers([OpenAlex("bad", http), SemanticScholar("", http), Arxiv("", http)])
    assert [x.found_by for x in p.search("attention", 5)] == [["arxiv"], ["arxiv"]]
    assert p.missed == ["openalex (401)", "semantic scholar (429, throttled)"]       # said, so the model knows
    rec_ok = routes()
    p_ok = Papers([SemanticScholar("", rec_ok.client())])
    p_ok.search("attention", 1)
    assert p_ok.missed == []
    dead = Recorder({}).client()
    with pytest.raises(Unavailable, match="no scholarly source can answer"):
        Papers([SemanticScholar("", dead), Arxiv("", dead)]).search("attention", 3)


def test_arxiv_is_asked_at_most_once_every_three_seconds(no_waiting):
    a = Arxiv("", routes().client())
    a.search("one", 1)
    a.search("two", 1)
    a.search("three", 1)
    assert no_waiting == [3.0, 3.0]


def test_a_query_holding_a_secret_is_never_sent(home):
    rec = routes()
    p = Papers([SemanticScholar("", rec.client()), Arxiv("", rec.client())])
    with pytest.raises(Exception, match="secret|credential"):
        p.search("why does " + CANARIES["github-token"] + " fail", 3)
    assert rec.requests == []


def test_default_sources_need_no_key_and_openalex_joins_with_one(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    assert [s.name for s in Papers().sources] == ["semantic scholar", "arxiv"]
    monkeypatch.setenv("OPENALEX_API_KEY", "oa-key")
    assert [s.name for s in Papers().sources] == ["openalex", "semantic scholar", "arxiv"]


# --- what the mutation run of 2026-10-03 showed no test held ----------------------------------------------------

@pytest.mark.parametrize("into, other, want", [
    (dict(kind="preprint", venue=""), dict(kind="conference paper", venue="NeurIPS", found_by=["b"]),
     dict(kind="conference paper", venue="NeurIPS", found_by=["a", "b"])),               # the published version wins
    (dict(kind="journal article", venue="Nature"), dict(kind="preprint", venue="arXiv", found_by=["b"]),
     dict(kind="journal article", venue="Nature", found_by=["a", "b"])),                 # a preprint never replaces it
    (dict(kind=""), dict(kind="preprint", found_by=["a"]), dict(kind="preprint")),       # no source named twice
    (dict(citations=None), dict(citations=5), dict(citations=5)), (dict(citations=7), dict(citations=5), dict(citations=7)),
    (dict(citations=5), dict(citations=7), dict(citations=7)),
    (dict(retracted=False), dict(retracted=True), dict(retracted=True)),
    (dict(retracted=True), dict(retracted=False), dict(retracted=True)),
    (dict(), dict(pdf="https://p", year=2020, venue="V", doi="10.1/x", arxiv="2001.1", abstract="abs", authors=["X"]),
     dict(pdf="https://p", year=2020, venue="V", doi="10.1/x", arxiv="2001.1", abstract="abs", authors=["X"])),
    (dict(pdf="https://mine", year=2019, authors=["Me"], abstract="mine"), dict(pdf="https://p", year=2020,
     authors=["X"], abstract="abs"), dict(pdf="https://mine", year=2019, authors=["Me"], abstract="mine")),
])
def test_two_records_of_one_paper_become_one(into, other, want):
    a = P(**into)
    papers._merge(a, P(**other))
    assert a == P(**{**into, **want})


def test_openalex_and_semantic_scholar_records_are_read_whole():
    oa = dict(OPENALEX)
    oa["results"] = OPENALEX["results"] + [
        {"id": "https://openalex.org/W5", "display_name": " A Review ", "publication_year": 2021, "type": "review",
         "cited_by_count": 3, "is_retracted": False, "doi": None, "primary_location": {"landing_page_url": "https://l/5",
         "source": None}, "best_oa_location": {"pdf_url": None}, "authorships": None, "abstract_inverted_index": {}},
        {"id": "https://openalex.org/W6", "display_name": "Repo Copy", "publication_year": None, "type": "article",
         "cited_by_count": None, "is_retracted": None, "doi": None, "primary_location": None,
         "best_oa_location": None, "authorships": [{"author": None}], "abstract_inverted_index": None}]
    rec = routes(**{OpenAlex.url: httpx.Response(200, json=oa)})
    got = OpenAlex("k", rec.client()).search("q", 250)
    assert rec.requests[0].url.params["search"] == "q" and rec.requests[0].url.params["per-page"] == "200"
    assert got[0] == P(title="Withdrawn Findings", url="https://doi.org/10.9/withdrawn", year=2019,
                       venue="Journal of Results", kind="journal article", citations=40, retracted=True,
                       doi="10.9/withdrawn", authors=["C. Author"], abstract="We report results.", found_by=["openalex"])
    assert got[1] == P(title="Attention is all you need", url="https://doi.org/10.5555/3295222.3295349",
                       pdf="https://arxiv.org/pdf/1706.03762", year=2017, venue="NeurIPS", kind="conference paper",
                       citations=130000, doi="10.5555/3295222.3295349", found_by=["openalex"])
    assert got[2] == P(title="A Review", url="https://l/5", year=2021, kind="review article", citations=3,
                       found_by=["openalex"])
    assert got[3] == P(title="Repo Copy", url="https://openalex.org/W6", kind="article", authors=[""],
                       found_by=["openalex"])
    s2 = dict(S2, data=S2["data"] + [{"paperId": "p4", "title": "Survey", "year": 2022, "venue": "ACM CSUR",
                                       "publicationTypes": ["Review", "JournalArticle"], "citationCount": 9,
                                       "externalIds": {}, "authors": [{}], "url": "https://s2/p4"}])
    rec = routes(**{SemanticScholar.url: httpx.Response(200, json=s2)})
    got = SemanticScholar("", rec.client()).search("q", 150)
    assert rec.requests[0].url.params["limit"] == "100"
    assert got[1] == P(title="A Journal Study", url="https://doi.org/10.1038/x", year=2020, venue="Nature",
                       kind="journal article", citations=12, doi="10.1038/x", authors=["A. Author"],
                       found_by=["semantic scholar"])
    assert got[2] == P(title="Only On Arxiv", url="https://arxiv.org/abs/2601.00001", year=2026, venue="arXiv.org",
                       kind="preprint", citations=0, arxiv="2601.00001", abstract="x", found_by=["semantic scholar"])
    assert got[3] == P(title="Survey", url="https://s2/p4", year=2022, venue="ACM CSUR", kind="review article",
                       citations=9, authors=[""], found_by=["semantic scholar"])


def test_arxiv_records_are_read_whole_and_requests_keep_their_spacing(no_waiting, monkeypatch):
    xml = ARXIV.replace("<arxiv:journal_ref>", "<arxiv:doi>10.1/abc</arxiv:doi><author><name> Noam   Shazeer"
                        "</name></author><arxiv:journal_ref>")
    rec = routes(**{Arxiv.url: httpx.Response(200, text=xml)})
    got = Arxiv("", rec.client()).search("q", 2)
    assert rec.requests[0].url.params["start"] == "0"
    assert got[0] == P(title="Attention Is All You Need", url="https://arxiv.org/abs/1706.03762v7",
                       pdf="https://arxiv.org/pdf/1706.03762v7", year=2017, venue="NeurIPS 2017",
                       kind="preprint; published as NeurIPS 2017", doi="10.1/abc", arxiv="1706.03762",
                       authors=["Ashish Vaswani", "Noam Shazeer"], abstract="The dominant sequence transduction models.",
                       found_by=["arxiv"])
    clock = papers.time.monotonic()
    monkeypatch.setattr(papers.time, "monotonic", lambda: clock + 2.5)        # the last request was 2.5 s ago
    no_waiting.clear()
    Arxiv("", rec.client()).search("q", 1)
    assert no_waiting == [pytest.approx(0.5)]


def test_papers_sends_the_query_to_every_source_and_builds_its_sources_from_settings(home, monkeypatch):
    rec = routes()
    http = rec.client()
    p = Papers([SemanticScholar("", http), Arxiv("", http)])
    assert p.missed == []
    p.search("rare words", 2)
    assert [r.url.params.get("query") or r.url.params.get("search_query") for r in rec.requests] == [
        "rare words", "all:rare words"]
    log = [json.loads(x) for x in (home / "audit.jsonl").read_text().splitlines()]
    assert log[-1]["event"] == "outbound" and log[-1]["channel"] == "paper_search"

    def down(request):
        raise httpx.ConnectError("no route")
    p = Papers([SemanticScholar("", Recorder({SemanticScholar.url: down}).client()), Arxiv("", http)])
    p.search("x", 1)
    assert p.missed == ["semantic scholar (ConnectError)"]
    log = [json.loads(x) for x in (home / "audit.jsonl").read_text().splitlines()]
    assert {k: v for k, v in log[-1].items() if k != "time"} == {"event": "paper_source_failed",
                                                                "source": "semantic scholar", "error": "ConnectError",
                                                                "status": None}
    monkeypatch.setenv("OPENALEX_API_KEY", "oa")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2")
    client = httpx.Client()
    built = Papers(http=client).sources
    assert [(s.name, getattr(s, "key", None), s.http is client) for s in built] == [
        ("openalex", "oa", True), ("semantic scholar", "s2", True), ("arxiv", None, True)]
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY")
    own = Papers().sources
    assert own[1].key == "" and own[0].http is own[1].http is own[2].http
    assert own[0].http.headers["user-agent"].startswith("peresearch/") and own[0].http.follow_redirects is True


def test_identifiers_kinds_and_missing_parts_at_their_edges():
    assert papers._doi("HTTPS://DOI.ORG/10.1/AB") == "10.1/ab" and papers._doi("https://dx.doi.org/10.2/x") == "10.2/x"
    assert papers._arxiv_id("http://arxiv.org/abs/math.GT/0309136v1") == "math.GT/0309136"
    assert papers._arxiv_id("1706.03762v7") == "1706.03762" and papers._arxiv_id("not an id") == ""
    s2 = {"data": [
        {"title": "J", "publicationTypes": ["JournalArticle"], "externalIds": {"ArXiv": "2001.00001"}, "venue": ""},
        {"title": "Bare", "publicationTypes": None, "externalIds": {}, "venue": "", "url": "https://s2/b"},
        {"title": "Pre", "publicationTypes": [], "externalIds": {"ArXiv": "2001.00002"}, "venue": None},
        {"title": "Pre2", "publicationTypes": [], "externalIds": {"ArXiv": "2001.00003"}, "venue": "ArXiv"}]}
    got = SemanticScholar("", routes(**{SemanticScholar.url: httpx.Response(200, json=s2)}).client()).search("q", 4)
    assert [p.kind for p in got] == ["journal article", "", "preprint", "preprint"]
    oa = {"results": [{"display_name": "OA pre", "type": "preprint", "primary_location": {"source": {"type": "journal"}}}]}
    assert OpenAlex("k", routes(**{OpenAlex.url: httpx.Response(200, json=oa)}).client()).search("q", 1)[0].kind == "preprint"
    xml = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2601.00009v1</id>'
           '<title>No date, no pdf</title><summary>s</summary></entry></feed>')
    got = Arxiv("", routes(**{Arxiv.url: httpx.Response(200, text=xml)}).client()).search("q", 1)
    assert (got[0].year, got[0].pdf, got[0].authors) == (None, None, [])
    a, b = P(title="Attention is all you need", found_by=["a"]), P(title="Attention Is All You Need.", found_by=["b"])
    merged = Papers([NS(name="a", search=lambda q, n: [a]), NS(name="b", search=lambda q, n: [b])]).search("q", 5)
    assert merged == [a] and a.found_by == ["a", "b"]                   # the same title, however written



@pytest.mark.parametrize("work, source, kind", [
    ("conference-paper", None, "conference paper"),          # OpenAlex's own type, seen live 2026-10-03
    ("article", "journal", "journal article"), ("article", "conference", "conference paper"),
    ("article", "repository", "preprint"), ("article", None, "article"), ("preprint", "journal", "preprint"),
    ("review", "journal", "review article"), ("book-chapter", "book series", "book chapter"),
    ("dissertation", None, "dissertation"), (None, None, ""),
])
def test_an_openalex_type_becomes_a_plain_kind(work, source, kind):
    oa = {"results": [{"display_name": "W", "type": work,
                       "primary_location": {"source": {"type": source} if source else None}}]}
    got = OpenAlex("k", routes(**{OpenAlex.url: httpx.Response(200, json=oa)}).client()).search("q", 1)
    assert got[0].kind == kind
