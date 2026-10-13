"""Contracts of the Wikipedia search (peresearch.wiki), with fake HTTP: what is asked of each language edition, how
articles are ordered and read (lead summary, address, the title in the other target languages, across continued
answers), that a language code cannot reach another host, and that a query holding a secret is never sent."""

import httpx
import pytest

from peresearch import wiki
from peresearch.wiki import Article, Wikipedia
from tests.test_guard import CANARIES
from tests.test_web import Recorder

VI = "https://vi.wikipedia.org/w/api.php"
EN = "https://en.wikipedia.org/w/api.php"


def page(i, title, extract, links):
    return {"pageid": i * 10, "index": i, "title": title, "fullurl": f"https://vi.wikipedia.org/wiki/{title}",
            "extract": extract, "langlinks": [{"lang": l, "title": t} for l, t in links]}


def vi_answer(request):
    if request.url.params.get("llcontinue"):                            # the second part of a long answer
        return httpx.Response(200, json={"query": {"pages": [{"pageid": 10, "index": 1, "title": "Học máy",
                                                               "langlinks": [{"lang": "ja", "title": "機械学習"}]}]}})
    return httpx.Response(200, json={"continue": {"llcontinue": "10|ja", "continue": "||"}, "query": {"pages": [
        page(2, "Khoa học máy tính", "Khoa học máy tính là ...", [("en", "Computer science"), ("fr", "Informatique")]),
        page(1, "Học máy", "Học máy là một lĩnh vực ...", [("en", "Machine learning"), ("zh", "机器学习"),
                                                             ("de", "Maschinelles Lernen")])]}})


def test_an_edition_is_searched_and_its_articles_read_in_rank_order_across_continued_answers():
    rec = Recorder({VI: vi_answer})
    got, missed = Wikipedia(rec.client()).search("học máy", ["vi"], 2)
    assert missed == []
    assert got == [Article("Học máy", "https://vi.wikipedia.org/wiki/Học máy", "vi", "Học máy là một lĩnh vực ...",
                           {"en": "Machine learning", "zh": "机器学习", "ja": "機械学習"}),
                   Article("Khoa học máy tính", "https://vi.wikipedia.org/wiki/Khoa học máy tính", "vi",
                           "Khoa học máy tính là ...", {"en": "Computer science"})]          # target languages only
    first = rec.requests[0].url.params
    assert (first["gsrsearch"], first["gsrlimit"], first["generator"], first["exintro"], first["explaintext"]) == (
        "học máy", "2", "search", "1", "1")
    assert {"extracts", "langlinks", "info"} == set(first["prop"].split("|")) and first["inprop"] == "url"
    assert rec.requests[1].url.params["llcontinue"] == "10|ja" and len(rec.requests) == 2
    assert Wikipedia().http.headers["user-agent"].startswith("peresearch/")       # Wikimedia asks a name and contact


def test_several_editions_at_most_three_and_a_failing_one_is_named():
    rec = Recorder({VI: vi_answer, EN: httpx.Response(503),
                    "https://ja.wikipedia.org/": httpx.Response(200, json={"batchcomplete": True})})   # nothing found
    got, missed = Wikipedia(rec.client()).search("x", ["vi", "en", "ja", "ko"], 5)
    assert [a.lang for a in got] == ["vi", "vi"] and missed == ["en.wikipedia (503)"]
    assert [r.url.host for r in rec.requests if not r.url.params.get("llcontinue")] == [
        "vi.wikipedia.org", "en.wikipedia.org", "ja.wikipedia.org"]
    assert Wikipedia(rec.client()).search("x", ["vi"], 50)[0] and rec.requests[-2].url.params["gsrlimit"] == "10"


@pytest.mark.parametrize("lang", ["evil.com/x?", "vi.wikipedia.org@evil", "", "VI", "a" * 12, "../"])
def test_a_language_code_can_only_name_a_wikipedia_edition(lang):
    rec = Recorder({})
    with pytest.raises(ValueError, match="language code"):
        Wikipedia(rec.client()).search("x", [lang], 3)
    assert rec.requests == []


def test_a_query_holding_a_secret_is_never_sent():
    rec = Recorder({VI: vi_answer})
    with pytest.raises(Exception, match="secret"):
        Wikipedia(rec.client()).search("why " + CANARIES["github-token"], ["vi"], 2)
    assert rec.requests == []


def test_a_continued_answer_stops_after_a_few_parts():
    def endless(request):
        return httpx.Response(200, json={"continue": {"llcontinue": "x"}, "query": {"pages": [page(1, "A", "a", [])]}})
    rec = Recorder({VI: endless})
    got, _ = Wikipedia(rec.client()).search("x", ["vi"], 1)
    assert len(got) == 1 and len(rec.requests) == wiki.MAX_PARTS


def test_what_is_sent_and_recorded_exactly(home):
    """What the mutation run of 2026-10-03 showed no test held."""
    import json
    rec = Recorder({VI: vi_answer})
    Wikipedia(rec.client()).search("học máy", ["vi"], 1)
    assert dict(rec.requests[0].url.params) == {
        "action": "query", "format": "json", "formatversion": "2", "generator": "search", "gsrsearch": "học máy",
        "gsrlimit": "1", "prop": "extracts|langlinks|info", "exintro": "1", "explaintext": "1", "exlimit": "max",
        "lllimit": "max", "inprop": "url"}
    log = [json.loads(x) for x in (home / "audit.jsonl").read_text().splitlines()]
    assert {k: v for k, v in log[-1].items() if k not in ("time", "chars")} == {
        "event": "outbound", "channel": "wikipedia", "redacted": 0}

    def down(request):
        raise httpx.ConnectError("no route")
    _, missed = Wikipedia(Recorder({VI: down}).client()).search("x", ["vi"], 1)
    assert missed == ["vi.wikipedia (ConnectError)"]
    log = [json.loads(x) for x in (home / "audit.jsonl").read_text().splitlines()]
    assert {k: v for k, v in log[-1].items() if k != "time"} == {"event": "wikipedia_failed", "lang": "vi",
                                                                "error": "ConnectError", "status": None}
    with pytest.raises(ValueError, match=r"not a Wikipedia language code: \['EN'\]"):
        Wikipedia(rec.client()).search("x", ["vi", "EN"], 1)
    with pytest.raises(ValueError, match=r"\[\]"):
        Wikipedia(rec.client()).search("x", [], 1)
    assert Wikipedia().http.follow_redirects is True
