"""Contracts of the data filter (zetokrag.datafilter): a plain-text file is kept out of ranking when it holds data
(a log, a table, a word list, JSON, labels, tokens, addresses, numbers) and kept when it holds something to read, in
any of the six target languages, whatever its line lengths, formulas, numbering or markup. The labelled public set
and its measurement live in eval/datafilter.py; these are the cases each rule exists for."""

import json

import pytest

from peresearch.zetokrag import datafilter

P = "Gradient descent updates the weights against the gradient, a small step at a time, until the loss stops falling."
PROSE = {
    "en": [P, "It works because the gradient points uphill.", "Momentum keeps a running average of past steps."],
    "vi": ["Thuật toán hạ gradient cập nhật trọng số ngược hướng đạo hàm, từng bước nhỏ một, cho tới khi mất mát dừng giảm.",
           "Nó hiệu quả vì đạo hàm chỉ hướng đi lên.", "Quán tính giữ trung bình chạy của các bước trước."],
    "zh": ["梯度下降法沿着梯度的反方向更新权重，每次走一小步，直到损失不再下降为止。", "它之所以有效，是因为梯度指向上升的方向。",
           "动量保留了过去各步的滑动平均。"],
    "ja": ["勾配降下法は勾配と逆の方向に重みを少しずつ更新し、損失が下がらなくなるまで続ける。", "勾配が上り坂を指すので、うまくいく。",
           "モーメンタムは過去のステップの移動平均を保つ。"],
    "ko": ["경사 하강법은 손실이 더 이상 줄지 않을 때까지 기울기의 반대 방향으로 가중치를 조금씩 갱신한다.",
           "기울기가 오르막을 가리키기 때문에 효과가 있다.", "모멘텀은 과거 단계의 이동 평균을 유지한다."],
    "hi": ["ग्रेडिएंट डिसेंट भार को ग्रेडिएंट की उलटी दिशा में थोड़ा-थोड़ा बदलता है, जब तक हानि घटना बंद न हो जाए।",
           "यह इसलिए काम करता है क्योंकि ग्रेडिएंट ऊपर की ओर इशारा करता है।", "मोमेंटम पिछले कदमों का चलता औसत रखता है।"],
}


def lines(*rows, times=10):
    return "\n".join(list(rows) * times) + "\n"


@pytest.mark.parametrize("lang", list(PROSE))
def test_prose_in_each_target_language_is_kept(lang):
    assert datafilter.reason("notes.txt", lines(*PROSE[lang])) is None


@pytest.mark.parametrize("name, text", [
    ("formulas.txt", lines(P, "x = w - lr * g", "", "where", "g = dL/dw", "and lr is the step size.", P)),
    ("poem.txt", lines("I celebrate myself, and sing myself,", "And what I assume you shall assume,",
                       "For every atom belonging to me as good belongs to you.")),
    ("todo.txt", "\n".join(f"{i}. Read the paper on {t} and write what it claims, with the numbers."
                           for i, t in enumerate(["BPE", "SuperBPE", "Unigram LM", "WordPiece"] * 6, 1))),
    ("library.bib", lines("@article{vaswani2017,", "  title = {Attention Is All You Need},", "  year = {2017},", "}")),
    ("tiny.txt", "a,b,c\n1,2,3\n"),                                    # too few lines to tell: kept
    ("notes.md", lines("1,2,3,4")),                                    # markdown and code are not judged here
    ("model.py", lines("x = [1, 2, 3]")),
])
def test_what_is_read_is_kept_however_it_is_laid_out(name, text):
    assert datafilter.reason(name, text) is None


@pytest.mark.parametrize("name, text, why", [
    ("server.txt", lines("081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block terminating",
                         "081109 203807 222 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block terminating"),
     "log"),
    ("hpc.txt", lines("134681 node-246 unix.hw state_change.unavailable 1077804742 1 Component State Change",
                      "2557 node-127 action start 1077805204 1 clusterAddMember (command 1902)"), "log"),
    ("results.txt", lines("model,dataset,accuracy,f1", "bert,imdb,0.912,0.905", "roberta,imdb,0.934,0.931"), "table"),
    ("words.txt", lines("the", "of", "and", "to", "in", "is"), "word list"),
    ("dict-zh.txt", lines("机器学习 3 n", "梯度 5 n", "下降 7 v"), "word list"),
    ("export.txt", lines('{"id": 1, "text": "a sentence of prose inside JSON is still a record"}',
                         '{"id": 2, "text": "and so is this one, however long its text may be"}'), "JSON"),
    ("0019.txt", lines("0 2 Car 0 0 -1.793451 296.744956 161.752147 455.226042 292.372804 2.000000 1.823255 4.433886",
                       "0 3 Pedestrian 0 0 -1.936993 737.619499 161.531951 931.112229 374.000000 1.739063 0.824591"),
     "numbers and codes"),
    ("conll.txt", lines("EU NNP B-NP B-ORG", "rejects VBZ B-VP O", "German JJ B-NP B-MISC", ""), "word list"),
    ("urls.txt", "".join(f"http://www.imdb.com/title/tt{i:07d}/\n" for i in range(60)), "URLs or numbers alone"),
    ("nums.txt", "".join(f"{i / 3:.4f}\n" for i in range(60)), "URLs or numbers alone"),
])
def test_data_is_left_out_with_what_it_looks_like(name, text, why):
    assert datafilter.reason(name, text) == why


def test_a_small_file_of_numbers_is_data_and_a_small_note_is_not():
    """Seven lines of camera calibration (KITTI calib/*.txt: 42 chunks of numbers came through, 2026-10-03)."""
    calib = "\n".join(f"P{i}: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02"
                      for i in range(7))
    assert datafilter.reason("0019.txt", calib) == "numbers and codes"
    assert datafilter.reason("note.txt", "Meeting on 2024-05-01.\nRead the BPE paper first.\nAsk about the loss.") is None
    assert datafilter.reason("two.txt", "1 2 3\n4 5 6") is None                 # two lines: too little to tell


def test_a_long_file_is_judged_by_its_start_middle_and_end():
    """A log with a prose header, or prose with a data appendix: the sample reaches every part."""
    head = [P] * 30
    log = ["081109 203615 148 INFO dfs.DataNode: block %d" % i for i in range(3000)]
    assert datafilter.reason("run.txt", "\n".join(head + log)) == "log"
    assert datafilter.reason("paper.txt", "\n".join([P] * 3000 + ["0.1,0.2,0.3"] * 30)) is None


def test_every_judgement_is_explained_by_the_features():
    f = datafilter.features(lines(*PROSE["hi"]))
    assert f["letters"] > 0.9                                          # Devanagari vowel signs count as letters
    assert set(f) == {"lines", "letters", "long", "stamps", "fields", "json", "digit_led", "addresses"}


@pytest.mark.parametrize("name, text, why", [
    ("model_pb2.py", "# -*- coding: utf-8 -*-\n# Generated by the protocol buffer compiler.  DO NOT EDIT!\n"
                     + "x = 1\n" * 50, "generated code"),
    ("api.go", "// Code generated by mockgen. DO NOT EDIT.\npackage api\n" + "func f() {}\n" * 30, "generated code"),
    ("schema.ts", "/**\n * @generated SignedSource<<abc>>\n */\n" + "export type A = string;\n" * 30, "generated code"),
    ("bundle.min.js", "!function(e){" + "var a=1;" * 2000 + "}();\n", "minified code"),
    ("app.css", ".a{color:red}" * 900 + "\n", "minified code"),
    ("index.html", "<div class=a>" + "<p>x</p>" * 900 + "</div>\n", "minified code"),   # HTML read as it is: markup
    ("train.py", "# Train the model. Do not edit the config by hand: copy it.\n" + "x = 1\n" * 50, None),   # a person's words
    ("train2.py", "x = 1\n" * 40 + "# DO NOT EDIT below this line\n", None),                   # not in the header
    ("long_line.py", "TABLE = [" + "1, " * 300 + "]\n" + "x = 1\n" * 50, None),          # one long line among many
    # mutmut's copies (nanovitok/mutants/: 18,284 of its 19,344 chunks, 2026-10-05): its trampoline comes after the
    # module's docstring, past any header, and every function is there many times, renamed
    ("adapt.py", '"""What the module does.\n' + "More of the docstring.\n" * 40 + '"""\nimport os\n'
                 "from mutmut.mutation.trampoline import wrap_in_trampoline as _mutmut_mutated, MutantDict\n"
                 + "".join(f"def x_adapt__mutmut_{i}(a, b):\n    return a + b\n\n" for i in ("orig", 1, 2, 3)),
     "mutation-testing copy"),
    ("mutation.py", "from mutmut import main\n\ndef run_mutmut(paths):\n    return main(paths)\n" + "x = 1\n" * 40,
     None),                                                                       # a person's code that uses mutmut
])
def test_generated_or_minified_code_is_named_and_written_code_is_kept(name, text, why):
    assert datafilter.reason(name, text) == why


# --- the features against a reference written line by line ----------------------------------------------------------

import re  # noqa: E402
import unicodedata  # noqa: E402
from collections import Counter  # noqa: E402

from hypothesis import example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

TOKENS = ["gradient", "the", "học", "máy", "机器学习", "勾配", "경사", "ग्रेडिएंट", "42", "3.14", "2024-05-01", "081109",
          "203615", "http://a.org/x", "{", "}", '"id":', "[1,", "|", ",", ";", "\t", "x=1", "INFO", "e+02", "—"]


def ref_features(text):
    """What features() measures, written out step by step (see the module's docstring)."""
    rows = []
    for raw in text.split("\n"):
        if raw.strip():
            rows.append(raw.strip())
    if len(rows) > 600:                                                 # 200 from the start, middle and end
        mid = len(rows) // 2
        rows = rows[:200] + rows[mid - 100:mid + 100] + rows[len(rows) - 200:]
    chars = [c for r in rows for c in r if not c.isspace()]
    if not chars:
        return None
    cjk = re.compile(r"[぀-ヿ㐀-鿿가-힯豈-﫿]")

    def words(r):
        n = len(cjk.findall(r)) / 2
        for w in cjk.sub(" ", r).split():
            if any(c.isalpha() for c in w):
                n += 1
        return n

    def shape(r):
        # a word character that is not a decimal digit or "_" → a (a combining mark is not one), a decimal digit → 9
        s = "".join("a" if c.isalnum() and not c.isdecimal() else "9" if c.isdecimal() else c for c in r[:16])
        out = ""
        for c in s:
            if not out or out[-1] != c:
                out += c
        return out

    n, total = len(rows), sum(len(r) for r in rows)
    start, count = Counter(shape(r) for r in rows).most_common(1)[0]
    fields = 0.0
    for d in [",", "\t", ";", "|"]:
        k = Counter(r.count(d) for r in rows).most_common(1)[0][0]
        if k >= 1:
            good = 0
            for r in rows:
                if r.count(d) == k and max(words(c) for c in r.split(d)) <= 4:
                    good += 1
            fields = max(fields, good / n)
    stamp = re.compile(r"^(\d{3,}|\d{1,4}[-/.:]\d{1,2}[-/.:]\d{1,4})")
    single = re.compile(r"^\S+://\S+$")
    number = re.compile(r"^[-+.,:;/|eE\d\s]*\d[-+.,:;/|eE\d\s]*$")
    return {"lines": n,
            "letters": sum(unicodedata.category(c)[0] in ("L", "M") for c in chars) / len(chars),
            "long": sum(len(r) for r in rows if words(r) >= 6) / total,
            "stamps": count / n if len(re.findall("9+", start)) >= 2 else 0.0,
            "fields": fields,
            "json": sum(r[0] in "{}[]\"" for r in rows) / n,
            "digit_led": sum(1 for r in rows if stamp.match(r)) / n,
            "addresses": sum(1 for r in rows if single.match(r) or number.match(r)) / n}


line = st.lists(st.sampled_from(TOKENS), max_size=9).map(" ".join)


@settings(max_examples=600, deadline=None)
@given(st.lists(line, max_size=40), st.sampled_from([1, 3, 20]))
# a long file whose start, middle and end differ, so that only the right sample windows give the reference's numbers
@example([f"{i} 2 3" for i in range(250)] + [f"word {i} of a longer line of prose here" for i in range(250)]
         + [f"http://a.org/{i}" for i in range(201)] + ["x"] * 99 + ["{" + str(i) for i in range(300)], 1)
def test_the_features_are_what_the_docstring_says(rows, times):
    """Every feature equals the reference, short files and long ones (sampled from start, middle and end)."""
    text = "\n".join(f"{r} {i}" if times > 1 else r for i, r in enumerate(rows * times))
    want = ref_features(text)
    got = datafilter.features(text)
    if want is None:
        assert got == {"lines": 0, "letters": 0.0, "long": 0.0, "stamps": 0.0, "fields": 0.0, "json": 0.0,
                       "digit_led": 0.0, "addresses": 0.0}
    else:
        assert got.keys() == want.keys() and all(got[k] == pytest.approx(want[k]) for k in want), (got, want)


def test_an_empty_or_blank_file_is_kept():
    assert datafilter.reason("empty.txt", "") is None and datafilter.reason("blank.txt", "\n \n\t\n") is None


# --- each rule at its threshold, in its order --------------------------------------------------------------------

BASE = {"lines": 50, "letters": 0.9, "long": 0.8, "stamps": 0.0, "fields": 0.0, "json": 0.0, "digit_led": 0.0,
        "addresses": 0.0}


@pytest.mark.parametrize("change, why", [
    ({}, None),
    ({"addresses": 0.9}, "URLs or numbers alone"), ({"addresses": 0.899}, None),
    ({"json": 0.6}, "JSON"), ({"json": 0.599}, None),
    ({"fields": 0.6}, "table"), ({"fields": 0.599}, None),
    ({"letters": 0.499}, "numbers and codes"), ({"letters": 0.5}, None),
    ({"stamps": 0.5}, "log"), ({"stamps": 0.499}, None), ({"digit_led": 0.7}, "log"), ({"digit_led": 0.699}, None),
    ({"long": 0.299}, "word list"), ({"long": 0.3}, None),
    ({"addresses": 0.9, "json": 1.0, "fields": 1.0}, "URLs or numbers alone"),       # the first rule that holds
    ({"json": 0.6, "fields": 1.0, "letters": 0.1}, "JSON"), ({"fields": 0.6, "letters": 0.1}, "table"),
    ({"letters": 0.1, "stamps": 1.0}, "numbers and codes"), ({"stamps": 0.5, "long": 0.0}, "log"),
    ({"lines": 20, "letters": 0.1}, "numbers and codes"),                              # 20 lines: the full rules
])
def test_each_rule_holds_from_its_threshold_in_its_order(monkeypatch, change, why):
    monkeypatch.setattr(datafilter, "features", lambda text: {**BASE, **change})
    assert datafilter.reason("a.b.txt", "x" * 100) == why                     # the last dot names the type


@pytest.mark.parametrize("lines_, text, letters, why", [
    (3, "x" * 40, 0.499, "numbers and codes"), (2, "x" * 40, 0.1, None), (3, "x" * 39, 0.1, None),
    (3, "x" * 40, 0.5, None), (19, "x" * 40, 0.1, "numbers and codes"), (19, "x" * 40, 0.9, None),
])
def test_a_file_under_twenty_lines_is_data_only_when_mostly_numbers(monkeypatch, lines_, text, letters, why):
    monkeypatch.setattr(datafilter, "features", lambda t: {**BASE, "lines": lines_, "letters": letters,
                                                          "long": 0.0, "stamps": 1.0})
    assert datafilter.reason("calib.txt", text) == why


@pytest.mark.parametrize("name, judged", [
    ("notes.txt", True), ("NOTES.TXT", True), ("a.b.rst", True), ("x.org", True), ("m.py", True),
    ("readme", False), ("paper.md", False), ("lib.bib", False), ("txt", False), ("archive.txt.gz", False),
])
def test_only_plain_text_and_code_are_judged(name, judged):
    assert datafilter.judges(name) is judged
    if not judged:
        assert datafilter.reason(name, "1,2\n" * 100) is None


@pytest.mark.parametrize("text, why", [
    ("x\n" * 9 + "# DO NOT EDIT\n" + "y\n" * 30, "generated code"),               # the 10th line is in the header
    ("x\n" * 10 + "# DO NOT EDIT\n" + "y\n" * 30, None),                            # the 11th is not
    (("a" * 500 + "\n") * 10, None), (("a" * 501 + "\n") * 10, "minified code"),   # a mean of 500 is still written
    ("a" * 999 + "\n" + "b\n", None), ("a" * 1000 + "\n" + "b\n", "minified code"),   # mean (999 + 1) / 2 = 500
    ("a" * 6000 + "\n" + "b\n" * 11, "minified code"),                              # one very long line among few
    ("\n\n", None),
    ("x = 1\n" * 30 + "".join(f"def f__mutmut_{i}():\n    pass\n" for i in (1, 2)), None),       # two copies: kept
    ("x = 1\n" * 30 + "".join(f"def f__mutmut_{i}():\n    pass\n" for i in (1, 2, 3)), "mutation-testing copy"),
    ("x = 1\n" * 30 + "".join(f"def f__mutmut_{i}():\n    pass\n" for i in range(1, 60)), "mutation-testing copy"),
    ("x = 1\n" * 30 + "".join(f"    async def m__mutmut_{i}(self):\n        pass\n" for i in (1, 2, 3)),
     "mutation-testing copy"),                                                     # methods, async ones too
    ("x = 1\n" * 30 + "# f__mutmut_1 f__mutmut_2 f__mutmut_3 are names, not definitions\n", None),
], ids=["header-10", "header-11", "mean-500", "mean-501", "two-500", "two-500.5", "long-line", "blank",
        "mutants-2", "mutants-3", "mutants-59", "mutant-methods", "mutant-names-in-a-comment"])
def test_code_is_left_out_only_from_each_threshold(text, why):
    assert datafilter.reason("m.js", text) == why


SCHEMA = ("CREATE TABLE users (\n  id INTEGER PRIMARY KEY,\n  name TEXT NOT NULL,\n  email TEXT UNIQUE,\n"
          "  created_at TIMESTAMP\n);\n") * 5


@pytest.mark.parametrize("text, why", [
    ("-- phpMyAdmin SQL Dump\nINSERT INTO `scam` (`id`, `text`) VALUES\n"
     + "".join(f"({i}, 'a message that looks like a scam number {i}'),\n" for i in range(60)) + "(60, 'last');\n",
     "SQL data"),                                                                   # one value tuple a line
    ("".join(f"INSERT INTO Track VALUES ({i}, 'Song {i}', 1, 343719);\n" for i in range(60)), "SQL data"),
    ("COPY public.actor (actor_id, first_name) FROM stdin;\n" + "".join(f"{i}\tPENELOPE\t2006-02-15\n" for i in range(60))
     + "\\.\n", "SQL data"),                                                        # pg_dump's COPY rows
    ("".join(f"Insert into city\n (city_id,city)\nValues\n('{i}','Kimchon')\n;\n" for i in range(30)),
     "SQL data"),                                                                   # one statement on five lines
    (SCHEMA, None),                                                                  # a schema is read
    (SCHEMA + "INSERT INTO users VALUES (1, 'admin', 'a@b.c', now());\n", None),     # a migration seeding one row
    ("".join(f"INSERT INTO t VALUES ({i});\n" for i in range(3)) + "CREATE TABLE a (x INT);\n" * 2, "SQL data"),  # 60%
    ("".join(f"INSERT INTO t VALUES ({i});\n" for i in range(3)) + "CREATE TABLE a (x INT);\n" * 3, None),        # 50%
    ("COPY t (a) FROM stdin;\n1\n2\n\\.\n" + "CREATE TABLE a (x INT);\n" * 5, None),            # a COPY block ends
    ("INSERT INTO t VALUES (1);\n" + "CREATE TABLE a (x INT);\n" * 4, None),              # an INSERT ends at ";"
    ("COPY t TO '/tmp/t.csv';\n" + "CREATE TABLE a (x INT);\n" * 4, None),                # COPY … TO: no rows
    ("CREATE TABLE a (x INT);\n" * 3 + "\n" + "".join(f"INSERT INTO a VALUES ({i});\n" for i in range(9)),
     "SQL data"),                                                                   # rows after a blank line
], ids=["phpmyadmin", "insert-lines", "pg-copy", "insert-five-lines", "schema", "migration", "rows-60", "rows-50",
        "copy-ends", "insert-ends", "copy-to", "after-blank"])
def test_an_sql_dump_is_data_and_a_schema_is_read(text, why):
    """A SQL file whose lines are mostly table rows (lines of INSERT statements, COPY rows) is data: a 4 MB
    phpMyAdmin dump of 39,505 value tuples made 2,911 chunks, 41% of the owner's index (2026-10-04). Only .sql files:
    the same lines in Python are code."""
    assert datafilter.reason("db.sql", text) == why
    assert datafilter.reason("seed.py", text) is None
