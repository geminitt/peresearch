"""The labelled set for the data filter (sub-architecture 4), and its measurement.

The filter keeps a plain-text file out of ranking when it holds data rather than something to read. It is measured
on PUBLIC files only, so that no rule rests on one person's folders: prose in the six target languages (whole
Wikipedia articles, pinned by revision id; licences, READMEs and reStructuredText pages of the installed packages,
the same on every machine through pixi.lock; a changelog; poetry) against data (13 logs from LogHub, KITTI labels,
CoNLL token files, word and stop-word lists in five languages, a tokenizer's merges, CSV tables and the installed
packages' tables and JSON, SQL dumps), plus files that are structured but worth reading (a BibTeX library, SQL
schemas). Each file's label comes from its source, written here; every file
is pinned by its sha256 in the manifest.

    pixi run python eval/datafilter.py build          # download into $PERESEARCH_HOME/eval/datafilter/corpus
    pixi run python eval/datafilter.py measure        # the filter on the set: confusion, every mistake listed
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from peresearch.web import USER_AGENT  # noqa: E402

from peresearch import guard  # noqa: E402
LANGS = ["en", "vi", "zh", "ja", "ko", "hi"]
TOPICS = ["Photosynthesis", "Machine learning", "Coffee", "World War II", "Mathematics", "Climate change",
          "Linear algebra", "Volcano"]
GH = "https://raw.githubusercontent.com"
# (name, url, label, what it is)
FILES = [
    ("log-hdfs.log", f"{GH}/logpai/loghub/master/HDFS/HDFS_2k.log", "data", "server log"),
    ("log-apache.log", f"{GH}/logpai/loghub/master/Apache/Apache_2k.log", "data", "server log"),
    ("log-linux.txt", f"{GH}/logpai/loghub/master/Linux/Linux_2k.log", "data", "system log saved as .txt"),
    ("log-android.txt", f"{GH}/logpai/loghub/master/Android/Android_2k.log", "data", "app log saved as .txt"),
    ("log-openssh.log", f"{GH}/logpai/loghub/master/OpenSSH/OpenSSH_2k.log", "data", "server log"),
    ("kitti-label-000001.txt", f"{GH}/kuixu/kitti_object_vis/master/data/object/training/label_2/000001.txt", "data",
     "KITTI object labels (a few lines)"),
    ("conll-eng-testa.txt", f"{GH}/glample/tagger/master/dataset/eng.testa", "data", "CoNLL-2003 tokens and tags"),
    ("conll-eng-testb.txt", f"{GH}/glample/tagger/master/dataset/eng.testb", "data", "CoNLL-2003 tokens and tags"),
    ("covid-countries.csv", f"{GH}/datasets/covid-19/main/data/countries-aggregated.csv", "data", "a large table"),
    ("iris.csv", f"{GH}/mwaskom/seaborn-data/master/iris.csv", "data", "a small table"),
    ("words-en-10000.txt", f"{GH}/first20hours/google-10000-english/master/google-10000-english.txt", "data",
     "English word list"),
    ("words-vi-74k.txt", f"{GH}/duyet/vietnamese-wordlist/master/Viet74K.txt", "data", "Vietnamese word list"),
    ("dict-zh-jieba.txt", f"{GH}/fxsjy/jieba/master/jieba/dict.txt", "data", "Chinese words, counts and tags"),
    ("stopwords-ja.txt", f"{GH}/stopwords-iso/stopwords-ja/master/stopwords-ja.txt", "data", "Japanese stop words"),
    ("stopwords-ko.txt", f"{GH}/stopwords-iso/stopwords-ko/master/stopwords-ko.txt", "data", "Korean stop words"),
    ("stopwords-hi.txt", f"{GH}/stopwords-iso/stopwords-hi/master/stopwords-hi.txt", "data", "Hindi stop words"),
    ("qwen3-merges.txt", "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/main/merges.txt", "data",
     "a tokenizer's BPE merges"),
    *[(f"log-{n.lower()}.{'txt' if i % 2 else 'log'}", f"{GH}/logpai/loghub/master/{n}/{n}_2k.log", "data", "a log")
      for i, n in enumerate(["Windows", "Mac", "Spark", "Zookeeper", "Hadoop", "HPC", "Proxifier", "HealthApp"])],
    ("titanic.csv", f"{GH}/mwaskom/seaborn-data/master/titanic.csv", "data", "a table"),
    ("requests-history.txt", f"{GH}/psf/requests/main/HISTORY.md", "prose", "a changelog: short list items"),
    ("leaves-of-grass.txt", "https://www.gutenberg.org/cache/epub/1322/pg1322.txt", "prose", "poetry: short lines"),
    ("biblatex-examples.bib", f"{GH}/plk/biblatex/master/bibtex/bib/biblatex/biblatex-examples.bib", "prose",
     "a BibTeX library: structured, worth reading"),
    ("csl-readme.txt", f"{GH}/citation-style-language/styles/master/README.md", "prose", "a README saved as .txt"),
    ("chinook.sql", f"{GH}/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql", "data",
     "a SQL dump: a schema, then one INSERT a line"),
    ("sakila-data.sql", f"{GH}/jOOQ/sakila/main/postgres-sakila-db/postgres-sakila-insert-data.sql", "data",
     "a SQL dump of INSERT statements"),
    ("sakila-schema-pg.sql", f"{GH}/jOOQ/sakila/main/postgres-sakila-db/postgres-sakila-schema.sql", "prose",
     "a SQL schema: tables, views, functions (worth reading)"),
    ("sakila-schema-mysql.sql", f"{GH}/jOOQ/sakila/main/mysql-sakila-db/mysql-sakila-schema.sql", "prose",
     "a SQL schema (worth reading)"),
]


def corpus() -> Path:
    """In peresearch's own data folder, beside the gold set: never walked, so it never becomes the user's notes."""
    return guard.home() / "eval" / "datafilter" / "corpus"


def build() -> list[dict]:
    http = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True)
    out = corpus()
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    def save(name: str, body: bytes, label: str, what: str, source: str, **extra):
        path = out / label / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(body)
        manifest.append({"file": f"{label}/{name}", "label": label, "what": what, "source": source,
                         "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), **extra})

    for name, url, label, what in FILES:
        r = http.get(url)
        r.raise_for_status()
        save(name, r.content, label, what, url)
    env = Path(sys.prefix)                     # the pixi environment: the same files on every machine (pixi.lock)
    for path in sorted(env.rglob("*")):
        name = path.name
        if not path.is_file() or path.is_symlink():
            continue
        if name.lower().startswith(("license", "readme")) and name.lower().endswith(".txt") or name.endswith(".rst"):
            label, what = "prose", "a licence, README or reStructuredText page of an installed package"
        elif name.endswith((".csv", ".json")) and path.stat().st_size > 2048:
            label, what = "data", "a table or JSON file of an installed package"
        else:
            continue
        rel = path.relative_to(env).as_posix()
        save(rel.replace("/", "__"), path.read_bytes(), label, what, f"pixi env: {rel}")
    api = "https://{}.wikipedia.org/w/api.php"
    for topic in TOPICS:
        r = http.get(api.format("en"), params={"action": "query", "format": "json", "formatversion": 2,
                                               "titles": topic, "prop": "langlinks", "lllimit": "max"})
        links = {x["lang"]: x["title"] for x in r.json()["query"]["pages"][0].get("langlinks", [])}
        for lang in LANGS:
            title = topic if lang == "en" else links.get(lang)
            if not title:
                continue
            r = http.get(api.format(lang), params={"action": "query", "format": "json", "formatversion": 2,
                                                   "titles": title, "prop": "extracts|revisions", "explaintext": 1,
                                                   "rvprop": "ids"})
            pg = r.json()["query"]["pages"][0]
            slug = topic.lower().replace(" ", "-")
            save(f"wiki-{lang}-{slug}.txt", pg["extract"].encode("utf-8"), "prose", "a whole Wikipedia article",
                 f"{lang}.wikipedia.org/wiki/{title}", revid=pg["revisions"][0]["revid"])
    (out.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


def main(argv):
    ap = argparse.ArgumentParser(prog="datafilter")
    ap.add_argument("cmd", choices=["build", "measure"])
    args = ap.parse_args(argv)
    if args.cmd == "build":
        m = build()
        print(f"{len(m)} files: {sum(x['label'] == 'prose' for x in m)} prose, {sum(x['label'] == 'data' for x in m)} "
              f"data, {sum(x['bytes'] for x in m) / 1e6:.1f} MB in {corpus()}")
        return 0
    return measure()


def measure() -> int:
    """The filter on the labelled set: how much data it catches, how much prose it keeps, every mistake by name. A
    file is judged as the index judges it (through the document converter), under a name of a judged type."""
    from peresearch.zetokrag import datafilter, parse

    manifest = json.loads((corpus().parent / "manifest.json").read_text())
    caught = kept = 0
    for x in manifest:
        doc = parse.read(corpus() / x["file"])
        text = "\n".join(b.text for b in doc.blocks) if doc.status == "ok" else ""
        name = x["file"] if x["file"].endswith((".bib", ".tex", ".sql")) else x["file"] + ".txt"
        why = datafilter.reason(name, text)
        if x["label"] == "data":
            caught += why is not None
            if why is None:
                print(f"missed data: {x['file']} ({x['what']}, {datafilter.features(text)['lines']} lines)")
        else:
            kept += why is None
            if why is not None:
                print(f"prose called {why}: {x['file']} ({x['what']})")
    n_data = sum(x["label"] == "data" for x in manifest)
    print(f"data caught {caught}/{n_data}, prose kept {kept}/{len(manifest) - n_data}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
