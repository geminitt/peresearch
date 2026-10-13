import json
import os
import re
from pathlib import Path

import pytest

from peresearch import guard

# Fake credentials in the real formats. Each is built at run time so this file holds no key-shaped string.
CANARIES = {
    "private-key": "-----BEGIN " + "RSA PRIVATE KEY-----",
    "aws-access-key": "AKIA" + "QX7PEXAMPLE0CANA",
    "github-token": "ghp_" + "Canary0Canary1Canary2Canary3Canary45",
    "huggingface-token": "hf_" + "CanaryCanaryCanaryCanaryCanaryAB",
    "anthropic-key": "sk-ant-" + "api03-Canary0Canary1Canary2",
    "openai-key": "sk-proj-" + "Canary0Canary1Canary2Canary3",
    "google-api-key": "AIza" + "SyCanary0Canary1Canary2Canary3Canar",
    "slack-token": "xoxb-" + "1234567890-canarycanary",
    "modal-token": "ak-" + "Canary0Canary1Canary2Canary3",
    "modal-proxy-token": "ws-" + "Canary0Canary1Canary2Canary3",
    "tavily-key": "tvly-" + "Canary0Canary1Canary2Canary3",
    "groq-key": "gsk_" + "Canary0Canary1Canary2Canary3Canary4Canary5Canary6",
    "cerebras-key": "csk-" + "Canary0Canary1Canary2Canary3Canary4",
    "jina-key": "jina_" + "Canary0Canary1Canary2Canary3Canary4Canary5Canary6",
    "jwt": "eyJ" + "hbGciOiJIUzI1Ni" + ".eyJ" + "zdWIiOiIxMjM0NTY" + ".SflKxwRJSMeKKF2QT4",
    "assigned-secret": "api_key = 'Zx9#Qw7!Er5$Ty3&Ui1Op0'".replace("#", "Q").replace("!", "w").replace("$", "e").replace("&", "r"),
}


@pytest.mark.parametrize("kind", sorted(CANARIES))
def test_every_credential_format_is_found_and_redacted(kind):
    text = f"some notes\n{CANARIES[kind]}\nmore notes"
    assert guard.find_secrets(text), kind
    clean, n = guard.redact(text)
    secret = CANARIES[kind].split("= ")[-1].strip("'")
    assert n >= 1 and secret not in clean and "some notes" in clean and "more notes" in clean
    assert f"[REDACTED:{kind}]" in clean


def test_ordinary_text_is_not_mistaken_for_a_secret():
    for text in ["token = tokenizer_config", "the password field must be hashed", "BPE merges ab and cd",
                 "học máy và xử lý ngôn ngữ tự nhiên", "sk-learn is a typo for scikit-learn", "key: value",
                 "token = abc123abc123", "password = v2v2v2v2",          # digits and letters, but not random
                 # seen in public libraries' code (2026-10-05, 960,398 paragraphs): a class being called, a CSS class
                 "workload_identity_auth = SyncX509WorkloadIdentityAuth(\n", "auth = AsyncX509WorkloadIdentityAuth(",
                 'e == "passthrough" else "sk-toggleable__label-arrow"', "sk-toggleable__label-arrow-down-right"]:
        assert guard.find_secrets(text) == [], text
    # long enough to be judged, and not random: under the entropy floor (3.5 bits a character)
    for text in ["api_key = 1111111111aaaaaaaaaa", "token = abababab12121212"]:
        assert guard.find_secrets(text) == [], text
    for text in ["api_key = Zx9QQw7wEr5Ty3Ui1Op0", "token: 'Zx9QQw7wEr5Ty3Ui1Op0' (", "sk-" + "a1" * 24,
                 "sk-Canary0Canary1Canary2Canary3"]:                    # a value, not a call; a key with digits
        assert guard.find_secrets(text), text
    # In a programming language's code a written-in secret is a string literal, so the assigned-value rule wants its
    # quotes there (detect-secrets' KeywordDetector does so for Python, JS, Java, Swift…): unquoted, the value is a
    # name or an expression (134 hits in 2,336,197 public paragraphs, 2026-10-05: self.img_tokens, cfg.MAX_TOKENS_…,
    # LayoutLMv2TokenizerFast). Config files, shell, notes and messages keep the unquoted form (.env: TOKEN=abc).
    for text in ["tokens = self.img_tokens2", "MAX_TOKENS = cfg.MAX_TOKENS_PER_EXPERT_2", "auth = LayoutLMv2TokenizerFast",
                 "api_key = Zx9QQw7wEr5Ty3Ui1Op0", "token = ya29Zx9QQw7wEr5Ty3Ui1Op0"]:
        assert guard.find_secrets(text, quoted=True) == [], text
    for text in ["api_key = 'Zx9QQw7wEr5Ty3Ui1Op0'", 'token = "Zx9QQw7wEr5Ty3Ui1Op0"', "secret := `Zx9QQw7wEr5Ty3Ui1Op0`",
                 "TOKEN = r'Zx9QQw7wEr5Ty3Ui1Op0'", 'password = f"Zx9QQw7wEr5Ty3Ui1Op0"', "auth = b'Zx9QQw7wEr5Ty3Ui1Op0'",
                 '"api_key": "Zx9QQw7wEr5Ty3Ui1Op0"', "{token: 'Zx9QQw7wEr5Ty3Ui1Op0'}",
                 'secret := "Zx9QQw7wEr5Ty3Ui1Op0"', "'api_key' => 'Zx9QQw7wEr5Ty3Ui1Op0'",      # Go; Ruby and PHP
                 "{'token': 'Zx9QQw7wEr5Ty3Ui1Op0'}"]:
        assert guard.find_secrets(text, quoted=True) == guard.find_secrets(text) != [], text
    assert guard.find_secrets("api_key = Zx9QQw7wEr5Ty3Ui1Op0") != []                  # elsewhere: as it was
    for text in ["API_KEY=Zx9QQw7wEr5Ty3Ui1Op0", "token: Zx9QQw7wEr5Ty3Ui1Op0"]:         # .env, YAML: bare values
        assert guard.redact(text)[0].endswith("[REDACTED:assigned-secret]"), text     # what leaves is redacted so
    # the entropy floor holds at 3.5 bits a character: 8 characters once and 4 twice in 16 is exactly 3.5
    assert guard.find_secrets("api_key = abcdefgh11223344") and not guard.find_secrets("api_key = abcdefg111223344")
    for name, quoted in [("a.py", True), ("b.go", True), ("c.ts", True), ("d.java", True), ("e.rs", True),
                         ("f.cpp", True), ("g.rb", True), ("h.php", True), ("i.swift", True), ("j.kt", True),
                         ("k.sh", False), (".env", False), ("l.yaml", False), ("m.toml", False), ("n.ini", False),
                         ("o.md", False), ("p.txt", False), ("q.ipynb", False), ("r.json", False), ("s.sql", False),
                         ("T.PY", True), ("u", False)]:
        assert guard.quoted(name) is quoted, name


def test_the_fingerprint_follows_what_finds_secrets(monkeypatch):
    """The index re-judges every file when it changes: it must change with any pattern or with gitleaks' rules."""
    now = guard.fingerprint()
    assert now == guard.fingerprint() and now.startswith(f"v{guard.SECRETS_VERSION}-")
    monkeypatch.setitem(guard.SECRET_PATTERNS, "new-kind", r"new_[a-z]{20}")
    assert guard.fingerprint() != now
    monkeypatch.delitem(guard.SECRET_PATTERNS, "new-kind")
    monkeypatch.setattr(secret_rules, "SHA256", "another release")
    assert guard.fingerprint() != now
    monkeypatch.undo()
    monkeypatch.setattr(guard, "SECRETS_VERSION", guard.SECRETS_VERSION + 1)
    assert guard.fingerprint() != now


def test_denied_paths_by_name_not_by_topic():
    for p in ["~/.ssh/id_ed25519", "/x/.env", "/x/.env.local", "/x/prod.env", "/x/.kaggle/access_token",
              "/x/.modal.toml", "/x/.cache/huggingface/token", "/x/.config/gh/hosts.yml", "/x/key.pem",
              "/x/credentials.json", "/x/.aws/config", "/x/secrets.yaml"]:
        assert guard.denied(Path(p)), p
    for p in ["/notes/tokenizer.md", "/notes/tokens-and-bpe.md", "/notes/secrets-of-attention.md",
              "/notes/credentials-howto.md", "/notes/environment.md"]:
        assert not guard.denied(Path(p)), p


def test_allowed_resolves_symlinks(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir(), outside.mkdir()
    (root / "a.md").write_text("x")
    (outside / "b.md").write_text("y")
    os.symlink(outside / "b.md", root / "link.md")
    os.symlink(tmp_path / "outside", root / "dir")
    assert guard.allowed(root / "a.md", [root])
    assert not guard.allowed(root / "link.md", [root])
    assert not guard.allowed(root / "dir" / "b.md", [root])
    (outside / ".env").write_text("z")
    os.symlink(outside / ".env", root / "harmless.md")        # an innocent name pointing at a denied file
    assert not guard.allowed(root / "harmless.md", [root, outside])


def test_outbound_redacts_and_logs_without_content(home):
    payload = "Question?\nExcerpt: " + CANARIES["openai-key"] + " and " + CANARIES["huggingface-token"]
    sent = guard.outbound(payload, "model")
    assert CANARIES["openai-key"] not in sent and CANARIES["huggingface-token"] not in sent
    log = (home / "audit.jsonl").read_text()
    assert json.loads(log.splitlines()[-1])["redacted"] == 2
    assert "Canary" not in log and "Excerpt" not in log


def test_queries_with_secrets_are_refused(home):
    with pytest.raises(guard.Refused):
        guard.outbound_query("search " + CANARIES["github-token"], "web_search")
    assert guard.outbound_query("bpe tokenizer vietnamese", "web_search") == "bpe tokenizer vietnamese"
    assert "Canary" not in (home / "audit.jsonl").read_text()


def test_sanitize_strips_terminal_control_sequences():
    evil = "ok \x1b[2J\x1b[H\x1b]0;fake title\x07\x1b[31mred\x1b[0m\x07\x08 done\r\n\ttab và tiếng Việt\x9b1m"
    out = guard.sanitize(evil)
    assert out == "ok red done\n\ttab và tiếng Việt1m"
    assert not re.search(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", out)


def test_roots_are_stored_outside_them(home, tmp_path):
    guard.set_roots([tmp_path / "a", tmp_path / "b", tmp_path / "a"])
    assert guard.roots() == [(tmp_path / "a").resolve(), (tmp_path / "b").resolve()]
    assert guard.config_path().parent == home


# --- properties: what must hold for any text and any set of folders ----------------------------------------------

from hypothesis import example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

filler = st.text(alphabet="abc xyz 0123 học máy\n\t.,:=-_/'\"", max_size=30)
# A secret with a distinctive prefix is found even glued to letters or to another secret; one with a short, generic
# prefix (ak-, ws-) only after a non-alphanumeric character, so that ordinary words are not taken for secrets.
GENERIC = {CANARIES["modal-token"], CANARIES["modal-proxy-token"]}
leaky = st.lists(st.one_of(filler, st.sampled_from(sorted(CANARIES.values()))), max_size=8).map(
    lambda parts: "".join((" " + x) if x in GENERIC else x for x in parts))


@given(leaky)
@settings(max_examples=300)
@example(CANARIES["github-token"] + CANARIES["openai-key"])            # two secrets back to back
@example("abc" + CANARIES["google-api-key"] + CANARIES["google-api-key"])   # glued to letters and to each other
@example("api_key = " + CANARIES["openai-key"])                         # one secret inside another's pattern
@example("AKIA0000000000000000 0000000000")          # ordinary text holding a piece of the secret (seen in Docker)
def test_redacted_text_holds_no_secret(text):
    clean, n = guard.redact(text)
    assert guard.find_secrets(clean) == []
    assert (n == 0) == (clean == text)
    # exactly the text with each found secret's stretch (overlapping ones as one) replaced: nothing of a secret left,
    # and ordinary text that happens to hold a piece of one kept (a test of pieces failed on "AKIA0000… 0000000000")
    spans = []
    for kind, a, b in guard.find_secrets(text):
        if spans and a < spans[-1][2]:
            spans[-1][2] = max(spans[-1][2], b)
        else:
            spans.append([kind, a, b])
    want, last = "", 0
    for kind, a, b in spans:
        want, last = want + text[last:a] + f"[REDACTED:{kind}]", b
    assert clean == want + text[last:]


@given(st.text(max_size=60))
@example("ok \x1b[2J\x1b[H\x1b]0;fake title\x07\x1b[31mred\x1b[0m\x07\x08 done\r\n\ttab và tiếng Việt\x9b1m")
def test_sanitized_text_has_no_control_characters_and_keeps_the_rest(text):
    out = guard.sanitize(text)
    assert not re.search(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", out)
    if not re.search(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", text):
        assert out == text


@given(st.lists(st.lists(st.sampled_from("abc"), max_size=3).map("/".join), max_size=6))
def test_outermost_keeps_one_folder_per_tree_in_order(tmp_path_factory, rels):
    base = tmp_path_factory.mktemp("roots")
    roots = [base / r if r else base for r in rels]
    out = guard.outermost(roots)
    assert all(r in roots for r in out) and out == list(dict.fromkeys(r for r in roots if r in out))   # in order
    assert not any(o != p and o in p.parents for o in out for p in out)            # none inside another
    assert len(set(out)) == len(out)
    assert all(any(o == r or o in r.parents for o in out) for r in roots)          # every folder still covered


def test_everything_that_leaves_is_logged_with_counts_only(home):
    def last():
        entry = json.loads((home / "audit.jsonl").read_text().splitlines()[-1])
        assert isinstance(entry.pop("time"), float)
        return entry
    guard.outbound("hello " + CANARIES["openai-key"], "model")
    assert last() == {"event": "outbound", "channel": "model", "chars": len("hello [REDACTED:openai-key]"), "redacted": 1}
    guard.outbound_query("bpe tokenizer", "web_search")
    assert last() == {"event": "outbound", "channel": "web_search", "chars": 13, "redacted": 0}
    with pytest.raises(guard.Refused, match="web_search: the text looks like it contains a secret"):
        guard.outbound_query("x " + CANARIES["github-token"], "web_search")
    assert last() == {"event": "refused", "channel": "web_search"}


def test_the_home_and_the_declared_folders(tmp_path, monkeypatch):
    home = tmp_path / "not" / "yet" / "made"
    monkeypatch.setenv("PERESEARCH_HOME", str(home))
    assert guard.home() == home and guard.config_path() == home / "config.json" and guard.roots() == []
    guard.audit("first")                                         # the first write makes the home
    assert (home / "audit.jsonl").exists()
    guard.set_roots([tmp_path / "b", tmp_path / "a" / ".." / "a", tmp_path / "b"])
    assert json.loads(guard.config_path().read_text()) == {"roots": [str(tmp_path / "a"), str(tmp_path / "b")]}
    guard.set_roots([tmp_path / "b", tmp_path / "a"])            # saved again over the first
    assert guard.roots() == [tmp_path / "a", tmp_path / "b"]
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "n.md").write_text("x")
    assert guard.allowed(tmp_path / "a" / "n.md") and not guard.allowed(tmp_path / "c.md")   # the config's folders
    monkeypatch.delenv("PERESEARCH_HOME")
    assert guard.home() == Path.home() / ".local/share/peresearch"


# --- the deny list: the same verdicts as matching each pattern, fast enough for a walk -------------------------------

import fnmatch  # noqa: E402
import time  # noqa: E402

PART = st.one_of(st.sampled_from(["id_rsa", "id_rsa.pub", ".ENV", ".env.local", "prod.env", "server.PEM", "token",
                                  "tokenizer.md", "credentials.json", "Secrets.yml", ".ssh", "notes", "a.key.md",
                                  "hosts.yml", ".modal.toml", "x.kdbx", "[a]", "*", "?"]),
                 st.text(alphabet="abcdefghijklmnopqrstuvwxyz.-_*?[]!ĐđÉ" + "İ", max_size=12).filter(bool))


@given(st.lists(PART, min_size=1, max_size=6))
@settings(max_examples=400)
def test_a_path_is_denied_exactly_when_one_of_its_parts_matches_a_pattern(parts):
    path = Path("/", *parts)
    want = any(fnmatch.fnmatchcase(part.lower(), pat) for part in path.parts for pat in guard.DENY)
    assert guard.denied(path) is want


def test_the_deny_check_is_fast_enough_to_walk_a_large_tree():
    path = Path("/home/user/projects/some-project/src/package/module/deep/folder/file_name.py")
    start = time.perf_counter()
    for _ in range(10_000):
        guard.denied(path)
    assert time.perf_counter() - start < 0.5          # was 1.4 s: every part matched against every pattern


# --- gitleaks' rules: the formats of hundreds of providers ------------------------------------------------------

from hypothesis import HealthCheck, assume  # noqa: E402

from peresearch import secret_rules  # noqa: E402

RULES = secret_rules.rules()


@pytest.mark.parametrize("go, hits, misses", [
    ("(?i)abc", ["ABC", "abc"], ["abd"]),                              # at the start: as Python has it
    ("x(?i)abc", ["xABC", "xabc"], ["XABC"]),                          # mid-pattern: from there to the group's end
    ("(a(?i)b)c", ["aBc", "abc"], ["aBC", "Abc"]),                     # ... and no further than its group
    ("a|(?i)b|c", ["a", "B", "C"], ["A"]),                             # every alternative after it
    ("[[:alnum:]]+$", ["a1Z"], ["_", "a-1"]),                          # a POSIX class: Python reads it otherwise
    ("[^[:space:]]{3}", ["abc"], ["a c"]),
    ("ab\\z", ["ab"], ["ab\n"]),                                         # the end of the text
    ("[(?i)]x", ["(x", "ix"], ["Ix"]),                                 # inside a class: characters, not flags
    ("\\(?i\\)x", ["(i)x", "i)x"], ["(I)x"]),                         # an escaped "(", made optional: no flag
    ("(ab\\z|c)", ["ab", "c"], ["ab\n"]),                                 # \z anywhere, not only at the end
    ("[]a]+", ["]a", "a]"], ["b", "X"]), ("[^]a]+", ["bc"], ["]", "a"]),      # "]" first in a class is a character
    ("[\\]x]+", ["]x"], ["y"]), ("[^\\d]+", ["ab"], ["a1"]),             # escapes inside a class kept whole
    ("[a[b]+", ["[a", "b"], ["c"]),                                      # a "[" inside a class: a character
    ("[[:digit:]x]+", ["1x"], ["y"]),                                    # a POSIX class among other characters
    ("a(?i)b(?s)c.d", ["aBC\nd", "abcxd"], ["ABcxd"]),                  # two flags in one group: both scoped
])
def test_go_patterns_mean_the_same_in_python(go, hits, misses):
    """gitleaks writes Go regular expressions: 22 rules set (?i) mid-pattern (Python refuses it), one uses a POSIX
    class (Python reads [[:alnum:]] as a set of ':' and letters, silently), four end with \\z."""
    rx = re.compile(secret_rules.python_pattern(go))
    assert all(rx.fullmatch(h) for h in hits) and not any(rx.fullmatch(m) for m in misses)


def test_an_ungreedy_flag_is_refused_not_mistranslated():
    with pytest.raises(ValueError, match="ungreedy flag"):
        secret_rules.python_pattern("a(?U)b+")


def test_the_rules_are_gitleaks_pinned_release_and_all_of_them_load():
    """v8.30.1 as fetched (its sha256); every rule with a pattern compiles, but the generic one (a key name and any
    value: the assigned-secret check covers it with an entropy floor) and those that match a path only."""
    import hashlib
    raw = (Path(secret_rules.__file__).parent / "data" / "gitleaks.toml").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == secret_rules.SHA256 == \
        "e163e53b9e7e8a8511e77271e2b323ed057759542a6d988258afe3a1fa329caf"
    assert "MIT License" in (Path(secret_rules.__file__).parent / "data" / "GITLEAKS-LICENSE").read_text()
    ids = [r.id for r in RULES]
    assert len(ids) == 220 and "generic-api-key" not in ids and len(set(ids)) == len(ids)
    # read exactly as the file says (built again here: the cached set was made at import)
    import tomllib
    config = tomllib.loads(raw.decode())
    fresh = secret_rules.rules.__wrapped__()
    want = [r for r in config["rules"] if "regex" in r and r["id"] not in secret_rules.SKIPPED]

    def allows(entries):
        return [([x.pattern for x in a.regexes], a.target, a.stopwords, a.condition, a.paths) for a in entries]

    def expected(entries):
        return [([secret_rules.python_pattern(x) for x in e.get("regexes", [])], e.get("regexTarget", "secret"),
                 [w.lower() for w in e.get("stopwords", [])], e.get("condition", "OR"), bool(e.get("paths")))
                for e in entries]
    assert [(r.id, r.pattern, r.keywords, r.entropy, r.group, allows(r.allows)) for r in fresh] == [
        (r["id"], secret_rules.python_pattern(r["regex"]), [k.lower() for k in r.get("keywords", [])],
         r.get("entropy"), r.get("secretGroup"), expected(r.get("allowlists", []))) for r in want]
    assert allows([secret_rules._global.__wrapped__()]) == expected([config["allowlist"]])
    assert sum(bool(r.allows) for r in fresh) == 8 and any(a.regexes for r in fresh for a in r.allows)


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
@given(data=st.data())
@settings(max_examples=8, deadline=None, suppress_health_check=list(HealthCheck))
def test_every_rule_finds_what_it_describes_and_it_is_redacted(rule, data):
    """A string the rule's own pattern matches, in a note naming one of its keywords (as gitleaks, a rule applies
    only then), random enough for its entropy floor and not on an allowlist, is found where the rule says (its
    secret: the group it names, else the first that matched) as gitleaks:<id>, and redaction covers it."""
    sample = data.draw(st.from_regex(rule.pattern, fullmatch=True))
    text = f"some notes on {rule.keywords[0] if rule.keywords else 'keys'}\n{sample}\nmore notes"
    found = rule.find(text)
    assume(found)                                        # under its entropy floor, or allowlisted: not a secret
    hits = guard.find_secrets(text)
    a, b = found[0]
    assert (f"gitleaks:{rule.id}", a, b) in hits
    clean, n = guard.redact(text)            # each found stretch replaced: test_redacted_text_holds_no_secret
    assert n >= 1 and clean != text


def test_a_rules_allowlist_and_the_global_one_drop_a_finding():
    """gitleaks' allowlists: a rule's own (regexes on the secret, the match or the line; stopwords; AND or OR) and the
    global one drop what they name; a path condition cannot hold for text, so an AND with one never drops."""
    r = secret_rules.Rule("t", r"key_([a-z0-9]{12})", ["key_"], None, 1, [
        secret_rules.Allow(regexes=[re.compile("^0+$")], target="secret", stopwords=["example"], condition="OR"),
        secret_rules.Allow(regexes=[re.compile("TESTLINE")], target="line", stopwords=[], condition="OR"),
        secret_rules.Allow(regexes=[re.compile("a1")], target="secret", stopwords=[], condition="AND", paths=True)])
    assert r.find("key_a1b2c3d4e5f6") == [(4, 16)]
    assert r.find("key_000000000000") == [] and r.find("key_myexample123") == []
    two = "x key_a1b2c3d4e5f6 TESTLINE\nkey_a1b2c3d4e5f7"                    # the first one's line is allowlisted
    assert r.find(two) == [(two.index("key_a1b2c3d4e5f7") + 4, len(two))]
    assert secret_rules.Rule("t", r"key_([a-z0-9]{12})", ["nothere"], None, 1, []).find("key_a1b2c3d4e5f6") == []
    assert secret_rules.Rule("t", r"key_([a-z0-9]{12})", ["key_"], 4.5, 1, []).find("key_aaaaaaaaaaaa") == []
    # the secret: the group named, else the first that matched; skipped at its entropy floor, not only under it
    first_empty = secret_rules.Rule("t", r"(q1)?(ab\d+)", [], None, None, [])    # ("x" alone: the global list
    assert first_empty.find("ab12") == [(0, 4)] and first_empty.find("q1ab12") == [(0, 2)]   # drops one letter repeated)
    assert (secret_rules.entropy("aabb"), secret_rules.entropy("abcd"), secret_rules.entropy("aaaa")) == (1.0, 2.0, 0.0)
    floor = secret_rules.entropy("ab12cd34")
    assert secret_rules.Rule("t", r"k=(\w+)", [], floor, None, []).find("k=ab12cd34") == []
    assert secret_rules.Rule("t", r"k=(\w+)", [], floor - 0.01, None, []).find("k=ab12cd34") == [(2, 10)]
    many = "k=aaaaaaaa k=ab12cd34ef k=bbbbbbbb k=q9w8e7r6"                 # a low one does not end the search
    assert secret_rules.Rule("t", r"k=(\w+)", [], 2.0, None, []).find(many) == [(13, 23), (37, 45)]
    # the line an allowlist reads is the match's own, whole, without its neighbours
    on_line = secret_rules.Rule("t", r"key_([a-z0-9]{12})", [], None, 1, [
        secret_rules.Allow(regexes=[re.compile(r"^skip .*!$")], target="line")])
    text = "first key_a1b2c3d4e5f6\nskip key_a1b2c3d4e5f7 !\nlast key_a1b2c3d4e5f8"
    assert [text[a:b] for a, b in on_line.find(text)] == ["a1b2c3d4e5f6", "a1b2c3d4e5f8"]
    assert on_line.find("skip key_a1b2c3d4e5f7 !") == [] and on_line.find("x\nskip key_a1b2c3d4e5f7 !") == []
    near = secret_rules.Rule("t", r"key_([a-z0-9]{12})", [], None, 1, [
        secret_rules.Allow(regexes=[re.compile("skip")], target="line")])      # its own line only, not a neighbour's
    assert near.find("a\nskip\nkey_a1b2c3d4e5f6") == [(11, 23)] and near.find("key_a1b2c3d4e5f6\nskip\nb") == [(4, 16)]
    # AND: every condition given must hold; one given alone decides
    only_words = secret_rules.Allow(stopwords=["test"], condition="AND")
    assert only_words.drops("mytest", "", "") and not only_words.drops("real", "", "")
    only_regex = secret_rules.Allow(regexes=[re.compile("^0+$")], condition="AND")
    assert only_regex.drops("000", "", "") and not only_regex.drops("001", "", "")


def test_every_file_peresearch_writes_is_the_users_alone(home, tmp_path):
    """Step 3f (2026-10-08): the audit log, the index (with SQLite's -wal and -shm beside it) and the web cache are
    readable by the user alone (0600) in folders only the user may enter (0700) — not only through the home folder's
    0700, which a user may loosen."""
    import os
    import stat
    import httpx
    from peresearch.web import Web
    from peresearch.zetokrag.index import Index
    from tests.conftest import FakeEmbedder
    mode = lambda p: stat.S_IMODE(os.stat(p).st_mode)                                      # noqa: E731
    guard.private_folder(home)
    (home / "audit.jsonl").write_text("")
    os.chmod(home / "audit.jsonl", 0o644)                        # a log made before 3f, readable by others
    guard.audit("test", n=1)
    assert mode(home / "audit.jsonl") == 0o600
    notes = tmp_path / "n"
    notes.mkdir()
    (notes / "a.md").write_text("# A\n\nalpha\n")
    idx = Index(home / "index", embedder=FakeEmbedder())
    idx.update([notes])
    files = [p for p in (home / "index").iterdir() if p.name.startswith("index.sqlite")]
    assert {p.name for p in files} >= {"index.sqlite"} and all(mode(p) == 0o600 for p in files), \
        {p.name: oct(mode(p)) for p in files}
    assert mode(home / "index") == 0o700
    page = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="page words",
                                                                               headers={"content-type": "text/plain"})))
    Web(None, http=page, cache=home / "web", pin=False).fetch("https://example.org/p")
    cached = list((home / "web").iterdir())
    assert cached and all(mode(p) == 0o600 for p in cached) and mode(home / "web") == 0o700
