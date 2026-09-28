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


def test_ordinary_text_is_not_mistaken_for_a_secret():
    for text in ["token = tokenizer_config", "the password field must be hashed", "BPE merges ab and cd",
                 "học máy và xử lý ngôn ngữ tự nhiên", "sk-learn is a typo for scikit-learn", "key: value"]:
        assert guard.find_secrets(text) == [], text


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
