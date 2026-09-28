"""Where the model and the web search live, and their keys.

Each setting is read from the environment, else from `$PERESEARCH_HOME/settings.env` (`NAME=value` lines; keep
the file `chmod 600`). Its name matches the guard's deny list, so it is never indexed, and keys only ever travel
in the request headers of their own service — never inside a prompt or a search query.

    PERESEARCH_LLM_URL    OpenAI-compatible base URL, e.g. https://<workspace>--peresearch-llm-server.modal.run/v1
    PERESEARCH_LLM_MODEL  the served model name (default "llm", as deploy/modal_vllm.py serves it)
    PERESEARCH_LLM_KEY    bearer token: a Modal proxy token "wk-….ws-…", or a provider's API key
    TAVILY_API_KEY        first web search provider (free tier)
    EXA_API_KEY           second web search provider, used when Tavily's credits run out
    JINA_API_KEY          optional; the page reader works without it at a lower rate
"""

import os

from peresearch import guard

DEFAULTS = {"PERESEARCH_LLM_MODEL": "llm"}


def _file() -> dict[str, str]:
    path = guard.home() / "settings.env"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def get(name: str) -> str:
    return os.environ.get(name) or _file().get(name) or DEFAULTS.get(name, "")
