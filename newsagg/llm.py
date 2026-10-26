"""Calls to the language model provider: "ollama" (local) or "openai" (any OpenAI-compatible
API: DeepInfra, OpenRouter, Groq, Together, or Ollama's own /v1).

Keeping every model call in this one module is what lets phase 2 switch providers
with a config change: callers ask for embed() and never see which service answers.
"""
import json
import os
import time

import httpx

from newsagg.config import ROOT, Config

RETRIES = 4     # tries per call to a hosted API, waiting 2, 4, 8 s between them


def embed(texts: list[str], cfg: Config, batch_size: int = 100) -> list[list[float]]:
    """Return one embedding vector per text, in order.

    Always local Ollama at provider.embed_url, whichever provider answers chat: no cloud host
    offers nomic-embed-text (checked 2026-10-26), and a different embedding model would need
    the grouping thresholds re-tuned. The 270 MB model is light enough to run alongside work.
    """
    vectors: list[list[float]] = []
    with httpx.Client(base_url=cfg.provider.embed_url, timeout=120) as client:
        for i in range(0, len(texts), batch_size):
            resp = client.post("/api/embed", json={
                "model": cfg.provider.embed_model,
                "input": texts[i:i + batch_size],
            })
            vectors.extend(resp.raise_for_status().json()["embeddings"])
    return vectors


def chat_json(system: str, user: str, schema: dict, cfg: Config, client: httpx.Client) -> str:
    """One chat call whose reply is forced to match `schema` (Ollama structured outputs).

    Returns the raw JSON text; the caller validates it. temperature 0 makes the same
    input give the same verdict, and keep_alive holds the model in GPU memory between calls.
    """
    if cfg.provider.name == "openai":
        if cfg.provider.json_schema:
            response_format = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
        else:
            # Plain JSON mode (DeepInfra's Llama 3.1 8B rejects json_schema): the shape is
            # stated in the user message, the system prompt stays byte-identical, and the
            # caller's validation catches a reply that ignores it.
            user += f"\n\nReply with only a JSON object matching this JSON Schema:\n{json.dumps(schema)}"
            response_format = {"type": "json_object"}
        body = {
            "model": cfg.provider.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": response_format,
            "temperature": 0,
            **cfg.provider.extra,
        }
        # Hosted APIs fail now and then with a rate limit or a server error (Google's free
        # tier returned an instant 500 on about 4 calls in 9, 2026-10-26); those are retried.
        for attempt in range(RETRIES):
            resp = client.post("chat/completions", json=body)
            if resp.status_code != 429 and resp.status_code < 500 or attempt == RETRIES - 1:
                break
            time.sleep(2 ** (attempt + 1))
        return resp.raise_for_status().json()["choices"][0]["message"]["content"]
    if cfg.provider.name != "ollama":
        raise NotImplementedError(f"provider {cfg.provider.name!r} has no chat_json() yet")
    resp = client.post("/api/chat", json={
        "model": cfg.provider.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "format": schema,
        "think": False,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0},
    })
    return resp.raise_for_status().json()["message"]["content"]


def api_key(name: str) -> str:
    """The key from the environment, else from a NAME=value line in the git-ignored .env file."""
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    for line in env.read_text(encoding="utf-8").splitlines() if env.exists() else []:
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip().strip('"')
    raise RuntimeError(f"no API key: set {name} or add {name}=... to {env}")


def client(cfg: Config) -> httpx.Client:
    # Loading the 5 GB model from disk took 122 s on the first call, so reads get 5 minutes.
    timeout = httpx.Timeout(30, read=300)
    if cfg.provider.name == "openai":
        headers = {"Authorization": f"Bearer {api_key(cfg.provider.api_key_env)}"}
        return httpx.Client(base_url=cfg.provider.url.rstrip("/") + "/", headers=headers, timeout=timeout)
    return httpx.Client(base_url=cfg.provider.url, timeout=timeout)
