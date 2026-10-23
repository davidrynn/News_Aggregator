"""Calls to the language model provider. Only Ollama is implemented in phase 1.

Keeping every model call in this one module is what lets phase 2 switch providers
with a config change: callers ask for embed() and never see which service answers.
"""
import httpx

from newsagg.config import Config


def embed(texts: list[str], cfg: Config, batch_size: int = 100) -> list[list[float]]:
    """Return one embedding vector per text, in order."""
    if cfg.provider.name != "ollama":
        raise NotImplementedError(f"provider {cfg.provider.name!r} has no embed() yet")
    vectors: list[list[float]] = []
    with httpx.Client(base_url=cfg.provider.url, timeout=120) as client:
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


def client(cfg: Config) -> httpx.Client:
    # Loading the 5 GB model from disk took 122 s on the first call, so reads get 5 minutes.
    return httpx.Client(base_url=cfg.provider.url, timeout=httpx.Timeout(30, read=300))
