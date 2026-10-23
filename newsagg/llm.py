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
