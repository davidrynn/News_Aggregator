"""Experiment: which hosted embedding model can replace local nomic-embed-text for grouping?

Run from the project root: PYTHONPATH=. .venv/Scripts/python experiments/embed_eval.py [--page] [--judge=MODEL]
--page compares headline + description (the page merge's text) instead of headlines. Changes nothing in the pipeline. Ollama must be running (nomic is the reference).

1. Takes N headlines from the current GDELT window, cleaned as grouping does.
2. Embeds them with nomic (local) and each candidate (DeepInfra).
3. Reference: the pairs nomic puts at or above grouping.threshold (0.87) = "same story".
   For each candidate, finds the threshold whose pairs best match (F1), and the one that
   gives as many pairs as nomic's candidate_threshold (0.75) does.
4. Where the best candidates and nomic disagree, the same-event judge (the configured chat
   model, both orders) decides who was right.
"""
import json
import random
import sys
import time

import httpx
import numpy as np

from newsagg import config, gdelt, grouping, llm
from newsagg.output import SAME_EVENT_PROMPT, SAME_EVENT_SCHEMA, SameEvent

N = 3000
JUDGE = 30            # disagreeing pairs judged per side per model
TITLES = config.ROOT / "data" / "embed_eval_titles.json"   # the same headlines on every run
URL = "https://api.deepinfra.com/v1/openai/embeddings"

# Model -> text prefix, as each model's card recommends for symmetric similarity.
CANDIDATES = {
    "BAAI/bge-base-en-v1.5": "",
    "thenlper/gte-base": "",
    "sentence-transformers/all-mpnet-base-v2": "",
    "intfloat/e5-base-v2": "query: ",
    "google/embeddinggemma-300m": "task: clustering | query: ",
    "Qwen/Qwen3-Embedding-0.6B": "Instruct: Identify news headlines that report the same news event\nQuery:",
}


def page_texts() -> list[str]:
    """Headline + description as the page merge embeds them (output.embed_text), from the
    newest scored stories, one per headline."""
    seen, texts = set(), []
    for line in reversed(open(config.ROOT / "data" / "verdicts.jsonl", encoding="utf-8").readlines()):
        msg = json.loads(line)["user_message"]
        title = msg.split("Title: ", 1)[1].split("\n", 1)[0].strip()
        desc = msg.split("Description: ", 1)[1].split("\n", 1)[0].strip() if "Description: " in msg else ""
        if title not in seen:
            seen.add(title)
            texts.append(f"{grouping.clean_title(title)}. {desc[:300]}")
    return texts[:N]


def NOMIC(cfg):
    """The config with local nomic-embed-text as the embedding model: the reference."""
    return cfg.model_copy(update={"provider": cfg.provider.model_copy(update={
        "embed_name": "ollama", "embed_model": "nomic-embed-text", "embed_url": "http://localhost:11434",
        "embed_prefix": "clustering: "})})


def titles(cfg) -> list[str]:
    if "--page" in sys.argv:
        return page_texts()
    if TITLES.exists():
        return json.loads(TITLES.read_text(encoding="utf-8"))
    articles = gdelt.fetch(cfg.feeds.gdelt_lookback, set()).articles   # done-list not saved
    unique = sorted({grouping.clean_title(a.title) for a in articles if len(a.title.split()) >= 4})
    picked = random.Random(0).sample(unique, min(N, len(unique)))
    TITLES.write_text(json.dumps(picked, ensure_ascii=False), encoding="utf-8")
    return picked


def embed_hosted(model: str, texts: list[str], key: str) -> np.ndarray:
    vectors = []
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120) as client:
        for i in range(0, len(texts), 200):
            resp = client.post(URL, json={"model": model, "input": texts[i:i + 200]})
            vectors += [d["embedding"] for d in resp.raise_for_status().json()["data"]]
    return grouping.normalize(vectors)


def upper(v: np.ndarray) -> np.ndarray:
    """Cosine similarity of every pair i < j, flattened."""
    return (v @ v.T)[np.triu_indices(len(v), k=1)]


def best_threshold(sims: np.ndarray, ref: np.ndarray) -> tuple[float, float, float, float]:
    best = (0.0, 0.0, 0.0, 0.0)
    for t in np.arange(0.50, 0.99, 0.005):
        got = sims >= t
        tp = int((got & ref).sum())
        if not tp:
            continue
        p, r = tp / got.sum(), tp / ref.sum()
        f1 = 2 * p * r / (p + r)
        if f1 > best[0]:
            best = (f1, float(t), p, r)
    return best


def judge(pairs: list[tuple[str, str]], cfg, client) -> float:
    """Share of pairs the chat model calls the same event (yes in both orders)."""
    def ask(p):
        a, b = p
        try:
            return SameEvent.model_validate_json(
                llm.chat_json(SAME_EVENT_PROMPT, f"Story A: {a}.\nStory B: {b}.", SAME_EVENT_SCHEMA, cfg, client)).same_event
        except Exception as e:
            print(f"  judge failed: {e}", file=sys.stderr)
            return False
    flat = llm.parallel(ask, [q for a, b in pairs for q in ((a, b), (b, a))], cfg)
    return sum(flat[i] and flat[i + 1] for i in range(0, len(flat), 2)) / max(len(pairs), 1)


def main() -> None:
    cfg = config.load()
    texts = titles(cfg)
    print(f"{len(texts)} headlines")
    idx = np.triu_indices(len(texts), k=1)
    nomic = upper(grouping.normalize(llm.embed(texts, NOMIC(cfg))))
    ref = nomic >= cfg.grouping.threshold
    n_cand = int((nomic >= cfg.grouping.candidate_threshold).sum())
    print(f"nomic: {ref.sum()} pairs >= {cfg.grouping.threshold}, {n_cand} >= {cfg.grouping.candidate_threshold}\n")

    key = llm.api_key("DEEPINFRA_API_KEY")
    results = {}
    print(f"{'model':42} {'best t':>6} {'F1':>5} {'prec':>5} {'recall':>6} {'cand t':>6} {'secs':>5}")
    for model, prefix in CANDIDATES.items():
        start = time.monotonic()
        try:
            sims = upper(embed_hosted(model, [prefix + t for t in texts], key))
        except httpx.HTTPError as e:
            print(f"{model:42} failed: {e}")
            continue
        f1, t, p, r = best_threshold(sims, ref)
        cand_t = float(np.sort(sims)[-n_cand])        # as many pairs as nomic's candidate zone
        results[model] = (f1, t, cand_t, sims)
        print(f"{model:42} {t:6.3f} {f1:5.2f} {p:5.2f} {r:6.2f} {cand_t:6.3f} {time.monotonic() - start:5.0f}")

    print("\nDisagreements judged by the chat model (share it calls the same event):")
    with llm.client(cfg) as client:
        rnd = random.Random(0)
        only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--judge=")]   # judge these instead of the best two
        ranked = sorted(results.items(), key=lambda kv: -kv[1][0])
        for model, (f1, t, _, sims) in [kv for kv in ranked if kv[0] in only] if only else ranked[:2]:
            got = sims >= t
            sides = {"nomic only": np.flatnonzero(ref & ~got), "candidate only": np.flatnonzero(got & ~ref)}
            print(f"  {model} at {t:.3f}:")
            for label, where in sides.items():
                sample = rnd.sample(list(where), min(JUDGE, len(where)))
                pairs = [(texts[idx[0][k]], texts[idx[1][k]]) for k in sample]
                share = judge(pairs, cfg, client) if pairs else 0.0
                print(f"    {label:15} {len(where):5} pairs; judged same: {share:.0%} of {len(pairs)}")
                for a, b in pairs[:4]:
                    print(f"      - {a}\n        {b}")


if __name__ == "__main__":
    main()
