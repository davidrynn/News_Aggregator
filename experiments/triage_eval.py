"""Experiment: can a headline-only first pass choose what to score better than outlet count?

Run from the project root: PYTHONPATH=. .venv/Scripts/python experiments/triage_eval.py
Changes nothing in the pipeline. Results of the first run (2026-10-24) are in the spec.
"""
import json
import random
import sys
import time
from typing import Literal

from pydantic import BaseModel, Field

from newsagg import config, llm

SYSTEM = """You triage news headlines for a news page that keeps only newsworthy, original journalism about world events, US news, science, technology, health, climate and business.

For each numbered headline, rate how likely the full article is worth a closer look:
5 = clearly significant news (major events, decisions, discoveries, disasters, conflicts)
4 = probably real news of broad interest
3 = possibly news; unclear from the headline
2 = probably not: local or minor items, routine announcements, sports, entertainment, lifestyle, product deals
1 = clearly not: opinion, advice, listicles, clickbait, quizzes, horoscopes, press releases

Judge the event described, not the writing style. Rate every headline. Return JSON."""


class Rating(BaseModel):
    n: int
    score: int = Field(ge=1, le=5)


class Ratings(BaseModel):
    ratings: list[Rating]


SCHEMA = Ratings.model_json_schema()


def title_of(user_message: str) -> str:
    return user_message.split("Title: ", 1)[1].split("\n", 1)[0].strip()


def rate(titles: list[str], cfg, client, batch: int = 50) -> list[int]:
    scores = []
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        user = "\n".join(f"{n}. {t}" for n, t in enumerate(chunk, 1))
        got = {}
        try:
            raw = llm.chat_json(SYSTEM, user, SCHEMA, cfg, client)
            got = {r.n: r.score for r in Ratings.model_validate_json(raw).ratings}
        except Exception as e:
            print(f"  batch {i // batch} failed: {e}", file=sys.stderr)
        missing = sum(1 for n in range(1, len(chunk) + 1) if n not in got)
        if missing:
            print(f"  batch {i // batch}: {missing} of {len(chunk)} unrated (counted as 3)", file=sys.stderr)
        scores += [got.get(n, 3) for n in range(1, len(chunk) + 1)]
    return scores


def recall_at(order_key, items, frac: float) -> float:
    """Share of the keeps that land in the top `frac` of items when sorted by order_key."""
    ranked = sorted(items, key=order_key, reverse=True)
    top = ranked[: int(len(ranked) * frac)]
    keeps = sum(x["keep"] for x in items)
    return sum(x["keep"] for x in top) / keeps if keeps else 0.0


def report(name: str, items: list[dict]) -> None:
    rnd = random.Random(0)
    for x in items:
        x["tiebreak"] = rnd.random()     # ties broken randomly, the same way for every method
    keep_rate = sum(x["keep"] for x in items) / len(items)
    print(f"\n{name}: {len(items)} stories, {keep_rate:.0%} kept by the answer key")
    print("  share of keeps caught if we score only the top ...   25%    50%    75%")
    methods = {"random (no information)": lambda x: x["tiebreak"],
               "headline triage": lambda x: (x["triage"], x["tiebreak"])}
    if "outlets" in items[0]:
        methods["outlet count (today)"] = lambda x: (x["outlets"], x["tiebreak"])
        methods["triage, then outlets"] = lambda x: (x["triage"], x["outlets"], x["tiebreak"])
    for label, key in methods.items():
        print(f"  {label:44} " + "  ".join(f"{recall_at(key, items, f):5.0%}" for f in (0.25, 0.5, 0.75)))
    by_score = {s: [x["keep"] for x in items if x["triage"] == s] for s in range(1, 6)}
    print("  keep rate by triage score: " + ", ".join(
        f"{s}: {sum(v) / len(v):.0%} of {len(v)}" for s, v in by_score.items() if v))


def main() -> None:
    cfg = config.load()
    rows = [json.loads(line) for line in open("data/verdicts.jsonl", encoding="utf-8")]
    seen, verdicts = set(), []
    for r in reversed(rows):                         # newest first, one per headline
        t = title_of(r["user_message"])
        if t not in seen:
            seen.add(t)
            verdicts.append({"title": t, "outlets": r["outlets"], "keep": r["kept"]})
    verdicts = verdicts[:1000]
    golden = []
    for line in open("tests/golden.jsonl", encoding="utf-8"):
        g = json.loads(line)
        keep = (g["kind"] in cfg.scoring.keep_kinds and g["newsworthiness"] >= cfg.scoring.min_newsworthiness
                and g["relevance"] >= cfg.scoring.min_relevance)
        golden.append({"title": title_of(g["user_message"]), "keep": keep})

    start = time.monotonic()
    with llm.client(cfg) as client:
        for items in (verdicts, golden):
            for x, s in zip(items, rate([x["title"] for x in items], cfg, client)):
                x["triage"] = s
    elapsed = time.monotonic() - start
    n = len(verdicts) + len(golden)
    print(f"rated {n} headlines in {elapsed:.0f}s ({n / elapsed:.1f} per second)")
    report("Answer key 1: full scoring by the model (verdicts.jsonl)", verdicts)
    report("Answer key 2: Fable's golden labels", golden)


if __name__ == "__main__":
    main()
