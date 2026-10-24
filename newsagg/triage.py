"""Headline triage: a cheap first pass that rates each story's headline 1-5 in batches.

Only stories rated min_triage or higher go on to page fetching and full scoring. Tested
2026-10-24 (experiments/triage_eval.py): it caught 66% of keeps in the top half against
45% for outlet count, and on the golden set no headline rated 1-2 was a keep.
"""
import logging
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError

from newsagg import llm
from newsagg.config import Config

log = logging.getLogger(__name__)

# Stripped so the prompt is byte-identical to the one tested in the experiment.
SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "triage_system.txt").read_text(encoding="utf-8").strip()
BATCH = 50          # headlines per call, as tested
UNRATED = 3         # a skipped headline or failed batch passes the default cutoff: never drop news on an error


class Rating(BaseModel):
    n: int
    score: int = Field(ge=1, le=5)


class Ratings(BaseModel):
    ratings: list[Rating]


SCHEMA = Ratings.model_json_schema()


def user_message(titles: list[str]) -> str:
    return "\n".join(f"{n}. {t}" for n, t in enumerate(titles, 1))


def rate(titles: list[str], cfg: Config, client: httpx.Client) -> list[int]:
    """Return one 1-5 rating per title, in order."""
    scores: list[int] = []
    for i in range(0, len(titles), BATCH):
        chunk = titles[i:i + BATCH]
        got: dict[int, int] = {}
        try:
            raw = llm.chat_json(SYSTEM_PROMPT, user_message(chunk), SCHEMA, cfg, client)
            got = {r.n: r.score for r in Ratings.model_validate_json(raw).ratings}
        except (httpx.HTTPError, ValidationError) as e:
            log.warning("triage batch %d failed, all %d rated %d: %s", i // BATCH, len(chunk), UNRATED, e)
        missing = sum(1 for n in range(1, len(chunk) + 1) if n not in got)
        if got and missing:
            log.warning("triage batch %d: %d of %d unrated, rated %d", i // BATCH, missing, len(chunk), UNRATED)
        scores += [got.get(n, UNRATED) for n in range(1, len(chunk) + 1)]
    return scores
