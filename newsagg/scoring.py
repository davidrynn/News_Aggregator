"""Model scoring: each story gets a fixed JSON verdict, several stories per call.

The system prompt lives in prompts/scorer_system.txt, copied verbatim from the spec.
It is sent byte-identical on every call so the provider can reuse its cached prefix;
everything that varies goes in the user message. Batching lives there too: Topics and
Regions once, then numbered articles (experiments/batch_eval.py, 2026-10-26: batches of 5
matched or beat one story per call on the golden set for the larger cloud models).
"""
import json
import logging
from pathlib import Path
from typing import Literal, get_args

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from newsagg import llm
from newsagg.config import Config
from newsagg.enrich import PageInfo
from newsagg.gdelt import Article

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "scorer_system.txt").read_text(encoding="utf-8")


class Verdict(BaseModel):
    kind: Literal["original_reporting", "wire_copy", "opinion", "aggregation", "press_release", "clickbait", "lifestyle"]
    newsworthiness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    topic: Literal["world", "us", "politics", "business", "science", "tech", "health", "climate", "arts", "other"]
    reason: str

    @field_validator("topic", mode="before")
    @classmethod
    def unlisted_topic_is_other(cls, v):
        # Without schema-forced output a model may answer "sports" or "entertainment". Topic
        # only places a story on the page (the keep rule ignores it), so keep the verdict.
        return v if v in get_args(cls.model_fields["topic"].annotation) else "other"


SCHEMA = Verdict.model_json_schema()


class NumberedVerdict(Verdict):
    n: int


class Verdicts(BaseModel):
    verdicts: list[NumberedVerdict]


BATCH_SCHEMA = Verdicts.model_json_schema()


def user_message(article: Article, info: PageInfo, cfg: Config) -> str:
    """The spec's user message template. The outlet is left out on purpose so the model
    cannot judge by name."""
    return (
        f"Topics: {', '.join(cfg.topics.include)}\n"
        f"Regions: {', '.join(cfg.topics.regions)}\n"
        f"\n"
        f"Title: {article.title}\n"
        f"Description: {info.description or '(none)'}\n"
        f"Byline present: {'yes' if info.has_byline else 'no'}\n"
        f"Page type: {', '.join(info.ld_types) or 'unknown'}\n"
        f"Section: {info.section or 'unknown'}"
    )


def batch_message(items: list[tuple[Article, PageInfo]], cfg: Config) -> str:
    """Topics and Regions once, then each article's lines from user_message(), numbered."""
    messages = [user_message(a, i, cfg).splitlines() for a, i in items]
    header = "\n".join(messages[0][:2])
    return (f"{header}\n\nScore each of the {len(items)} numbered articles below separately, "
            f"one verdict per number.\n\n"
            + "\n\n".join(f"Article {n}\n" + "\n".join(m[3:]) for n, m in enumerate(messages, 1)))


def score_batch(items: list[tuple[Article, PageInfo]], cfg: Config, client: httpx.Client) -> list[Verdict | None]:
    """One verdict per item, in order; None where the call failed or that verdict was missing
    or invalid. Each verdict is checked on its own, so one bad one loses one story, not the batch."""
    if len(items) == 1:
        return [score(*items[0], cfg, client)]
    got: dict[int, Verdict] = {}
    try:
        raw = llm.chat_json(SYSTEM_PROMPT, batch_message(items, cfg), BATCH_SCHEMA, cfg, client)
        for item in json.loads(raw).get("verdicts", []):
            try:
                v = NumberedVerdict.model_validate(item)
                got[v.n] = Verdict(**v.model_dump(exclude={"n"}))
            except ValidationError as e:
                log.warning("one verdict in a batch was invalid: %s", e)
    except (httpx.HTTPError, ValueError) as e:     # ValueError covers a reply that isn't JSON
        log.warning("scoring batch of %d failed: %s", len(items), e)
    return [got.get(n) for n in range(1, len(items) + 1)]


def score(article: Article, info: PageInfo, cfg: Config, client: httpx.Client) -> Verdict | None:
    """Return the model's verdict, or None if the call or validation fails."""
    try:
        raw = llm.chat_json(SYSTEM_PROMPT, user_message(article, info, cfg), SCHEMA, cfg, client)
        return Verdict.model_validate_json(raw)
    except (httpx.HTTPError, ValidationError) as e:
        log.warning("scoring failed for %s: %s", article.url, e)
        return None


def keep(v: Verdict, cfg: Config) -> bool:
    """The keep rule. Thresholds are 3 of 5 on purpose; don't raise without asking the owner."""
    return (v.kind in cfg.scoring.keep_kinds
            and v.newsworthiness >= cfg.scoring.min_newsworthiness
            and v.relevance >= cfg.scoring.min_relevance)
