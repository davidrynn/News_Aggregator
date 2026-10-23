"""Model scoring: one call per story returns a fixed JSON verdict.

The system prompt lives in prompts/scorer_system.txt, copied verbatim from the spec.
It is sent byte-identical on every call so Ollama can reuse its cached prefix;
everything that varies goes in the user message.
"""
import logging
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

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
    topic: Literal["world", "us", "politics", "business", "science", "tech", "health", "climate", "other"]
    reason: str


SCHEMA = Verdict.model_json_schema()


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
