"""Output: merge this run's kept stories into out/news.json, the contract every front end reads.

news.json holds up to output.max_stories stories from the last output.max_age_hours. Stories
carry over between runs. A new story whose headline matches a published one (same embedding
threshold as grouping) is merged into it instead of becoming a duplicate tile.

Fields starting with "_" (the headline vector, the full outlet list) are kept in
data/published.json for merging and stripped from news.json.
"""
import hashlib
import json
import logging
import math
import os
import re
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError

from newsagg import grouping, llm
from newsagg.config import ROOT, Config
from newsagg.enrich import PageInfo
from newsagg.gdelt import Article
from newsagg.scoring import Verdict

log = logging.getLogger(__name__)

STATE = ROOT / "data" / "published.json"
SAME_CACHE = ROOT / "data" / "same_event.json"   # model answers by story-id pair
SITE = Path(__file__).parent / "site"
MAX_ALSO = 10   # "also reported by" links kept per story
MAX_CHECKS = 3  # closest earlier stories checked when merging a story
SAME_EVENT_PROMPT = (Path(__file__).parent / "prompts" / "same_event.txt").read_text(encoding="utf-8")

# Wire services' internal editing labels: "1st LD Writethru: ...", "UPDATE 2-...", "REFILE-...".
# Only the displayed headline is cleaned; the model still sees the original, where the label
# is a useful sign of wire copy. Other prefixes ("Report:", "Opinion |") carry meaning and stay.
WIRE_SLUG = re.compile(r"^(?:\d+(?:st|nd|rd|th)\s+)?(?:LD[\s-]+)?Writethru\s*:\s*"
                       r"|^(?:UPDATE|REFILE|CORRECTED|WRAPUP)(?:\s+\d+)?\s*-\s*",
                       re.IGNORECASE)


def display_title(title: str) -> str:
    return WIRE_SLUG.sub("", title).strip() or title


def story_id(url: str) -> str:
    """sha1 of the lead URL without query string or fragment."""
    parts = urlsplit(url)
    normalized = urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, "", ""))
    return hashlib.sha1(normalized.encode()).hexdigest()


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def rank(story: dict, now: datetime, coverage_weight: float = 0.5) -> float:
    """newsworthiness x 2 + relevance + outlet bonus + recency + coverage.

    recency is 1 when new and 0 at 24 hours. coverage is coverage_weight x log2(outlets):
    0 for one outlet, 0.5 for two, about 2.6 for 35, so wide coverage breaks ties between
    equally newsworthy stories without outweighing newsworthiness itself.
    """
    age_hours = (now - parse_iso(story["published_at"])).total_seconds() / 3600
    recency = max(0.0, 1 - age_hours / 24)
    coverage = coverage_weight * math.log2(max(story["outlet_count"], 1))
    return round(story["newsworthiness"] * 2 + story["relevance"]
                 + story.get("_outlet_bonus", 0.0) + recency + coverage, 2)


def build_story(members: list[Article], lead: Article, info: PageInfo, verdict: Verdict) -> dict:
    outlets = list(dict.fromkeys(a.domain for a in [lead, *members]))   # unique, lead first
    also, seen = [], {lead.domain}
    for a in members:
        if a.domain not in seen and len(also) < MAX_ALSO:
            also.append({"outlet": a.domain, "url": a.url})
            seen.add(a.domain)
    return {
        "id": story_id(lead.url),
        "title": display_title(lead.title),
        "url": lead.url,
        "outlet": lead.domain,
        "published_at": iso(max(a.seen_at for a in members + [lead])),
        "image": lead.image or info.image,
        "description": info.description,
        "topic": verdict.topic,
        "kind": verdict.kind,
        "score": 0.0,                  # set by rank() when written
        "newsworthiness": verdict.newsworthiness,
        "relevance": verdict.relevance,
        "outlet_count": len(outlets),
        "also_reported_by": also,
        "_outlets": outlets,
    }


def merge_into(old: dict, new: dict) -> dict:
    """Combine a new story with the published one it matches. The higher-newsworthiness
    version supplies the headline and link; outlets are unioned; the time is the latest."""
    base, other = (new, old) if new["newsworthiness"] > old["newsworthiness"] else (old, new)
    merged = dict(base)
    outlets = list(dict.fromkeys(base["_outlets"] + other["_outlets"]))
    other_lead = {"outlet": other["outlet"], "url": other["url"]}
    links = {x["outlet"]: x for x in [other_lead] + other["also_reported_by"] + base["also_reported_by"]}
    links.pop(base["outlet"], None)
    merged.update(
        _outlets=outlets,
        outlet_count=len(outlets),
        also_reported_by=list(links.values())[:MAX_ALSO],
        published_at=max(old["published_at"], new["published_at"]),
        _vector=base["_vector"],
    )
    return merged


def embed_text(story: dict) -> str:
    """Headline plus description: the description's specifics (who, where, how many) separate
    same-event pairs from same-subject pairs better than the headline alone."""
    return f"{grouping.PREFIX}{grouping.clean_title(story['title'])}. {story['description'][:300]}"


class SameEvent(BaseModel):
    same_event: bool


SAME_EVENT_SCHEMA = SameEvent.model_json_schema()


class EventMatcher:
    """Decide whether two stories report the same event.

    Embedding similarity finds candidates cheaply; the model confirms the uncertain ones.
    At or above `threshold` they match outright, below `candidate` they never do, and in
    between the model is asked, once in each order: an 8B model's answer can depend on
    which story comes first, so both must say yes. Answers are cached by id pair.
    """

    def __init__(self, cfg: Config, client: httpx.Client, cache: dict[str, bool]):
        self.cfg, self.client, self.cache = cfg, client, cache
        self.threshold = cfg.grouping.threshold
        self.candidate = cfg.grouping.candidate_threshold
        self.asked = 0

    def __call__(self, a: dict, b: dict, sim: float) -> bool:
        if sim >= self.threshold:
            return True
        if sim < self.candidate:
            return False
        key = "|".join(sorted((a["id"], b["id"])))
        if key not in self.cache:
            try:
                self.cache[key] = self.ask(a, b) and self.ask(b, a)
            except (httpx.HTTPError, ValidationError):
                return False   # unsure: keep them apart, ask again next run
            self.asked += 1
        return self.cache[key]

    def ask(self, a: dict, b: dict) -> bool:
        message = (f"Story A: {a['title']}. {a['description'][:300]}\n"
                   f"Story B: {b['title']}. {b['description'][:300]}")
        raw = llm.chat_json(SAME_EVENT_PROMPT, message, SAME_EVENT_SCHEMA, self.cfg, self.client)
        return SameEvent.model_validate_json(raw).same_event


def merge(stories: list[dict], same: Callable[[dict, dict, float], bool]) -> list[dict]:
    """Fold stories into a list, merging each into the closest earlier story that `same`
    accepts. Up to MAX_CHECKS of the closest earlier stories are tried."""
    merged: list[dict] = []
    for story in stories:
        if merged:
            vectors = np.asarray([s["_vector"] for s in merged], dtype=np.float32)
            sims = vectors @ np.asarray(story["_vector"], dtype=np.float32)
            match = next((int(i) for i in np.argsort(-sims)[:MAX_CHECKS]
                          if same(merged[i], story, float(sims[i]))), None)
            if match is not None:
                merged[match] = merge_into(merged[match], story)
                continue
        merged.append(story)
    return merged


def select(stories: list[dict], now: datetime, cfg: Config) -> list[dict]:
    """Drop stories older than max_age_hours, rank the rest, keep the top max_stories."""
    fresh = [s for s in stories
             if (now - parse_iso(s["published_at"])).total_seconds() / 3600 <= cfg.output.max_age_hours]
    for s in fresh:
        s["score"] = rank(s, now, cfg.output.coverage_weight)
    fresh.sort(key=lambda s: s["score"], reverse=True)
    return fresh[: cfg.output.max_stories]


def write_json(path: Path, data: dict) -> None:
    """Write to a temporary file, then rename it into place. A reader never sees a
    half-written file, because the rename is a single step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def publish(kept: list[tuple], cfg: Config, now: datetime | None = None) -> list[dict]:
    """kept: (story, lead article, page info, verdict) for each story that passed the keep rule.
    Returns the stories written to news.json."""
    now = now or datetime.now(timezone.utc)
    published = json.loads(STATE.read_text(encoding="utf-8"))["stories"] if STATE.exists() else []

    new = [build_story(story.members, lead, info, verdict) for story, lead, info, verdict in kept]
    # Published stories go first so their ids stay stable; merging the whole list also
    # catches duplicates that were already on the page. Every story is re-embedded each run
    # (a few hundred texts take seconds), so all vectors come from the same text and model.
    candidates = published + new
    if candidates:
        for s, v in zip(candidates, grouping.normalize(llm.embed([embed_text(s) for s in candidates], cfg))):
            s["_vector"] = [round(float(x), 5) for x in v]

    cache = json.loads(SAME_CACHE.read_text(encoding="utf-8")) if SAME_CACHE.exists() else {}
    with llm.client(cfg) as client:
        matcher = EventMatcher(cfg, client, cache)
        stories = select(merge(candidates, matcher), now, cfg)
    log.info("same-event checks: %d asked, %d cached", matcher.asked, len(cache) - matcher.asked)

    ids = {s["id"] for s in stories}   # forget pairs whose stories have left the page
    write_json(SAME_CACHE, {k: v for k, v in cache.items() if set(k.split("|")) <= ids})
    write_json(STATE, {"stories": stories})
    write_json(cfg.out_dir / "news.json", {
        "generated_at": iso(now),
        "version": 1,
        "stories": [{k: v for k, v in s.items() if not k.startswith("_")} for s in stories],
    })
    shutil.copy2(SITE / "index.html", cfg.out_dir / "index.html")
    return stories
