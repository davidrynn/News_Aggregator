"""Output: merge this run's kept stories into out/news.json, the contract every front end reads.

news.json holds up to output.max_stories stories from the last output.max_age_hours. Stories
carry over between runs. A new story whose headline matches a published one (same embedding
threshold as grouping) is merged into it instead of becoming a duplicate tile.

Fields starting with "_" (the headline vector, the full outlet list) are kept in
data/published.json for merging and stripped from news.json.
"""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from newsagg import grouping, llm
from newsagg.config import ROOT, Config
from newsagg.enrich import PageInfo
from newsagg.gdelt import Article
from newsagg.scoring import Verdict

STATE = ROOT / "data" / "published.json"
SITE = Path(__file__).parent / "site"
MAX_ALSO = 10   # "also reported by" links kept per story


def story_id(url: str) -> str:
    """sha1 of the lead URL without query string or fragment."""
    parts = urlsplit(url)
    normalized = urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, "", ""))
    return hashlib.sha1(normalized.encode()).hexdigest()


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def rank(story: dict, now: datetime) -> float:
    """newsworthiness x 2 + relevance + outlet bonus + recency (1 when new, 0 at 24 hours)."""
    age_hours = (now - parse_iso(story["published_at"])).total_seconds() / 3600
    recency = max(0.0, 1 - age_hours / 24)
    return round(story["newsworthiness"] * 2 + story["relevance"]
                 + story.get("_outlet_bonus", 0.0) + recency, 2)


def build_story(members: list[Article], lead: Article, info: PageInfo, verdict: Verdict) -> dict:
    outlets = list(dict.fromkeys(a.domain for a in [lead, *members]))   # unique, lead first
    also, seen = [], {lead.domain}
    for a in members:
        if a.domain not in seen and len(also) < MAX_ALSO:
            also.append({"outlet": a.domain, "url": a.url})
            seen.add(a.domain)
    return {
        "id": story_id(lead.url),
        "title": lead.title,
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
    links = {x["outlet"]: x for x in other["also_reported_by"] + base["also_reported_by"]}
    links.pop(base["outlet"], None)
    merged.update(
        _outlets=outlets,
        outlet_count=len(outlets),
        also_reported_by=list(links.values())[:MAX_ALSO],
        published_at=max(old["published_at"], new["published_at"]),
        _vector=base["_vector"],
    )
    return merged


def merge(published: list[dict], new: list[dict], threshold: float) -> list[dict]:
    stories = list(published)
    for story in new:
        if stories:
            vectors = np.asarray([s["_vector"] for s in stories], dtype=np.float32)
            sims = vectors @ np.asarray(story["_vector"], dtype=np.float32)
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                stories[best] = merge_into(stories[best], story)
                continue
        stories.append(story)
    return stories


def select(stories: list[dict], now: datetime, cfg: Config) -> list[dict]:
    """Drop stories older than max_age_hours, rank the rest, keep the top max_stories."""
    fresh = [s for s in stories
             if (now - parse_iso(s["published_at"])).total_seconds() / 3600 <= cfg.output.max_age_hours]
    for s in fresh:
        s["score"] = rank(s, now)
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
    if new:
        titles = [grouping.PREFIX + grouping.clean_title(s["title"]) for s in new]
        for s, v in zip(new, grouping.normalize(llm.embed(titles, cfg))):
            s["_vector"] = [round(float(x), 5) for x in v]

    stories = select(merge(published, new, cfg.grouping.threshold), now, cfg)
    write_json(STATE, {"stories": stories})
    write_json(cfg.out_dir / "news.json", {
        "generated_at": iso(now),
        "version": 1,
        "stories": [{k: v for k, v in s.items() if not k.startswith("_")} for s in stories],
    })
    shutil.copy2(SITE / "index.html", cfg.out_dir / "index.html")
    return stories
