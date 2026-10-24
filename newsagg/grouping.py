"""Story grouping: articles about the same event share one tile.

Headlines are embedded (turned into vectors that capture meaning), then grouped by
cosine similarity. Word-overlap matching missed most same-event pairs written in
different words; see the spec's Story grouping notes.
"""
import re
from dataclasses import dataclass, field

import numpy as np

from newsagg import llm
from newsagg.config import Config
from newsagg.gdelt import Article

# " | The Manila Times", " - NBC Connecticut", " – Isle of Wight County Press"
SUFFIX = re.compile(r"\s+[|\u2013\u2014-]\s+[^|\u2013\u2014-]+$")

# nomic-embed-text was trained with task prefixes; "clustering: " suits grouping.
PREFIX = "clustering: "


def clean_title(title: str) -> str:
    """Strip a trailing outlet name, unless that would leave a very short title."""
    stripped = SUFFIX.sub("", title).strip()
    return stripped if len(stripped.split()) >= 4 else title


@dataclass
class Story:
    lead: Article
    members: list[Article] = field(default_factory=list)
    triage: int | None = None    # headline rating 1-5, set when scoring.triage is on

    @property
    def outlets(self) -> set[str]:
        return {outlet(a) for a in self.members}

    @property
    def has_feed_article(self) -> bool:
        return any(a.from_feed for a in self.members)


# Big News Network runs dozens of local-sounding sites (brazilsun.com, irishsun.com,
# northkoreatimes.com, ...) that republish wire copy and opinion under their own names.
# All of them, and only them, use article URLs like /news/279390436/slug (checked
# 2026-10-23: 43 domains). They count as one outlet and never lead a story.
NETWORK = "bignewsnetwork.com"
NETWORK_URL = re.compile(r"^https?://(www\.)?[^/]+/news/\d{9}/")


def outlet(a: Article) -> str:
    """The outlet an article counts as: its domain, or the network it belongs to."""
    return NETWORK if NETWORK_URL.match(a.url) else a.domain


def lead_rank(a: Article, wire_outlets: list[str]) -> int:
    """Lower is a better lead: 0 wire service (the original source), 1 a major outlet's
    own feed (curated and readable), 2 any other outlet, 3 the syndication network."""
    if a.domain in wire_outlets:
        return 0
    if a.from_feed:
        return 1
    return 3 if outlet(a) == NETWORK else 2


def group_vectors(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Leader clustering: each row joins the first group whose leader it matches.

    Rows must be unit length, so a dot product is the cosine similarity.
    Comparing only against leaders avoids chaining unrelated stories together.
    """
    leaders: list[int] = []
    groups: list[list[int]] = []
    for i in range(len(vectors)):
        if leaders:
            sims = vectors[leaders] @ vectors[i]
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                groups[best].append(i)
                continue
        leaders.append(i)
        groups.append([i])
    return groups


def normalize(vectors: list[list[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def pick_lead(members: list[Article], wire_outlets: list[str]) -> list[Article]:
    """Order members best lead first, by lead_rank; within a rank, the existing order
    (newest first) is kept. Enrichment tries them in this order, so the fallbacks are
    the next best leads too."""
    return sorted(members, key=lambda a: lead_rank(a, wire_outlets))


def group(articles: list[Article], cfg: Config) -> list[Story]:
    """Group articles into stories, largest first. Members are ordered best lead first
    (see pick_lead)."""
    articles = sorted(articles, key=lambda a: a.seen_at, reverse=True)
    vectors = normalize(llm.embed([PREFIX + clean_title(a.title) for a in articles], cfg))
    stories = []
    for g in group_vectors(vectors, cfg.grouping.threshold):
        members = pick_lead([articles[i] for i in g], cfg.grouping.wire_outlets)
        stories.append(Story(lead=members[0], members=members))
    return sorted(stories, key=lambda s: len(s.outlets), reverse=True)
