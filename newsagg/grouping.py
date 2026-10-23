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

    @property
    def outlets(self) -> set[str]:
        return {a.domain for a in self.members}


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
    """Reorder members so the lead comes first: the wire service itself if it is in the
    group (the tile then links to the source, not a reprint), otherwise the newest."""
    wire = [a for a in members if a.domain in wire_outlets]
    return wire[:1] + [a for a in members if a not in wire[:1]]


def group(articles: list[Article], cfg: Config) -> list[Story]:
    """Group articles into stories, largest first. Members are newest first, with a
    wire-service article moved to the front when there is one."""
    articles = sorted(articles, key=lambda a: a.seen_at, reverse=True)
    vectors = normalize(llm.embed([PREFIX + clean_title(a.title) for a in articles], cfg))
    stories = []
    for g in group_vectors(vectors, cfg.grouping.threshold):
        members = pick_lead([articles[i] for i in g], cfg.grouping.wire_outlets)
        stories.append(Story(lead=members[0], members=members))
    return sorted(stories, key=lambda s: len(s.outlets), reverse=True)
