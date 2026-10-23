from datetime import datetime, timezone

import numpy as np
import pytest

from newsagg import grouping
from newsagg.gdelt import Article


@pytest.mark.parametrize("title, expected", [
    ("Florida officer accused of taking over home – NBC Connecticut", "Florida officer accused of taking over home"),
    ("HOLDING HER BACK? | The Manila Times", "HOLDING HER BACK? | The Manila Times"),  # too short once stripped
    ("Senate passes budget bill after long debate", "Senate passes budget bill after long debate"),
])
def test_clean_title(title, expected):
    assert grouping.clean_title(title) == expected


def test_group_vectors_uses_leaders_not_chains():
    # b is close to a, c is close to b but not to a: c must start its own group.
    angle = np.radians([0, 25, 50])
    vectors = np.stack([np.cos(angle), np.sin(angle)], axis=1)  # unit vectors
    assert grouping.group_vectors(vectors, threshold=0.85) == [[0, 1], [2]]


def _art(domain: str) -> Article:
    return Article(url=f"https://{domain}/a", domain=domain, title="t", image=None, themes=set(),
                   countries=set(), word_count=100, seen_at=datetime(2026, 10, 23, tzinfo=timezone.utc))


def test_pick_lead_prefers_wire_service():
    members = [_art("localpaper.com"), _art("apnews.com"), _art("tvstation.com")]
    lead_first = grouping.pick_lead(members, ["apnews.com"])
    assert [a.domain for a in lead_first] == ["apnews.com", "localpaper.com", "tvstation.com"]


def test_pick_lead_keeps_order_without_wire():
    members = [_art("a.com"), _art("b.com")]
    assert grouping.pick_lead(members, ["apnews.com"]) == members
