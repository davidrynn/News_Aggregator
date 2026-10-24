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


def test_pick_lead_order_wire_then_feed_then_other_then_network():
    from dataclasses import replace
    network = replace(_art("brazilsun.com"), url="http://www.brazilsun.com/news/279390436/slug")
    feed = replace(_art("nytimes.com"), from_feed=True)
    other, wire = _art("localpaper.com"), _art("apnews.com")
    ordered = grouping.pick_lead([network, other, feed, wire], ["apnews.com"])
    assert [a.domain for a in ordered] == ["apnews.com", "nytimes.com", "localpaper.com", "brazilsun.com"]


def test_network_sites_count_as_one_outlet():
    from dataclasses import replace
    sites = [replace(_art(d), url=f"https://{d}/news/279390436/slug")
             for d in ("brazilsun.com", "irishsun.com", "taiwansun.com")]
    story = grouping.Story(lead=sites[0], members=sites + [_art("bbc.com")])
    assert story.outlets == {"bignewsnetwork.com", "bbc.com"}


def test_network_pattern_ignores_ordinary_news_urls():
    assert grouping.outlet(_art("example.com")) == "example.com"          # https://example.com/a
    from dataclasses import replace
    dated = replace(_art("example.com"), url="https://example.com/news/2026/10/23/slug")
    assert grouping.outlet(dated) == "example.com"
