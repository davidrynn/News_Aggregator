from datetime import datetime, timedelta, timezone

import pytest

from newsagg import filters
from newsagg.blocklist import Blocklist
from newsagg.gdelt import Article

NOW = datetime(2026, 10, 23, 16, 0, tzinfo=timezone.utc)


def art(url="https://example.com/news/a", title="Senate passes budget", words=400,
        domain="example.com", minutes_ago=0) -> Article:
    return Article(url=url, domain=domain, title=title, seen_at=NOW - timedelta(minutes=minutes_ago),
                   image=None, themes=set(), countries=set(), word_count=words)


BL = Blocklist(["badsite.com", "economictimes.indiatimes.com", "westernjournal.com/commentary"],
               allow=["notbad.badsite.com"])


@pytest.mark.parametrize("url, blocked", [
    ("https://badsite.com/x", True),
    ("https://www.badsite.com/x", True),
    ("https://news.badsite.com/x", True),                    # subdomain of a listed site
    ("https://notbad.badsite.com/x", False),                 # allow overrides
    ("https://economictimes.indiatimes.com/x", True),
    ("https://timesofindia.indiatimes.com/x", False),        # sibling subdomain is not blocked
    ("https://www.westernjournal.com/commentary/piece", True),
    ("https://www.westernjournal.com/news/story", False),    # only the listed section
    ("https://goodsite.com/x", False),
])
def test_blocklist(url, blocked):
    assert BL.is_blocked(url) is blocked


@pytest.mark.parametrize("url, dropped", [
    ("https://x.com/opinion/2026/piece", True),
    ("https://x.com/news/opinion-polls-tighten", False),
    ("https://x.com/business/food-processing-minister", False),
])
def test_section_path(url, dropped):
    assert (filters.section_path(art(url=url)) is not None) is dropped


@pytest.mark.parametrize("title, dropped", [
    ("17 Things You Need To Know", True),
    ("You Won't Believe What This Dog Did", True),
    ("3 killed in highway crash", False),
    ("Senate passes budget", False),
])
def test_clickbait(title, dropped):
    assert (filters.clickbait(art(title=title)) is not None) is dropped


@pytest.mark.parametrize("title, dropped", [
    ("破解了罗塞塔石碑，却拼不回一个古埃及 | www.wenxuecity.com", True),   # Chinese with a Latin URL
    ("ਬਿਜਲੀ ਬੋਰਡ ਦੇ ਨਿਜੀਕਰਨ ਵਿਰੁੱਧ ਰੋਸ ਰੈਲੀ - Protest rally", True),
    ("Nestlé reports strong Q3 in São Paulo", False),          # accented Latin is fine
    ("2026", False),                                           # no letters at all
])
def test_non_latin(title, dropped):
    assert (filters.non_latin(art(title=title)) is not None) is dropped


def test_apply_counts_reasons_and_caps_outlets():
    articles = [art(url=f"https://busy.com/news/{i}", domain="busy.com", minutes_ago=i) for i in range(5)]
    articles += [art(url="https://badsite.com/x"), art(words=10)]
    kept, drops = filters.apply(articles, BL, min_words=50, max_per_outlet=3)
    assert [a.url for a in kept] == [f"https://busy.com/news/{i}" for i in range(3)]  # newest 3
    assert drops == {"blocklist": 1, "short": 1, "outlet_cap": 2}
