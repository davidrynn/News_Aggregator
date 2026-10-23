from datetime import datetime, timedelta, timezone

from newsagg import config, output
from newsagg.enrich import PageInfo
from newsagg.gdelt import Article
from newsagg.scoring import Verdict

CFG = config.load()
NOW = datetime(2026, 10, 23, 18, 0, tzinfo=timezone.utc)


def art(domain, minutes_ago=0, image=None):
    return Article(url=f"https://{domain}/story?utm=x", domain=domain, title="Senate passes budget",
                   seen_at=NOW - timedelta(minutes=minutes_ago), image=image, themes=set(),
                   countries=set(), word_count=400)


def story(title="T", nw=4, hours_ago=0, outlets=("a.com",), vector=(1.0, 0.0)):
    return {"id": title, "title": title, "url": f"https://{outlets[0]}/x", "outlet": outlets[0],
            "published_at": output.iso(NOW - timedelta(hours=hours_ago)), "image": None,
            "description": "", "topic": "world", "kind": "original_reporting", "score": 0.0,
            "newsworthiness": nw, "relevance": 4, "outlet_count": len(outlets),
            "also_reported_by": [{"outlet": o, "url": f"https://{o}/x"} for o in outlets[1:]],
            "_outlets": list(outlets), "_vector": list(vector)}


def test_story_id_ignores_query_string():
    assert output.story_id("https://Ex.com/a?utm=1#top") == output.story_id("https://ex.com/a")


def test_rank_recency_fades_over_24_hours():
    assert output.rank(story(nw=4), NOW) == 13.0              # 4 x 2 + 4 + 0 + 1
    assert output.rank(story(nw=4, hours_ago=12), NOW) == 12.5
    assert output.rank(story(nw=4, hours_ago=30), NOW) == 12.0


def test_build_story_excludes_lead_from_also_reported():
    lead = art("apnews.com", image="https://apnews.com/i.jpg")
    members = [lead, art("local.com", minutes_ago=5), art("local.com", minutes_ago=9), art("tv.com")]
    verdict = Verdict(kind="original_reporting", newsworthiness=5, relevance=5, topic="world", reason="")
    s = output.build_story(members, lead, PageInfo(description="d"), verdict)
    assert s["outlet"] == "apnews.com"
    assert [x["outlet"] for x in s["also_reported_by"]] == ["local.com", "tv.com"]
    assert s["outlet_count"] == 3
    assert s["image"] == "https://apnews.com/i.jpg"


def test_merge_joins_matching_story_and_keeps_stronger_headline():
    old = story("Old headline", nw=3, hours_ago=2, outlets=("a.com", "b.com"))
    new = story("New headline", nw=5, outlets=("c.com", "a.com"), vector=(0.99, 0.141))
    unrelated = story("Other event", vector=(0.0, 1.0))
    merged = output.merge([old], [new, unrelated], threshold=0.87)
    assert len(merged) == 2
    assert merged[0]["title"] == "New headline"
    assert merged[0]["outlet_count"] == 3                    # a, b, c
    assert merged[0]["published_at"] == output.iso(NOW)      # latest coverage


def test_select_drops_old_stories_and_sorts_by_score():
    stories = [story("low", nw=2), story("high", nw=5), story("stale", nw=5, hours_ago=40)]
    chosen = output.select(stories, NOW, CFG)
    assert [s["title"] for s in chosen] == ["high", "low"]
