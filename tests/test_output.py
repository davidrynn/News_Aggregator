from datetime import datetime, timedelta, timezone

import pytest

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


@pytest.mark.parametrize("title, shown", [
    ("1st LD Writethru: Netanyahu says plans major West Bank operation",
     "Netanyahu says plans major West Bank operation"),
    ("UPDATE 2-Oil prices rise on supply worries", "Oil prices rise on supply worries"),
    ("REFILE-Bank of Japan holds rates", "Bank of Japan holds rates"),
    ("Report: Devils acquire Hellebuyck", "Report: Devils acquire Hellebuyck"),       # meaningful, kept
    ("Opinion | Voboril: Know the code", "Opinion | Voboril: Know the code"),
    ("Update on the city budget", "Update on the city budget"),                     # a real word, not a label
])
def test_display_title_strips_only_wire_labels(title, shown):
    assert output.display_title(title) == shown


def test_story_id_ignores_query_string():
    assert output.story_id("https://Ex.com/a?utm=1#top") == output.story_id("https://ex.com/a")


def test_rank_recency_fades_over_24_hours():
    assert output.rank(story(nw=4), NOW) == 13.0              # 4 x 2 + 4 + 0 + 1
    assert output.rank(story(nw=4, hours_ago=12), NOW) == 12.5
    assert output.rank(story(nw=4, hours_ago=30), NOW) == 12.0


def test_rank_coverage_breaks_ties_but_not_newsworthiness():
    wide = story(nw=4, outlets=tuple(f"o{i}.com" for i in range(32)))
    assert output.rank(wide, NOW) == 15.5                     # 13 + 0.5 x log2(32)
    assert output.rank(story(nw=5), NOW) > output.rank(story(nw=4, outlets=("a.com", "b.com")), NOW)


def test_build_story_excludes_lead_from_also_reported():
    lead = art("apnews.com", image="https://apnews.com/i.jpg")
    members = [lead, art("local.com", minutes_ago=5), art("local.com", minutes_ago=9), art("tv.com")]
    verdict = Verdict(kind="original_reporting", newsworthiness=5, relevance=5, topic="world", reason="")
    s = output.build_story(members, lead, PageInfo(description="d"), verdict)
    assert s["outlet"] == "apnews.com"
    assert [x["outlet"] for x in s["also_reported_by"]] == ["local.com", "tv.com"]
    assert s["outlet_count"] == 3
    assert s["image"] == "https://apnews.com/i.jpg"


def by_similarity(a, b, sim):
    return sim >= 0.87


def test_merge_joins_matching_story_and_keeps_stronger_headline():
    old = story("Old headline", nw=3, hours_ago=2, outlets=("a.com", "b.com"))
    new = story("New headline", nw=5, outlets=("c.com", "a.com"), vector=(0.99, 0.141))
    unrelated = story("Other event", vector=(0.0, 1.0))
    merged = output.merge([old, new, unrelated], by_similarity)
    assert len(merged) == 2
    assert merged[0]["title"] == "New headline"
    assert merged[0]["outlet_count"] == 3                    # a, b, c
    assert merged[0]["published_at"] == output.iso(NOW)      # latest coverage
    assert [x["outlet"] for x in merged[0]["also_reported_by"]] == ["a.com", "b.com"]  # old lead kept


def test_merge_tries_next_closest_when_closest_is_rejected():
    first = story("Iran cables", vector=(1.0, 0.0))
    second = story("Iran strike", vector=(0.8, 0.6))
    new = story("US may strike Iran", vector=(0.95, 0.312))   # closer to first, same event as second
    merged = output.merge([first, second, new], lambda a, b, sim: a["title"] == "Iran strike" and b is new)
    assert [s["title"] for s in merged] == ["Iran cables", "Iran strike"]
    assert merged[1]["outlet_count"] == 1                     # same outlet a.com, unioned


def test_matcher_asks_model_only_between_thresholds(monkeypatch):
    calls = []
    monkeypatch.setattr(output.llm, "chat_json",
                        lambda *args: calls.append(args) or '{"same_event": true}')
    m = output.EventMatcher(CFG, client=None, cache={})
    a, b = story("A"), story("B")
    assert m(a, b, 0.95) is True and m(a, b, 0.70) is False and not calls
    assert m(a, b, 0.85) is True and m(b, a, 0.85) is True     # second call comes from the cache
    assert len(calls) == 2 and m.cache == {"A|B": True}      # asked once in each order


def test_matcher_needs_yes_in_both_orders(monkeypatch):
    answers = iter(['{"same_event": true}', '{"same_event": false}'])
    monkeypatch.setattr(output.llm, "chat_json", lambda *args: next(answers))
    m = output.EventMatcher(CFG, client=None, cache={})
    assert m(story("A"), story("B"), 0.85) is False


def test_select_drops_old_stories_and_sorts_by_score():
    stories = [story("low", nw=2), story("high", nw=5), story("stale", nw=5, hours_ago=40)]
    chosen = output.select(stories, NOW, CFG)
    assert [s["title"] for s in chosen] == ["high", "low"]


def test_merge_prefers_better_lead_over_one_point_of_newsworthiness():
    local = dict(story("Local headline", nw=5, hours_ago=1, outlets=("local.com",)), _lead_rank=2)
    nyt = dict(story("NYT headline", nw=4, outlets=("nytimes.com",), vector=(0.99, 0.141)), _lead_rank=1)
    [merged] = output.merge([local, nyt], by_similarity)
    assert merged["title"] == "NYT headline"
    assert merged["newsworthiness"] == 5           # the higher of the two is kept


def topical(title, topic, nw):
    return dict(story(title, nw=nw), topic=topic)


def test_select_guarantees_topic_minimums_without_padding():
    cfg = CFG.model_copy(update={"output": CFG.output.model_copy(
        update={"max_stories": 3, "min_per_topic": {"arts": 1, "science": 2}})})
    stories = [topical("w1", "world", 5), topical("w2", "world", 5), topical("w3", "world", 4),
               topical("a1", "arts", 3), topical("a2", "arts", 3)]          # no science kept
    chosen = output.select(stories, NOW, cfg)
    assert [s["title"] for s in chosen] == ["w1", "w2", "a1"]             # one arts place, sorted by score


def test_select_without_minimums_is_pure_rank():
    cfg = CFG.model_copy(update={"output": CFG.output.model_copy(update={"max_stories": 2, "min_per_topic": {}})})
    stories = [topical("w1", "world", 5), topical("a1", "arts", 3), topical("w2", "world", 4)]
    assert [s["title"] for s in output.select(stories, NOW, cfg)] == ["w1", "w2"]


def test_select_gives_feed_led_stories_the_outlet_bonus():
    cfg = CFG.model_copy(update={"output": CFG.output.model_copy(
        update={"max_stories": 1, "min_per_topic": {}, "feed_bonus": 1.5})})
    hype = story("Tabloid hype", nw=4)                                    # 13.0
    nature = dict(story("Nature finding", nw=4, hours_ago=12), _feed_lead=True)   # 12.5 + 1.5
    [top] = output.select([hype, nature], NOW, cfg)
    assert top["title"] == "Nature finding" and top["score"] == 14.0
    assert output.rank(story(nw=5), NOW) > 14.0                           # still worth less than a point of newsworthiness
