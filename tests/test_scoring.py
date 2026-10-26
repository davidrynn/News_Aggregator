import dataclasses
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from newsagg import config, scoring
from newsagg.enrich import PageInfo
from newsagg.gdelt import Article

CFG = config.load()
ART = Article(url="https://x.com/a", domain="x.com", title="Seven killed in Jordan church attack",
              seen_at=datetime(2026, 10, 23, tzinfo=timezone.utc), image=None, themes=set(),
              countries=set(), word_count=400)


def test_system_prompt_matches_spec_start_and_end():
    assert scoring.SYSTEM_PROMPT.startswith("You classify news articles for an automated news page.")
    assert scoring.SYSTEM_PROMPT.rstrip().endswith('"Dateline (AP) marks wire copy."')


def test_user_message_follows_template_and_omits_outlet():
    msg = scoring.user_message(ART, PageInfo(), CFG)
    assert msg.splitlines()[3] == "Title: Seven killed in Jordan church attack"
    assert "Description: (none)" in msg and "Byline present: no" in msg
    assert "Page type: unknown" in msg and "Section: unknown" in msg
    assert "x.com" not in msg


def test_verdict_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        scoring.Verdict(kind="opinion", newsworthiness=7, relevance=3, topic="world", reason="x")


@pytest.mark.parametrize("kind, nw, rel, kept", [
    ("original_reporting", 3, 3, True),
    ("original_reporting", 2, 5, False),
    ("aggregation", 5, 5, False),
    ("wire_copy", 4, 4, True),
])
def test_keep_rule(kind, nw, rel, kept):
    v = scoring.Verdict(kind=kind, newsworthiness=nw, relevance=rel, topic="world", reason="x")
    assert scoring.keep(v, CFG) is kept


def test_batch_message_numbers_articles_under_one_header():
    other = dataclasses.replace(ART, title="Flood warnings across Pakistan")
    msg = scoring.batch_message([(ART, PageInfo()), (other, PageInfo())], CFG)
    lines = msg.splitlines()
    assert lines[0].startswith("Topics: ") and lines[1].startswith("Regions: ")
    assert msg.count("Topics: ") == 1
    assert "Article 1\nTitle: Seven killed in Jordan church attack" in msg
    assert "Article 2\nTitle: Flood warnings across Pakistan" in msg


def test_score_batch_keeps_good_verdicts_when_one_is_bad_or_missing(monkeypatch):
    good = {"kind": "wire_copy", "newsworthiness": 4, "relevance": 4, "topic": "world", "reason": "x"}
    reply = {"verdicts": [{**good, "n": 3}, {**good, "n": 1, "newsworthiness": 9}]}   # 1 invalid, 2 missing
    monkeypatch.setattr(scoring.llm, "chat_json", lambda *a: json.dumps(reply))
    got = scoring.score_batch([(ART, PageInfo())] * 3, CFG, client=None)
    assert got[0] is None and got[1] is None
    assert got[2] == scoring.Verdict(**good)


def test_score_batch_failed_call_gives_none_for_every_story(monkeypatch):
    monkeypatch.setattr(scoring.llm, "chat_json", lambda *a: "not json")
    assert scoring.score_batch([(ART, PageInfo())] * 2, CFG, client=None) == [None, None]


def test_unlisted_topic_becomes_other():
    v = scoring.Verdict(kind="lifestyle", newsworthiness=1, relevance=1, topic="sports", reason="x")
    assert v.topic == "other"
