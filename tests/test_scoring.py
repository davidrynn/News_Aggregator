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
    assert scoring.SYSTEM_PROMPT.rstrip().endswith('"Restates a Reuters report."')


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
