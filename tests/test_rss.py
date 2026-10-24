from datetime import datetime, timedelta, timezone

from newsagg import rss

NOW = datetime(2026, 10, 23, 18, 0, tzinfo=timezone.utc)

FEED = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
<channel><title>Example</title>
<item>
  <title>Senate passes budget &amp; tax bill</title>
  <link>https://www.example.co.uk/news/senate-budget</link>
  <pubDate>Fri, 23 Oct 2026 17:00:00 GMT</pubDate>
  <description>&lt;p&gt;The Senate voted &lt;b&gt;52-48&lt;/b&gt; late on Friday.&lt;/p&gt;</description>
  <media:thumbnail url="https://img.example.co.uk/a.jpg"/>
</item>
<item>
  <title>Old story</title>
  <link>https://www.example.co.uk/news/old</link>
  <pubDate>Mon, 19 Oct 2026 10:00:00 GMT</pubDate>
</item>
<item>
  <title>No date at all</title>
  <link>https://www.example.co.uk/news/undated</link>
</item>
</channel></rss>"""


def test_parse_feed_builds_articles_like_gdelt():
    [a] = rss.parse_feed(FEED, NOW, timedelta(hours=36))     # old and undated items skipped
    assert a.title == "Senate passes budget & tax bill"
    assert a.domain == "example.co.uk"
    assert a.seen_at == datetime(2026, 10, 23, 17, 0, tzinfo=timezone.utc)
    assert a.image == "https://img.example.co.uk/a.jpg"
    assert a.summary == "The Senate voted 52-48 late on Friday."
    assert a.word_count is None


def test_future_dates_are_clamped_to_now():
    future = FEED.replace(b"Fri, 23 Oct 2026 17:00:00", b"Fri, 23 Oct 2026 19:30:00")
    [a] = rss.parse_feed(future, NOW, timedelta(hours=36))
    assert a.seen_at == NOW
