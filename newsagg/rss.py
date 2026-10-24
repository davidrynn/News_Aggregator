"""Articles straight from outlets' RSS feeds (config feeds.extra), alongside GDELT.

GDELT barely covers the largest outlets (in a typical two hours: no AP, Reuters, NYT,
NPR or CNN), so their own feeds bring them in. Each feed item becomes the same Article
GDELT produces, and every later stage treats the two alike.
"""
import asyncio
import calendar
import html
import logging
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import feedparser
import httpx

from newsagg.blocklist import normalize_host, registered_domain
from newsagg.gdelt import Article

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; newsagg/0.1; personal news aggregator)"}
TIMEOUT = 15
TAGS = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    """Feed summaries often hold HTML: drop tags, decode entities, collapse spaces."""
    return " ".join(html.unescape(TAGS.sub(" ", text)).split())


def entry_time(entry) -> datetime | None:
    """Most feeds give a publish time; some (DW, Science) only an updated time."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc) if parsed else None


def entry_image(entry) -> str | None:
    for key in ("media_content", "media_thumbnail"):
        for item in entry.get(key) or []:
            if item.get("url"):
                return item["url"]
    for link in entry.get("links", []):
        if link.get("type", "").startswith("image/") and link.get("href"):
            return link["href"]
    return None


def parse_feed(content: bytes, now: datetime, max_age: timedelta) -> list[Article]:
    articles = []
    for entry in feedparser.parse(content).entries:
        url = entry.get("link", "")
        title = clean_text(entry.get("title", ""))
        seen_at = entry_time(entry)
        if not url.startswith(("http://", "https://")) or not title or seen_at is None:
            continue    # undated items would look new forever, so they are skipped
        if now - seen_at > max_age:
            continue
        articles.append(Article(
            url=url,
            domain=registered_domain(normalize_host(urlsplit(url).hostname or "")),
            title=title,
            seen_at=min(seen_at, now),      # a few feeds date items slightly in the future
            image=entry_image(entry),
            themes=set(),
            countries=set(),
            word_count=None,
            summary=clean_text(entry.get("summary", ""))[:500],
            from_feed=True,
        ))
    return articles


async def fetch_feed(client: httpx.AsyncClient, url: str, now: datetime,
                     max_age: timedelta) -> list[Article]:
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as e:
        log.warning(f"feed failed, skipped this run: {url} ({type(e).__name__})")
        return []
    return parse_feed(response.content, now, max_age)


async def fetch_all(urls: list[str], now: datetime, max_age: timedelta) -> list[Article]:
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True, headers=HEADERS) as client:
        results = await asyncio.gather(*(fetch_feed(client, u, now, max_age) for u in urls))
    return [a for feed in results for a in feed]


def fetch(urls: list[str], max_age_hours: float, now: datetime | None = None) -> list[Article]:
    """All items from all feeds published within max_age_hours, fetched in parallel."""
    if not urls:
        return []
    return asyncio.run(fetch_all(urls, now or datetime.now(timezone.utc), timedelta(hours=max_age_hours)))
