"""Enrichment: fetch each story's page and read publisher metadata (rule 10).

Reads og:description, og:image, whether an author is credited, and the JSON-LD
@type and articleSection. Article body text is never stored.

Pages are fetched concurrently with asyncio: while one request waits on the network,
others are sent. A semaphore caps how many are in flight at once.
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from newsagg.gdelt import Article

log = logging.getLogger(__name__)

MAX_BYTES = 2_000_000
TIMEOUT = 10
CONCURRENCY = 16
MAX_ATTEMPTS = 3   # members tried per story before giving up
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; newsagg/0.1; personal news aggregator)"}


@dataclass
class PageInfo:
    description: str = ""
    image: str | None = None
    has_byline: bool = False
    ld_types: list[str] = field(default_factory=list)
    section: str = ""


def parse_page(html: str) -> PageInfo:
    tree = HTMLParser(html)

    def meta(*keys: str) -> str:
        for key in keys:
            node = tree.css_first(f'meta[property="{key}"], meta[name="{key}"]')
            if node and (content := (node.attributes.get("content") or "").strip()):
                return content
        return ""

    info = PageInfo(
        description=meta("og:description", "description", "twitter:description"),
        image=meta("og:image", "twitter:image") or None,
        has_byline=bool(meta("author", "article:author") or tree.css_first('[rel="author"]')),
    )
    for obj in json_ld_objects(tree):
        types = obj.get("@type", [])
        types = [types] if isinstance(types, str) else [t for t in types if isinstance(t, str)]
        if not any(t.endswith(("Article", "BlogPosting", "Report")) for t in types):
            continue
        info.ld_types = types
        section = obj.get("articleSection", "")
        info.section = ", ".join(section) if isinstance(section, list) else str(section)
        info.has_byline = info.has_byline or bool(obj.get("author"))
        info.description = info.description or str(obj.get("description", ""))
        break
    return info


def json_ld_objects(tree: HTMLParser) -> list[dict]:
    """Every JSON-LD object on the page, flattening lists and @graph containers."""
    found: list[dict] = []

    def walk(data) -> None:
        if isinstance(data, list):
            for item in data:
                walk(item)
        elif isinstance(data, dict):
            found.append(data)
            walk(data.get("@graph", []))

    for node in tree.css('script[type="application/ld+json"]'):
        try:
            walk(json.loads(node.text(strip=True) or "null"))
        except json.JSONDecodeError:
            continue   # malformed JSON-LD is common; ignore it
    return found


async def fetch_html(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        async with client.stream("GET", url) as resp:
            if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
                return None
            chunks, size = [], 0
            async for chunk in resp.aiter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size >= MAX_BYTES:
                    break
            return b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
    except (httpx.HTTPError, UnicodeDecodeError):
        return None


async def _enrich(candidates: list[list[Article]]) -> list[tuple[Article, PageInfo] | None]:
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True, headers=HEADERS) as client:

        async def one(members: list[Article]) -> tuple[Article, PageInfo] | None:
            for article in members[:MAX_ATTEMPTS]:
                async with sem:
                    html = await fetch_html(client, article.url)
                if html:
                    return article, parse_page(html)
            return None

        return await asyncio.gather(*(one(m) for m in candidates))


def enrich(candidates: list[list[Article]]) -> list[tuple[Article, PageInfo] | None]:
    """For each story's member list (best lead first), return the first article whose
    page could be fetched with its PageInfo, or None if every attempt failed."""
    return asyncio.run(_enrich(candidates))
