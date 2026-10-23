"""Rule-based pre-filters: cheap checks that run before any page fetch or model call.

Each rule returns a short reason when an article should be dropped, or None to keep it.
Rule numbers match the spec's Rule-based pre-filters table.
"""
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Container
from urllib.parse import urlsplit

from newsagg.blocklist import Blocklist
from newsagg.enrich import PageInfo
from newsagg.gdelt import Article

# Rule 3: matched against whole path segments, so /opinion/ matches but /opinion-polls/ does not.
SECTION_SEGMENTS = {
    "opinion", "opinions", "commentary", "editorial", "editorials", "op-ed", "oped",
    "column", "columns", "blog", "blogs", "sponsored", "deals", "shopping",
    "horoscope", "horoscopes",
}

# Rule 4. The number-list pattern needs a listicle noun after the number, so
# "3 killed in crash" is not caught.
CLICKBAIT = re.compile(
    r"you won'?t believe|here'?s why|\bthis one\b|what happened next"
    r"|^\d+\s+(things|ways|reasons|tips|signs|facts|secrets|mistakes|photos|pictures"
    r"|best|times|foods|places|questions|products)\b",
    re.IGNORECASE,
)


# Rule 11: schema.org types that mark a page as not news reporting.
NOT_NEWS_TYPES = {"OpinionNewsArticle", "ReviewNewsArticle", "SatiricalArticle",
                  "AdvertiserContentArticle", "BackgroundNewsArticle"}
OPINION_SECTION = re.compile(r"\b(opinion|commentary|editorial|op-ed|column)", re.IGNORECASE)


def page_type(info: PageInfo) -> str | None:
    if NOT_NEWS_TYPES.intersection(info.ld_types) or OPINION_SECTION.search(info.section):
        return "page_type"
    return None


def blocked(a: Article, bl: Blocklist) -> str | None:
    return "blocklist" if bl.is_blocked(a.url) else None


def section_path(a: Article) -> str | None:
    segments = {s.lower() for s in urlsplit(a.url).path.split("/")}
    return "section" if segments & SECTION_SEGMENTS else None


def clickbait(a: Article) -> str | None:
    return "clickbait" if CLICKBAIT.search(a.title) else None


def non_latin(a: Article) -> str | None:
    """Drop titles where over 20% of letters are not Latin script (Chinese, Punjabi, ...).

    GDELT's English feed still carries some non-English pages, often with an English
    translation appended, so a majority test is not enough. unicodedata.name gives every
    character's official name; Latin letters' names start with LATIN. Vowel signs in
    scripts like Gurmukhi are "marks", not letters, so they are counted explicitly.
    """
    letters = [ch for ch in a.title if ch.isalpha() or unicodedata.category(ch).startswith("M")]
    if not letters:
        return None
    non_latin = sum(not unicodedata.name(ch, "").startswith(("LATIN", "COMBINING")) for ch in letters)
    return "language" if non_latin / len(letters) > 0.2 else None


def too_short(a: Article, min_words: int) -> str | None:
    return "short" if a.word_count < min_words else None


def cap_per_outlet(articles: list[Article], limit: int) -> tuple[list[Article], int]:
    """Rule 7: keep only the newest `limit` articles per outlet."""
    by_outlet: dict[str, list[Article]] = defaultdict(list)
    for a in articles:
        by_outlet[a.domain].append(a)
    kept = [a for group in by_outlet.values()
            for a in sorted(group, key=lambda a: a.seen_at, reverse=True)[:limit]]
    return kept, len(articles) - len(kept)


def apply(articles: list[Article], bl: Blocklist, min_words: int, max_per_outlet: int,
          seen: Container[str] = ()) -> tuple[list[Article], Counter]:
    """Run the rules in order. Returns the survivors and a count of drops per reason.
    seen holds URLs already scored in an earlier run (rule 2)."""
    drops: Counter = Counter()
    kept = []
    for a in articles:
        reason = (("seen" if a.url in seen else None) or blocked(a, bl) or section_path(a)
                  or non_latin(a) or clickbait(a) or too_short(a, min_words))
        if reason:
            drops[reason] += 1
        else:
            kept.append(a)
    kept, capped = cap_per_outlet(kept, max_per_outlet)
    if capped:
        drops["outlet_cap"] = capped
    return kept, drops
