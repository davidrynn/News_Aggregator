"""Article discovery from GDELT's raw 15-minute GKG files.

GDELT publishes a new zipped file every 15 minutes listing every article it saw, with
its URL, outlet, title, themes and locations. Downloading the files avoids the DOC
search API, whose rate limit is far stricter than documented.

GDELT sometimes lists a file in lastupdate.txt before it is uploaded. So each run looks
back over several files and reads any it has not read before; the names of files already
read are kept in data/gdelt_done.txt.

File format: tab-separated, one article per row, no header. Column numbers below come
from the GKG 2.1 codebook.
"""
import html
import io
import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from newsagg.config import ROOT

log = logging.getLogger(__name__)

BASE = "https://data.gdeltproject.org/gdeltv2"
LAST_UPDATE = f"{BASE}/lastupdate.txt"
DONE_FILE = ROOT / "data" / "gdelt_done.txt"

# GKG 2.1 column indexes
COL_DATE = 1
COL_COLLECTION = 2   # 1 = web article; others are broadcast, PDFs, etc.
COL_DOMAIN = 3
COL_URL = 4
COL_THEMES = 7
COL_LOCATIONS = 9
COL_TONE = 15        # comma-separated; the last value is the article's word count
COL_IMAGE = 18
COL_EXTRAS = 26

TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)


@dataclass
class Article:
    url: str
    domain: str
    title: str
    seen_at: datetime
    image: str | None
    themes: set[str]
    countries: set[str]   # FIPS country codes, e.g. "US", "UK"
    word_count: int


def latest_timestamp(client: httpx.Client) -> datetime:
    """Read lastupdate.txt to find the newest published file's timestamp."""
    text = client.get(LAST_UPDATE).raise_for_status().text
    # Each line: <size> <md5> <url>; the gkg file name starts with the timestamp.
    url = next(line.split()[-1] for line in text.splitlines() if ".gkg." in line)
    stamp = url.rsplit("/", 1)[-1].split(".")[0]
    return datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def file_urls(latest: datetime, count: int) -> list[str]:
    """URLs of the newest `count` GKG files, newest first, stepping back 15 minutes each."""
    stamps = [latest - timedelta(minutes=15 * i) for i in range(count)]
    return [f"{BASE}/{s:%Y%m%d%H%M%S}.gkg.csv.zip" for s in stamps]


def parse_row(line: str) -> Article | None:
    cols = line.split("\t")
    if len(cols) <= COL_EXTRAS or cols[COL_COLLECTION] != "1":
        return None
    match = TITLE_RE.search(cols[COL_EXTRAS])
    title = html.unescape(match.group(1)).strip() if match else ""
    if not title:
        return None
    return Article(
        url=cols[COL_URL],
        domain=cols[COL_DOMAIN].lower(),
        title=title,
        seen_at=datetime.strptime(cols[COL_DATE], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc),
        image=cols[COL_IMAGE] or None,
        themes={t for t in cols[COL_THEMES].split(";") if t},
        countries=parse_countries(cols[COL_LOCATIONS]),
        word_count=parse_word_count(cols[COL_TONE]),
    )


def parse_word_count(field: str) -> int:
    try:
        return int(float(field.rsplit(",", 1)[-1]))
    except ValueError:
        return 0


def parse_countries(field: str) -> set[str]:
    # Locations are ';'-separated; each is '#'-separated: type#name#country_code#...
    return {parts[2] for loc in field.split(";") if len(parts := loc.split("#")) > 2 and parts[2]}


def parse_file(data: bytes) -> list[Article]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        text = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
    return [a for line in text.splitlines() if (a := parse_row(line))]


@dataclass
class FetchResult:
    articles: list[Article]
    read: list[str]      # file URLs read this run
    window: list[str]    # every file URL in the lookback window


def fetch(lookback: int, done: set[str]) -> FetchResult:
    """Read every GKG file in the lookback window that is not in `done`.

    A file that is not uploaded yet is skipped and retried by the next run.
    """
    result = FetchResult([], [], [])
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        result.window = file_urls(latest_timestamp(client), lookback)
        for url in result.window:
            if url in done:
                continue
            resp = client.get(url)
            if resp.status_code != 200:
                log.warning("GDELT file %s returned %s; will retry next run", url, resp.status_code)
                continue
            result.articles.extend(parse_file(resp.content))
            result.read.append(url)
    return result


def load_done() -> set[str]:
    return set(DONE_FILE.read_text().split()) if DONE_FILE.exists() else set()


def save_done(done: set[str], result: FetchResult) -> None:
    """Record files read this run. Only names inside the window are kept, so the file stays small."""
    keep = (done | set(result.read)) & set(result.window)
    DONE_FILE.parent.mkdir(exist_ok=True)
    DONE_FILE.write_text("\n".join(sorted(keep)) + "\n")
