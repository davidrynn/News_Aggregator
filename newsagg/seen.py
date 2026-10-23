"""Rule 2: an article already scored is not scored again. Kept in data/seen.db (SQLite)."""
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

from newsagg.config import ROOT
from newsagg.output import story_id

DB = ROOT / "data" / "seen.db"
KEEP_DAYS = 7   # GDELT rarely re-lists an article after a day; a week is plenty


class Seen:
    """URLs are stored as story_id hashes, so query strings and fragments don't matter."""

    def __init__(self, path: Path = DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY, seen_at REAL NOT NULL)")

    def __contains__(self, url: str) -> bool:
        return self.db.execute("SELECT 1 FROM seen WHERE id = ?", (story_id(url),)).fetchone() is not None

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM seen").fetchone()[0]

    def add(self, urls: Iterable[str], now: float | None = None) -> None:
        now = now or time.time()
        self.db.executemany("INSERT OR IGNORE INTO seen VALUES (?, ?)", ((story_id(u), now) for u in urls))
        self.db.commit()

    def prune(self, days: float = KEEP_DAYS, now: float | None = None) -> int:
        cutoff = (now or time.time()) - days * 86400
        removed = self.db.execute("DELETE FROM seen WHERE seen_at < ?", (cutoff,)).rowcount
        self.db.commit()
        return removed

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Seen":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
