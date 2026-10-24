"""The Iffy.news blocklist plus the user's own additions (rule 1).

The list is cached in data/iffy.csv and re-downloaded every `refresh_days`. If a
download fails, the cached copy is used.
"""
import csv
import io
import logging
import time
from urllib.parse import urlsplit

import httpx
import tldextract

from newsagg.config import ROOT, Config

log = logging.getLogger(__name__)

CACHE = ROOT / "data" / "iffy.csv"

# suffix_list_urls=() uses tldextract's bundled public-suffix snapshot: no network call.
_extract = tldextract.TLDExtract(suffix_list_urls=())


def normalize_host(host: str) -> str:
    host = host.lower().strip().split(":")[0].rstrip(".")
    for prefix in ("www.", "m.", "amp."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host


def registered_domain(host: str) -> str:
    """news.bbc.co.uk -> bbc.co.uk: the part an outlet registers."""
    return _extract(host).top_domain_under_public_suffix or host


def parent_hosts(host: str) -> list[str]:
    """The host and each parent down to its registered domain.

    news.example.co.uk -> [news.example.co.uk, example.co.uk]. We stop at the
    registered domain so we never check bare suffixes like co.uk.
    """
    registered = registered_domain(host)
    labels = host.split(".")
    hosts = [".".join(labels[i:]) for i in range(len(labels))]
    return hosts[: hosts.index(registered) + 1] if registered in hosts else [host]


class Blocklist:
    """Entries are matched as written: a host (blocks it and its subdomains) or a
    host/path prefix like westernjournal.com/commentary (blocks only that section)."""

    def __init__(self, entries: list[str], allow: list[str] = ()):
        self.hosts: set[str] = set()
        self.paths: dict[str, list[str]] = {}
        for entry in entries:
            host, _, path = entry.strip().lower().partition("/")
            host = normalize_host(host)
            if not host:
                continue
            if path:
                self.paths.setdefault(host, []).append("/" + path.rstrip("/"))
            else:
                self.hosts.add(host)
        self.allow = {normalize_host(a) for a in allow}

    def __len__(self) -> int:
        return len(self.hosts) + sum(len(p) for p in self.paths.values())

    def is_blocked(self, url: str) -> bool:
        parts = urlsplit(url)
        candidates = parent_hosts(normalize_host(parts.hostname or ""))
        if self.allow.intersection(candidates):
            return False
        if self.hosts.intersection(candidates):
            return True
        path = parts.path.lower()
        return any(
            path == prefix or path.startswith(prefix + "/")
            for host in candidates
            for prefix in self.paths.get(host, ())
        )


def parse_iffy(text: str) -> list[str]:
    return [row["Domain"] for row in csv.DictReader(io.StringIO(text)) if row.get("Domain")]


def refresh(cfg: Config) -> None:
    """Download the Iffy CSV if the cache is missing or older than refresh_days."""
    age_days = (time.time() - CACHE.stat().st_mtime) / 86400 if CACHE.exists() else None
    if age_days is not None and age_days < cfg.blocklist.refresh_days:
        return
    try:
        resp = httpx.get(cfg.blocklist.iffy_url, timeout=60, follow_redirects=True).raise_for_status()
        if len(parse_iffy(resp.text)) < 500:  # guard against an error page saved as the list
            raise ValueError("downloaded Iffy list looks truncated")
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(resp.text, encoding="utf-8")
        log.info("Iffy list refreshed")
    except (httpx.HTTPError, ValueError) as e:
        if not CACHE.exists():
            raise RuntimeError("No Iffy list cached and download failed") from e
        log.warning("Iffy download failed (%s); using cached copy", e)


def load(cfg: Config) -> Blocklist:
    refresh(cfg)
    entries = parse_iffy(CACHE.read_text(encoding="utf-8")) + cfg.blocklist.extra
    return Blocklist(entries, cfg.blocklist.allow)
