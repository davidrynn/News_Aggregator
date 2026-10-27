"""Carry a run's memory between GitHub Actions runs, on the `state` branch.

Each Actions run starts on an empty machine, but a run needs what the last one left:
the articles already scored, the GDELT files already read, the stories on the page.
save() pushes those files as the one commit of the `state` branch (force-pushed, like
gh-pages), and load() writes them back into data/. Both use git's low-level commands,
so the working folder and the current branch are never touched.
"""
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from newsagg.config import ROOT
from newsagg.pages import git, tree_listing

log = logging.getLogger(__name__)

BRANCH = "state"
DATA = ROOT / "data"
FILES = [
    "seen.db",            # articles already scored or triaged low
    "gdelt_done.txt",     # GDELT files already read
    "published.json",     # the stories on the page, with their vectors
    "same_event.json",    # same-event answers for stories on the page
    "verdicts.jsonl",     # every verdict; the golden set is labeled from it
    "iffy.csv",           # the blocklist, so a failed download falls back to it
]


def save(remote: str = "origin", data: Path = DATA) -> str:
    """Push data/'s state files as the state branch. Returns the new commit's hash."""
    # --no-filters stores the bytes as they are: no line-ending changes to seen.db or the logs.
    blobs = {name: git("hash-object", "-w", "--no-filters", str(data / name))
             for name in FILES if (data / name).exists()}
    tree = git("mktree", stdin=tree_listing(blobs))
    commit = git("commit-tree", tree, "-m", f"State {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    git("push", "--force", remote, f"{commit}:refs/heads/{BRANCH}")
    log.info(f"saved {len(blobs)} state files to {BRANCH} ({commit[:7]})")
    return commit


def load(remote: str = "origin", data: Path = DATA) -> bool:
    """Write the state branch's files into data/. False if there is no state branch yet."""
    try:
        git("fetch", "-q", remote, BRANCH)
    except subprocess.CalledProcessError:
        log.warning(f"no {BRANCH} branch on {remote}; starting with empty state")
        return False
    data.mkdir(exist_ok=True)
    names = git("ls-tree", "--name-only", "FETCH_HEAD").split()
    for name in names:
        blob = subprocess.run(["git", "cat-file", "blob", f"FETCH_HEAD:{name}"],
                              cwd=ROOT, capture_output=True, check=True).stdout
        (data / name).write_bytes(blob)
    # Git keeps no file dates, so the blocklist would look new forever and never refresh.
    # Dated long ago, it is downloaded again each run, with this copy as the fallback.
    if (data / "iffy.csv").exists():
        os.utime(data / "iffy.csv", (0, 0))
    log.info(f"loaded {len(names)} state files from {BRANCH}")
    return True
