"""Publish out/ to GitHub Pages by pushing it as the gh-pages branch.

The branch holds only the site files and is replaced on every publish (one commit,
force-pushed): old copies of news.json have no use, and keeping them would grow the
repo by ~100 KB a run. Git's low-level commands build the commit straight from out/,
so the working folder, the current branch and main's history are never touched.
"""
import logging
import subprocess
from datetime import datetime, timezone

from newsagg.config import ROOT, Config

log = logging.getLogger(__name__)

FILES = ["index.html", "news.json"]
BRANCH = "gh-pages"


def git(*args: str, stdin: str | None = None) -> str:
    # Bytes, not text mode: on Windows, text mode turns each line ending into CR+LF on the
    # way in, and mktree would then make the CR part of every file name.
    data = stdin.encode() if stdin is not None else None
    result = subprocess.run(["git", *args], cwd=ROOT, input=data, capture_output=True, check=True)
    return result.stdout.decode().strip()


def tree_listing(blobs: dict[str, str]) -> str:
    """mktree input: one 'mode type hash<TAB>name' line per file."""
    return "".join(f"100644 blob {sha}\t{name}\n" for name, sha in sorted(blobs.items()))


def publish(cfg: Config, remote: str = "origin") -> str:
    """Push out/'s site files to the gh-pages branch. Returns the new commit's hash."""
    blobs = {name: git("hash-object", "-w", str(cfg.out_dir / name)) for name in FILES}
    # GitHub Pages runs files through Jekyll unless .nojekyll exists; we want them as-is.
    blobs[".nojekyll"] = git("hash-object", "-w", "--stdin", stdin="")
    tree = git("mktree", stdin=tree_listing(blobs))
    commit = git("commit-tree", tree, "-m", f"Publish {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    git("push", "--force", remote, f"{commit}:refs/heads/{BRANCH}")
    log.info(f"published to {BRANCH} ({commit[:7]})")
    return commit
