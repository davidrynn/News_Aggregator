import subprocess

from newsagg import config, pages, state


def test_tree_listing_format():
    listing = pages.tree_listing({"news.json": "b" * 40, "index.html": "a" * 40})
    assert listing == f"100644 blob {'a' * 40}\tindex.html\n100644 blob {'b' * 40}\tnews.json\n"


def test_publish_pushes_only_site_files(tmp_path, monkeypatch):
    # A throwaway repo plays the part of GitHub, so nothing real is pushed.
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text("<html></html>")
    (out / "news.json").write_text('{"stories": []}')
    (out / "notes.txt").write_text("not part of the site")
    cfg = config.load().model_copy(update={"output": config.Output(dir=str(out))})
    pages.publish(cfg, remote=str(remote))
    pages.publish(cfg, remote=str(remote))            # replaces, doesn't stack
    files = subprocess.run(["git", "--git-dir", str(remote), "ls-tree", "--name-only", "gh-pages"],
                           capture_output=True, text=True, check=True).stdout.split()
    count = subprocess.run(["git", "--git-dir", str(remote), "rev-list", "--count", "gh-pages"],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert files == [".nojekyll", "index.html", "news.json"]
    assert count == "1"


def test_state_round_trip_through_a_remote(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    (before / "seen.db").write_bytes(b"SQLite\x00\r\n\x01")        # binary, with a CR LF to keep
    (before / "iffy.csv").write_text("domain\nexample.com\n")
    (before / "notes.txt").write_text("not state")
    assert state.load(remote=str(remote), data=after) is False    # no state branch yet
    state.save(remote=str(remote), data=before)
    state.save(remote=str(remote), data=before)                   # replaces, doesn't stack
    assert state.load(remote=str(remote), data=after) is True
    assert sorted(p.name for p in after.iterdir()) == ["iffy.csv", "seen.db"]
    assert (after / "seen.db").read_bytes() == b"SQLite\x00\r\n\x01"
    assert (after / "iffy.csv").stat().st_mtime == 0              # so the blocklist refreshes
    count = subprocess.run(["git", "--git-dir", str(remote), "rev-list", "--count", "state"],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert count == "1"
