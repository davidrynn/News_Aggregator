import subprocess

from newsagg import config, pages


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
