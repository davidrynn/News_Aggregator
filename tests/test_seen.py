from newsagg.seen import Seen


def test_seen_ignores_query_string_and_survives_reopen(tmp_path):
    path = tmp_path / "seen.db"
    with Seen(path) as seen:
        seen.add(["https://Ex.com/a?utm=1"])
        assert "https://ex.com/a#top" in seen
        assert "https://ex.com/b" not in seen
    with Seen(path) as seen:            # a new run sees the last run's entries
        assert "https://ex.com/a" in seen


def test_prune_removes_only_old_entries(tmp_path):
    with Seen(tmp_path / "seen.db") as seen:
        seen.add(["https://ex.com/old"], now=1_000_000)
        seen.add(["https://ex.com/new"], now=1_000_000 + 8 * 86400)
        assert seen.prune(days=7, now=1_000_000 + 8 * 86400) == 1
        assert len(seen) == 1 and "https://ex.com/new" in seen
