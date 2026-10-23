from datetime import datetime, timezone

from newsagg import gdelt


def make_row(**overrides) -> str:
    cols = [""] * 27
    cols[gdelt.COL_DATE] = "20261023161500"
    cols[gdelt.COL_COLLECTION] = "1"
    cols[gdelt.COL_DOMAIN] = "Example.com"
    cols[gdelt.COL_URL] = "https://example.com/news/story"
    cols[gdelt.COL_THEMES] = "ENV_CLIMATECHANGE;ECON_INFLATION;"
    cols[gdelt.COL_LOCATIONS] = "1#United States#US#US#39.8#-98.5#US;4#London, UK#UK#UKH9#51.5#-0.1#-2601889"
    cols[gdelt.COL_TONE] = "-5.03,3.28,8.31,11.59,21.66,0,421"
    cols[gdelt.COL_IMAGE] = "https://example.com/img.jpg"
    cols[gdelt.COL_EXTRAS] = "<PAGE_TITLE>Nestl&#xE9; reports Q3</PAGE_TITLE>"
    for k, v in overrides.items():
        cols[getattr(gdelt, f"COL_{k.upper()}")] = v
    return "\t".join(cols)


def test_parse_row_extracts_fields():
    a = gdelt.parse_row(make_row())
    assert a.title == "Nestlé reports Q3"          # HTML entity decoded
    assert a.domain == "example.com"               # lowercased
    assert a.seen_at == datetime(2026, 10, 23, 16, 15, tzinfo=timezone.utc)
    assert a.themes == {"ENV_CLIMATECHANGE", "ECON_INFLATION"}
    assert a.countries == {"US", "UK"}
    assert a.word_count == 421


def test_parse_row_skips_non_web_and_untitled():
    assert gdelt.parse_row(make_row(collection="2")) is None
    assert gdelt.parse_row(make_row(extras="")) is None


def test_save_done_keeps_only_window(tmp_path, monkeypatch):
    monkeypatch.setattr(gdelt, "DONE_FILE", tmp_path / "done.txt")
    result = gdelt.FetchResult(articles=[], read=["b"], window=["a", "b", "c"])
    gdelt.save_done({"a", "old"}, result)       # "old" fell out of the window
    assert gdelt.load_done() == {"a", "b"}      # "c" was missing, so it is retried


def test_file_urls_step_back_15_minutes():
    latest = datetime(2026, 10, 23, 16, 0, tzinfo=timezone.utc)
    assert gdelt.file_urls(latest, 3) == [
        f"{gdelt.BASE}/20261023160000.gkg.csv.zip",
        f"{gdelt.BASE}/20261023154500.gkg.csv.zip",
        f"{gdelt.BASE}/20261023153000.gkg.csv.zip",
    ]
