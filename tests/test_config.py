import pytest
from pydantic import ValidationError

from newsagg import config


def test_loads_project_config():
    cfg = config.load()
    assert cfg.provider.name == "ollama"
    assert cfg.scoring.min_newsworthiness == 3


def test_rejects_out_of_range_threshold(tmp_path):
    text = config.DEFAULT_PATH.read_text().replace("min_relevance = 3", "min_relevance = 9")
    bad = tmp_path / "config.toml"
    bad.write_text(text)
    with pytest.raises(ValidationError):
        config.load(bad)
