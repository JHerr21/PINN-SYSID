from pathlib import Path

import pytest

from sz.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_nominal_config_loads():
    cfg = load_config(CONFIGS / "nominal.yaml")
    assert cfg["name"] == "nominal"
    assert isinstance(cfg["seed"], int)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_non_mapping_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)
