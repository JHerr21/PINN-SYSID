import json

import numpy as np
import pytest
import torch
import yaml

from sz.runs import make_run_dir, set_seed


def _draw(seed):
    rng = set_seed(seed)
    return torch.randn(100), rng.standard_normal(100)


def test_same_seed_reproduces_draws():
    t1, n1 = _draw(123)
    t2, n2 = _draw(123)
    assert torch.equal(t1, t2)
    assert np.array_equal(n1, n2)


def test_different_seeds_differ():
    t1, n1 = _draw(1)
    t2, n2 = _draw(2)
    assert not torch.equal(t1, t2)
    assert not np.array_equal(n1, n2)


def test_run_dir_records_config_commit_and_seeds(tmp_path):
    cfg = {"name": "nominal", "seed": 7}
    run_dir = make_run_dir(cfg, results_root=tmp_path, run_name="test_run")

    assert run_dir == tmp_path / "test_run"

    with (run_dir / "config.yaml").open(encoding="utf-8") as f:
        assert yaml.safe_load(f) == cfg

    with (run_dir / "meta.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["seeds"] == {"torch": 7, "numpy": 7}
    assert "git_commit" in meta
    commit = meta["git_commit"]
    assert commit is None or (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit))


def test_run_dir_never_overwrites(tmp_path):
    cfg = {"name": "nominal", "seed": 0}
    make_run_dir(cfg, results_root=tmp_path, run_name="dup")
    with pytest.raises(FileExistsError):
        make_run_dir(cfg, results_root=tmp_path, run_name="dup")


def test_run_dir_requires_name_and_seed(tmp_path):
    with pytest.raises(KeyError):
        make_run_dir({"name": "nominal"}, results_root=tmp_path, run_name="x")
