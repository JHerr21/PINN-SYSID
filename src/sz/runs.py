"""Seeding and per-run result folders.

Every run saves its config, the git commit, and the PyTorch and NumPy
seeds in its own folder under results/, so any result can be rebuilt
from results/ alone.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml

# src/sz/runs.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> np.random.Generator:
    """Seed PyTorch (CPU and all GPUs) and return a seeded NumPy Generator.

    Use the returned Generator for all NumPy randomness in the run.
    """
    seed = int(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def git_commit(repo_root: Union[str, Path] = REPO_ROOT) -> Optional[str]:
    """Return the current git commit hash, or None if it cannot be read
    (git not installed, not a git repository, or no commits yet)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def make_run_dir(
    cfg: Dict[str, Any],
    results_root: Union[str, Path] = REPO_ROOT / "results",
    run_name: Optional[str] = None,
) -> Path:
    """Create a new run folder and record the config, git commit and seeds.

    The folder is results_root/run_name. If run_name is not given, it is
    "<cfg['name']>_seed<cfg['seed']>_<YYYYmmdd-HHMMSS>". The config must
    contain "name" and "seed". An existing folder is never overwritten.

    Writes:
        config.yaml  copy of the config
        meta.json    {"git_commit": ..., "seeds": {"torch": ..., "numpy": ...}}
    """
    for key in ("name", "seed"):
        if key not in cfg:
            raise KeyError(f"Config is missing required key: {key!r}")

    seed = int(cfg["seed"])
    if run_name is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{cfg['name']}_seed{seed}_{stamp}"

    run_dir = Path(results_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    meta = {
        "git_commit": git_commit(),
        "seeds": {"torch": seed, "numpy": seed},
    }
    with (run_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return run_dir
