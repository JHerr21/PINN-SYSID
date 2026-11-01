"""Load experiment configs from YAML files in configs/."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a YAML config file and return it as a dict.

    Raises FileNotFoundError if the file does not exist, and ValueError if
    the file does not contain a YAML mapping (key: value pairs) at the top
    level.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config file {path} must contain a mapping at the top level, "
            f"got {type(cfg).__name__}"
        )

    return cfg
