"""Config loading. Every hyperparameter comes from a YAML file, never a source constant."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


class Config(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


def _wrap(o: Any) -> Any:
    if isinstance(o, dict):
        return Config({k: _wrap(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_wrap(v) for v in o]
    return o


def load(name: str = "default.yaml", **overrides: Any) -> Config:
    path = Path(name)
    if not path.exists():
        path = CONFIG_DIR / name
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = _wrap(raw)
    for dotted, value in overrides.items():
        node = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return cfg


def clone(cfg: Config) -> Config:
    return _wrap(copy.deepcopy(dict(cfg)))
