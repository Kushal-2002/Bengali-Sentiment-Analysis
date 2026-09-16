"""YAML config loading with dotted-key CLI overrides."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> dict:
    """Load ``path`` and apply ``key.sub=value`` overrides (values parsed as YAML)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"unknown config section {part!r} in {key!r}")
            node = node[part]
        if leaf not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[leaf] = yaml.safe_load(raw)
    return cfg


def flatten(cfg: dict, prefix: str = "") -> dict:
    """``{"a": {"b": 1}}`` -> ``{"a.b": 1}`` (for logging as MLflow params)."""
    out = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
