"""YAML config loader with env-var override and pydantic validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base* (override wins)."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str | Path, model: type[BaseModel]) -> BaseModel:
    """
    Load a YAML file and validate it against *model*.

    Any key in the YAML that has the form ``some_section.some_key``
    can also be overridden by the env var ``MOONLAKE_SOME_SECTION__SOME_KEY``
    (double-underscore separator, all upper-case).
    """
    raw = load_yaml(config_path)
    # Env-var overrides: MOONLAKE_GENERATION__TEMPERATURE=0.5
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("MOONLAKE_"):
            continue
        parts = env_key[len("MOONLAKE_"):].lower().split("__")
        node = raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # Try to coerce numeric / bool strings
        coerced: Any = env_val
        if env_val.lower() in ("true", "false"):
            coerced = env_val.lower() == "true"
        else:
            try:
                coerced = int(env_val)
            except ValueError:
                try:
                    coerced = float(env_val)
                except ValueError:
                    pass
        node[parts[-1]] = coerced
    return model.model_validate(raw)
