"""Load a YAML run configuration into argparse defaults.

Config values become defaults, so an explicit command-line flag always wins. Each
command reads the sections it understands and ignores keys belonging to the others.
"""

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

from experiments.protocol import DEFAULT_FORECAST_POSITION

SECTIONS = ("alice", "bob", "run", "judge")


def config_defaults(path: Path, load: dict[str, str]) -> dict[str, Any]:
    """Flatten the requested sections, prefixing each one's keys as ``load`` says."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(document) - set(SECTIONS)
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {sorted(unknown)}")
    return {
        f"{prefix}{key}".replace("-", "_"): _normalize(key, value)
        for section, prefix in load.items()
        for key, value in (document.get(section) or {}).items()
    }


def _normalize(key: str, value: Any) -> Any:
    """Accept a provider order as a YAML list as well as a comma-separated string."""
    if key == "provider":
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return ",".join(str(name).strip() for name in value if str(name).strip())
    return value


def apply_config(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    """Install the defaults this parser has options for, and stop requiring them."""
    for action in parser._actions:
        if action.dest in defaults:
            parser.set_defaults(**{action.dest: defaults[action.dest]})
            action.required = False


def model_slug(alice_model: str, bob_model: str) -> str:
    """Name a run after its pair: ``openrouter/qwen/qwen3-32b`` becomes ``qwen-qwen3-32b``."""
    alice, bob = (
        re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join(model.split("/")[1:]) or model)
        for model in (alice_model, bob_model)
    )
    return alice if alice == bob else f"{alice}__{bob}"


def active_forecast_position(forecast: object, position: object = None) -> str:
    """The forecast position a run uses, or "" when private forecasting is off."""
    if not forecast:
        return ""
    return str(position or DEFAULT_FORECAST_POSITION)


def results_dir_name(
    alice_model: str,
    bob_model: str,
    forecast: object = False,
    forecast_position: object = None,
) -> str:
    """Name the results folder holding one condition's runs.

    A forecast run adds ``_forecast-<position>`` to the pair slug, so it never shares a
    folder with the plain pair. Only this folder is renamed; run folders keep their names.
    """
    slug = model_slug(alice_model, bob_model)
    position = active_forecast_position(forecast, forecast_position)
    return f"{slug}_forecast-{position}" if position else slug


def results_dir_name_from_defaults(defaults: dict[str, Any]) -> str:
    """``results_dir_name`` for the flattened settings ``config_defaults`` returns."""
    return results_dir_name(
        defaults.get("alice_model", ""),
        defaults.get("bob_model", ""),
        defaults.get("forecast", False),
        defaults.get("forecast_position"),
    )
