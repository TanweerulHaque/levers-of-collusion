"""Load a YAML run configuration into argparse defaults.

Config values become defaults, so an explicit command-line flag always wins. Each
command reads the sections it understands and ignores keys belonging to the others.
"""

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

from experiments.protocol.message_cost import (
    DEFAULT_MESSAGE_COST_RATE,
    rate_slug,
)
from experiments.protocol.rewards import (
    DEFAULT_REWARD_PENALTY,
    DEFAULT_REWARD_SHARING,
    DEFAULT_REWARD_TERMS,
)

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


def results_dir_name(
    alice_model: str,
    bob_model: str,
    *,
    message_cost: object = False,
    message_cost_rate: object = None,
    reward_scope: object = None,
    reward_type: object = None,
    reward_penalty: object = None,
) -> str:
    """Name the results folder holding one condition's runs.

    A run without ``--message-cost`` keeps the plain pair slug, as it always did. A
    message-cost run adds ``_message_cost_rate0_005`` (the rate, with no dot) and every
    reward setting that differs from the default (``_with_reward_penalty_per_term``),
    so the same experiment under another reward or rate never shares a folder.
    """
    slug = model_slug(alice_model, bob_model)
    if not message_cost:
        return slug
    rate = float(
        DEFAULT_MESSAGE_COST_RATE if message_cost_rate in (None, "") else message_cost_rate
    )
    name = f"{slug}_message_cost_rate{rate_slug(rate)}"
    reward_parts = [
        f"reward_{label}_{str(value).replace('-', '_')}"
        for label, value, default in (
            ("scope", reward_scope, DEFAULT_REWARD_SHARING),
            ("type", reward_type, DEFAULT_REWARD_TERMS),
            ("penalty", reward_penalty, DEFAULT_REWARD_PENALTY),
        )
        if value not in (None, "", default)
    ]
    return f"{name}_with_{'_'.join(reward_parts)}" if reward_parts else name
