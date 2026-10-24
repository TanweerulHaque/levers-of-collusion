"""Load a YAML run configuration into argparse defaults.

Config values become defaults, so an explicit command-line flag always wins. Each
command reads the sections it understands and ignores keys belonging to the others.
"""

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

from experiments.protocol.channel_restore import (
    DEFAULT_RESTORE_CHANNEL_COST_PERCENT,
    format_percent,
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


def _reward_suffix(
    reward: object = True,
    reward_scope: object = None,
    reward_type: object = None,
    reward_penalty: object = None,
) -> str:
    """Name the reward settings that differ from the defaults, or "" when none do."""
    if reward is False:
        return "no_reward"
    parts = [
        f"reward_{name}_{str(value).replace('-', '_')}"
        for name, value, default in (
            ("scope", reward_scope, DEFAULT_REWARD_SHARING),
            ("type", reward_type, DEFAULT_REWARD_TERMS),
            ("penalty", reward_penalty, DEFAULT_REWARD_PENALTY),
        )
        if value not in (None, "", default)
    ]
    return "_".join(parts)


def results_dir_name(
    alice_model: str,
    bob_model: str,
    *,
    restore_channel_tool: object = False,
    restore_channel_cost_percent: object = None,
    reward: object = True,
    reward_scope: object = None,
    reward_type: object = None,
    reward_penalty: object = None,
) -> str:
    """Name the results folder holding one condition's runs.

    The pair slug is followed by the paid channel restore and its cost when the run
    offers it (``_restore_channel_link_cost50``) and by every reward setting that
    differs from the default (``_reward_penalty_per_term``), joined with ``_with_``
    when there are both. Runs that differ in either never share a folder; a default
    run keeps the plain pair slug. Run folders inside keep their own names.
    """
    slug = model_slug(alice_model, bob_model)
    experiment = ""
    if restore_channel_tool:
        cost = float(
            DEFAULT_RESTORE_CHANNEL_COST_PERCENT
            if restore_channel_cost_percent in (None, "")
            else restore_channel_cost_percent
        )
        experiment = f"restore_channel_link_cost{format_percent(cost)}"
    reward_part = _reward_suffix(reward, reward_scope, reward_type, reward_penalty)
    if experiment and reward_part:
        return f"{slug}_{experiment}_with_{reward_part}"
    if experiment or reward_part:
        return f"{slug}_{experiment or reward_part}"
    return slug


def results_dir_name_from_defaults(defaults: dict[str, Any]) -> str:
    """``results_dir_name`` for the flattened settings ``config_defaults`` returns."""
    return results_dir_name(
        defaults.get("alice_model", ""),
        defaults.get("bob_model", ""),
        restore_channel_tool=defaults.get("restore_channel_tool", False),
        restore_channel_cost_percent=defaults.get("restore_channel_cost_percent"),
        reward=defaults.get("reward", True),
        reward_scope=defaults.get("reward_scope"),
        reward_type=defaults.get("reward_type"),
        reward_penalty=defaults.get("reward_penalty"),
    )
