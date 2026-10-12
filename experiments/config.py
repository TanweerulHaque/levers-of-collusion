"""Load a YAML run configuration into argparse defaults.

Config values become defaults, so an explicit command-line flag always wins. Each
command reads the sections it understands and ignores keys belonging to the others.
"""

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

from experiments.protocol.announcements import ANNOUNCEMENT_CHANNELS

SECTIONS = ("alice", "bob", "run", "judge")


def config_defaults(path: Path, load: dict[str, str]) -> dict[str, Any]:
    """Flatten the requested sections, prefixing each one's keys as ``load`` says."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(document) - set(SECTIONS)
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {sorted(unknown)}")
    return {
        f"{prefix}{key}".replace("-", "_"): value
        for section, prefix in load.items()
        for key, value in (document.get(section) or {}).items()
    }


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
    announcement: str = "off",
    channels: "list[str] | tuple[str, ...] | None" = None,
) -> str:
    """The pair folder under ``results/``, tagged with the episode announcement in use.

    No announcement keeps the plain model-pair name. ``countdown`` with every channel
    appends ``_episode_announcement_all``; ``final`` adds the mode
    (``_episode_announcement_final_all``) and a partial channel set names its channels
    (``_episode_announcement_feedback``), so runs that differ in what agents are told
    never share a folder.
    """
    name = model_slug(alice_model, bob_model)
    if not announcement or announcement == "off":
        return name
    chosen = [c for c in ANNOUNCEMENT_CHANNELS if c in (channels or ANNOUNCEMENT_CHANNELS)]
    where = "all" if len(chosen) == len(ANNOUNCEMENT_CHANNELS) else "-".join(chosen)
    mode = "" if announcement == "countdown" else f"{announcement}_"
    return f"{name}_episode_announcement_{mode}{where}"
