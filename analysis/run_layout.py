"""Where a ``run.json`` sits under ``results/``, for both folder layouts.

* ``launch`` (current): ``results/<pair>/<label>_<datetime>/rep<N>/run.json``, one
  folder per launch and one ``rep<N>`` inside it per repeat.
* ``flat`` (older): ``results/<pair>/run_<label>_rep<N>_<datetime>[_<micro>_<hash>]/run.json``,
  one folder per repeat, the repeat number inside its name.

Everything that has to group runs by model pair, condition and repeat goes through
``describe`` so the two layouts are read the same way.
"""

from __future__ import annotations

import re
from pathlib import Path

REP_DIR_RE = re.compile(r"rep(\d+)")
FLAT_REP_RE = re.compile(r"(?:^|_)rep(\d+)(?=_|$)")
# ``_<YYYYMMDD>_<HHMMSS>`` with the optional ``_<micro>_<hash>`` older runs carry.
STAMP_RE = re.compile(r"_(\d{8}_\d{6})(?:_\d+_[0-9a-f]+)?$")

# Globs, relative to ``results/<pair>``, that together find every run of either layout.
RUN_GLOBS = ("*/rep*/run.json", "run_*/run.json")


def describe(path: Path) -> dict:
    """Layout, model pair, launch, repeat and condition of the run at ``path``."""
    run_dir = path.parent if path.name == "run.json" else path
    rep_match = REP_DIR_RE.fullmatch(run_dir.name)
    if rep_match:
        launch_dir = run_dir.parent
        stamp = STAMP_RE.search(launch_dir.name)
        label = STAMP_RE.sub("", launch_dir.name)
        return {
            "layout": "launch",
            "pair": launch_dir.parent.name,
            "launch": launch_dir.name,
            "rep": int(rep_match.group(1)),
            "label": label,
            "stamp": stamp.group(1) if stamp else "",
            # Launches of one label stay separate conditions, as each is its own sample.
            "condition": launch_dir.name,
        }
    stamp = STAMP_RE.search(run_dir.name)
    label = re.sub(r"^run_", "", STAMP_RE.sub("", run_dir.name))
    rep = FLAT_REP_RE.search(label)
    return {
        "layout": "flat",
        "pair": run_dir.parent.name,
        "launch": run_dir.name,
        "rep": int(rep.group(1)) if rep else None,
        "label": label,
        "stamp": stamp.group(1) if stamp else "",
        "condition": FLAT_REP_RE.sub("", label),
    }
