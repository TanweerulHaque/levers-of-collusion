"""Which difficulty label, if any, the judging agent is told about its peer's task.

Each agent sees the label of its peer's task in the communication-phase brief, and
never the label of its own task, so the treatment is symmetric:

``none`` (default, the original protocol): no label.

``true``: the measured difficulty of the peer's task (``true_label`` in the label file).

``random``: the same labels permuted across tasks (``random_label``), so the label an
agent sees is independent of how hard the peer's task really is.

Labels come from a file written by ``analysis/task_difficulty_labels.py``.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DIFFICULTY_LABEL_MODES = ("none", "true", "random")
DEFAULT_DIFFICULTY_LABEL_MODE = "none"
DIFFICULTY_LEVELS = ("easy", "medium", "hard")
DEFAULT_LABEL_FILE = "task/difficulty_labels/task_sequences_50x10_rep001-005.json"

_MODE_FIELD = {"true": "true_label", "random": "random_label"}


@lru_cache(maxsize=None)
def _read_label_file(path: str) -> dict[str, dict[str, Any]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = document.get("tasks") if isinstance(document, dict) else None
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"{path} is not a difficulty-label file: no \"tasks\" object")
    return tasks


def load_difficulty_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    """``task_id -> entry`` with ``true_label`` and ``random_label``."""
    return _read_label_file(str(Path(path).resolve()))


def shown_label(mode: str, path: str | Path, task_id: str) -> str | None:
    """The label shown to the agent judging ``task_id``, or ``None`` under ``none``."""
    if mode not in DIFFICULTY_LABEL_MODES:
        raise ValueError(f"Unknown difficulty label mode: {mode}")
    if mode == "none":
        return None
    return _entry(path, task_id)[_MODE_FIELD[mode]]


def true_label(path: str | Path, task_id: str) -> str | None:
    """The measured label of ``task_id``, or ``None`` when the file does not cover it."""
    if not path:
        return None
    try:
        entry = load_difficulty_labels(path).get(task_id)
    except (OSError, ValueError):
        return None
    return entry.get("true_label") if entry else None


def _entry(path: str | Path, task_id: str) -> dict[str, Any]:
    entry = load_difficulty_labels(path).get(task_id)
    if entry is None:
        raise ValueError(
            f"Task {task_id!r} has no difficulty label in {path}; regenerate it with "
            "analysis/task_difficulty_labels.py over the sequences this run uses."
        )
    return entry


def check_label_coverage(mode: str, path: str | Path, task_ids: list[str]) -> None:
    """Fail before any LLM call when a labelled run would meet an unlabelled task."""
    if mode == "none":
        return
    missing = sorted(set(task_ids) - set(load_difficulty_labels(path)))
    if missing:
        raise ValueError(
            f"--peer-difficulty-label {mode}: {len(missing)} task(s) have no label in "
            f"{path}: {', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}. "
            "Regenerate it with analysis/task_difficulty_labels.py over these sequences."
        )
