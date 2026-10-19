#!/usr/bin/env python3
"""Label task difficulty from a manual annotation and earlier runs, and write the label file.

Two independent sources, combined into one label per task:

* **Manual annotation** (``--manual``): every task read and scored 1-5 against a fixed
  rubric (``task/difficulty_labels/annotation_rubric.md``) without looking at any run
  output. The score answers: how likely is an agent to get its own answer exactly right?
* **Measured difficulty** (``--results``): the share of earlier episodes in which the
  agent who did the task solved it (its own ``expected_verdict`` is ``pass``), pooled
  over every ``run.json``. Own-task correctness is settled in the task phase, before the
  peer brief that would carry a label, so labelled runs do not bias it. Smoothed as
  ``solve_rate = (solved + 1) / (attempts + 2)``.

Combination. Annotators of different task families need not share one 1-5 scale, but
within a family the manual order is consistent. So:

1. Within each task family, each source becomes a hardness percentile rank (manual
   score; ``1 - solve_rate``), ties sharing their mean rank, and the two are blended:
   ``family_rank = w * manual_rank + (1 - w) * measured_rank``, ``w = --manual-weight``
   (default 0.5).
2. The runs set the scale: a task's ``combined_hardness`` is the family's measured
   failure rate (``1 - solve_rate``) at quantile ``family_rank``. Each family keeps its
   measured level and spread; which task sits where inside it is the blended order.
3. Tertiles of ``combined_hardness`` over all unique tasks give ``true_label`` (top
   third ``hard``, bottom third ``easy``).

``manual_label`` and ``measured_label`` are within-family tertiles of each source alone,
kept for inspection with their rank correlations and the tasks two levels apart.

``random_label`` is ``true_label`` permuted across tasks with ``--seed``: the label
distribution is kept and its link to difficulty is broken.

The file only describes tasks; it never touches a sequence manifest. The experiment
reads it through ``--difficulty-labels`` (see ``experiments/protocol/difficulty_labels.py``).

Standard library only. Usage::

    python analysis/task_difficulty_labels.py
    python analysis/task_difficulty_labels.py --manual-weight 1      # manual annotation only
    python analysis/task_difficulty_labels.py --manual '' --manual-weight 0   # runs only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.protocol.difficulty_labels import (  # noqa: E402
    DEFAULT_LABEL_FILE,
    DIFFICULTY_LEVELS,
)
from experiments.tasks import display_task_id, read_task_pairs, task_type_of  # noqa: E402

DEFAULT_MANUAL_FILE = "task/difficulty_labels/manual_annotations.json"
# Directory names under results/ that hold scratch files, never trajectories.
SKIP_DIRS = {"tmp_tests", "logs"}


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def sequence_tasks(record: Path, start: int, repeats: int) -> list[dict]:
    """Every (repeat, episode, agent, task) slot of the chosen sequences, in order."""
    slots = []
    for rep in range(start, start + repeats):
        manifest = record / f"rep{rep:03d}_sampled_manifest.json"
        for episode, pair in enumerate(read_task_pairs(manifest, ROOT), start=1):
            for agent, task in zip(("alice", "bob"), pair):
                slots.append({"rep": rep, "episode": episode, "agent": agent, "task": task})
    return slots


def measure(results: Path) -> tuple[dict[str, list[int]], list[str]]:
    """``task_id -> [solved, attempts]`` over every graded own-task attempt in the runs."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    sources = []
    for path in sorted(results.rglob("run.json")):
        if SKIP_DIRS.intersection(path.relative_to(results).parts[:-1]):
            continue
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        models = (run.get("run_config") or {}).get("models") or {}
        used = False
        for episode in run.get("results") or []:
            for agent_id, agent in (episode.get("agents") or {}).items():
                # A scripted Bob replays a recorded answer; it is not a fresh attempt.
                if models.get(agent_id) == "controlled":
                    continue
                expected = agent.get("expected_verdict")
                if expected not in ("pass", "fail"):
                    continue
                count = counts[str(agent.get("task_id"))]
                count[0] += expected == "pass"
                count[1] += 1
                used = True
        if used:
            sources.append(_rel(path))
    return counts, sources


def read_manual(path: Path) -> dict[str, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    tasks = document.get("tasks") if isinstance(document, dict) else None
    if not isinstance(tasks, dict):
        raise SystemExit(f"{path} is not a manual annotation file: no \"tasks\" object")
    for task_id, entry in tasks.items():
        if entry.get("score") not in (1, 2, 3, 4, 5):
            raise SystemExit(f"{path}: {task_id} has no integer score 1-5")
    return tasks


def percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    """0 for the lowest value, 1 for the highest; tied values share their mean rank."""
    ordered = sorted(values, key=values.get)
    n = len(ordered)
    ranks: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[ordered[j + 1]] == values[ordered[i]]:
            j += 1
        mean_rank = (i + j) / 2
        for key in ordered[i:j + 1]:
            ranks[key] = mean_rank / (n - 1) if n > 1 else 0.5
        i = j + 1
    return ranks


def spearman(a: dict[str, float], b: dict[str, float]) -> float | None:
    """Rank correlation of two hardness measures over the same tasks."""
    ra, rb = percentile_ranks(a), percentile_ranks(b)
    keys = list(ra)
    if len(keys) < 3:
        return None
    ma, mb = sum(ra[k] for k in keys) / len(keys), sum(rb[k] for k in keys) / len(keys)
    cov = sum((ra[k] - ma) * (rb[k] - mb) for k in keys)
    va = sum((ra[k] - ma) ** 2 for k in keys)
    vb = sum((rb[k] - mb) ** 2 for k in keys)
    return round(cov / (va * vb) ** 0.5, 4) if va and vb else None


def quantile(ordered: list[float], q: float) -> float:
    """Linearly interpolated value at fraction ``q`` of a sorted list."""
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def tertile_labels(hardness: dict[str, float]) -> tuple[dict[str, str], tuple[float, float]]:
    """Top third of hardness is ``hard``, bottom third ``easy``; ties share a label."""
    ordered = sorted(hardness.values())
    n = len(ordered)
    easy_max, medium_max = ordered[max(0, n // 3 - 1)], ordered[max(0, 2 * n // 3 - 1)]
    easy, medium, hard = DIFFICULTY_LEVELS
    labels = {
        task_id: easy if value <= easy_max else medium if value <= medium_max else hard
        for task_id, value in hardness.items()
    }
    return labels, (easy_max, medium_max)


def build(record: Path, start: int, repeats: int, results: Path, manual_path: Path | None,
          manual_weight: float, seed: int, min_attempts: int) -> dict:
    if not 0 <= manual_weight <= 1:
        raise SystemExit("--manual-weight must be between 0 and 1")
    if manual_path is None and manual_weight > 0:
        raise SystemExit("--manual-weight above 0 needs a --manual annotation file")
    slots = sequence_tasks(record, start, repeats)
    counts, sources = measure(results)
    manual = read_manual(manual_path) if manual_path else {}

    tasks: dict[str, dict] = {}
    for slot in slots:
        task = slot["task"]
        task_id = str(task["task_id"])
        if task_id in tasks:
            continue
        solved, attempts = counts.get(task_id, [0, 0])
        if attempts < min_attempts and manual_weight < 1:
            raise SystemExit(
                f"{task_id}: only {attempts} graded attempt(s) under {results}, fewer than "
                f"--min-attempts {min_attempts}; run more trajectories over these sequences first."
            )
        if manual_path and task_id not in manual:
            raise SystemExit(f"{task_id}: not annotated in {manual_path}")
        entry = {
            "task_type": task_type_of(task),
            "display_id": display_task_id(task),
            "solved": solved,
            "attempts": attempts,
            "raw_solve_rate": round(solved / attempts, 4) if attempts else None,
            "solve_rate": round((solved + 1) / (attempts + 2), 4),
        }
        if manual_path:
            note = manual[task_id]
            entry["manual_score"] = note["score"]
            entry["manual_factors"] = note.get("factors") or {}
            entry["manual_rationale"] = note.get("rationale", "")
        tasks[task_id] = entry
    order = sorted(tasks)

    measured_hardness = {t: 1 - tasks[t]["solve_rate"] for t in order}
    manual_hardness = {t: float(tasks[t]["manual_score"]) for t in order} if manual_path else {}
    families = sorted({tasks[t]["task_type"] for t in order})
    combined: dict[str, float] = {}
    for family in families:
        members = [t for t in order if tasks[t]["task_type"] == family]
        measured_rank = percentile_ranks({t: measured_hardness[t] for t in members})
        manual_rank = percentile_ranks({t: manual_hardness[t] for t in members}) if manual_path else {}
        measured_labels = tertile_labels({t: measured_hardness[t] for t in members})[0]
        manual_labels = tertile_labels({t: manual_hardness[t] for t in members})[0] if manual_path else {}
        scale = sorted(measured_hardness[t] for t in members)
        for t in members:
            rank = manual_weight * manual_rank.get(t, 0.0) + (1 - manual_weight) * measured_rank[t]
            combined[t] = round(quantile(scale, rank), 4)
            entry = tasks[t]
            entry["measured_family_rank"] = round(measured_rank[t], 4)
            entry["measured_label"] = measured_labels[t]
            if manual_path:
                entry["manual_family_rank"] = round(manual_rank[t], 4)
                entry["manual_label"] = manual_labels[t]
            entry["family_rank"] = round(rank, 4)
            entry["combined_hardness"] = combined[t]
    true_labels, combined_cuts = tertile_labels(combined)
    for t in order:
        tasks[t]["true_label"] = true_labels[t]

    shuffled = [tasks[t]["true_label"] for t in order]
    random.Random(seed).shuffle(shuffled)
    for t, label in zip(order, shuffled):
        tasks[t]["random_label"] = label

    sequences: dict[str, list[dict]] = defaultdict(list)
    for slot in slots:
        rep_key = f"rep{slot['rep']:03d}"
        while len(sequences[rep_key]) < slot["episode"]:
            sequences[rep_key].append({"episode": len(sequences[rep_key]) + 1})
        entry = tasks[str(slot["task"]["task_id"])]
        sequences[rep_key][slot["episode"] - 1][slot["agent"]] = {
            "task_id": slot["task"]["task_id"],
            "true_label": entry["true_label"],
            "random_label": entry["random_label"],
        }

    def tally(key: str) -> dict:
        return {level: sum(1 for e in tasks.values() if e.get(key) == level) for level in DIFFICULTY_LEVELS}

    agreement = None
    if manual_path:
        levels = {level: i for i, level in enumerate(DIFFICULTY_LEVELS)}
        apart = [t for t in order
                 if abs(levels[tasks[t]["manual_label"]] - levels[tasks[t]["measured_label"]]) == 2]
        by_family = {}
        for family in sorted({e["task_type"] for e in tasks.values()}):
            members = [t for t in order if tasks[t]["task_type"] == family]
            by_family[family] = spearman({t: manual_hardness[t] for t in members},
                                         {t: measured_hardness[t] for t in members})
        agreement = {
            "note": "manual_label and measured_label are within-family tertiles",
            "spearman_manual_vs_measured": spearman(manual_hardness, measured_hardness),
            "spearman_by_task_type": by_family,
            "same_label": sum(1 for t in order if tasks[t]["manual_label"] == tasks[t]["measured_label"]),
            "two_levels_apart": [
                {"task_id": t, "manual": tasks[t]["manual_label"], "measured": tasks[t]["measured_label"],
                 "manual_score": tasks[t]["manual_score"], "solve_rate": tasks[t]["solve_rate"],
                 "final": tasks[t]["true_label"]}
                for t in apart
            ],
        }

    return {
        "description": (
            "Peer task-difficulty labels for the difficulty-label experiment. Each "
            "agent is shown the label of its PEER's task in the communication-phase "
            "brief, never its own. true_label combines a blind manual annotation with "
            "the solve rate measured in earlier runs; random_label is the same labels "
            "permuted across tasks. Written by analysis/task_difficulty_labels.py; "
            "regenerate rather than edit."
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "levels": list(DIFFICULTY_LEVELS),
        "method": {
            "manual": "1-5 score per task against task/difficulty_labels/annotation_rubric.md, "
                      "annotated from the task material without seeing any run output",
            "measured": "share of earlier episodes in which the task's own agent solved it "
                        "(expected_verdict == pass), pooled over all models; "
                        "solve_rate = (solved + 1) / (attempts + 2)",
            "combination": "within each task family, family_rank = manual_weight * manual_rank + "
                           "(1 - manual_weight) * measured_rank (hardness percentiles); "
                           "combined_hardness = the family's measured failure rate at quantile family_rank",
            "manual_weight": manual_weight,
            "binning": "tertiles of combined_hardness: top third hard, bottom third easy; ties share a label",
            "cuts": {"combined_hardness_easy_max": combined_cuts[0],
                     "combined_hardness_medium_max": combined_cuts[1]},
            "random_seed": seed,
            "random_matches_true": sum(1 for e in tasks.values() if e["true_label"] == e["random_label"]),
        },
        "source": {
            "task_sequence_record": _rel(record),
            "repeats": [f"rep{rep:03d}" for rep in range(start, start + repeats)],
            "manual_annotations": _rel(manual_path) if manual_path else None,
            "results_dir": _rel(results),
            "run_files": len(sources),
            "graded_attempts": sum(e["attempts"] for e in tasks.values()),
        },
        "agreement": agreement,
        "counts": {key: tally(key) for key in ("true_label", "random_label", "manual_label", "measured_label")},
        "tasks": {t: tasks[t] for t in order},
        "sequences": dict(sequences),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task-sequence-record", default="task/task_sequences_50x10")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--manual", default=DEFAULT_MANUAL_FILE,
                        help="manual annotation file; '' to label from the runs alone")
    parser.add_argument("--manual-weight", type=float, default=0.5,
                        help="weight of the manual rank in the combined hardness (0-1)")
    parser.add_argument("--results", default="results", help="Where the earlier run.json files are.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random-label permutation.")
    parser.add_argument("--min-attempts", type=int, default=5,
                        help="Refuse to use a measured rate from fewer attempts than this.")
    parser.add_argument("--out", default=DEFAULT_LABEL_FILE)
    args = parser.parse_args(argv)

    document = build(ROOT / args.task_sequence_record, args.start_index, args.repeats,
                     ROOT / args.results, ROOT / args.manual if args.manual else None,
                     args.manual_weight, args.seed, args.min_attempts)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    counts = document["counts"]["true_label"]
    print(f"wrote {args.out}: {len(document['tasks'])} tasks; "
          + ", ".join(f"{level} {counts[level]}" for level in DIFFICULTY_LEVELS))
    agreement = document["agreement"]
    if agreement:
        print(f"manual vs measured: spearman {agreement['spearman_manual_vs_measured']}, "
              f"same label {agreement['same_label']}/{len(document['tasks'])}, "
              f"two levels apart {len(agreement['two_levels_apart'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
