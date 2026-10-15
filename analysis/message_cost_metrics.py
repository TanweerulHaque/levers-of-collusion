#!/usr/bin/env python3
"""Message length and what it cost, from the episodes in ``run.json``.

Runs launched with ``--message-cost`` leave the channel unlimited and bill each
delivered message to its sender: ``max(0, ln((chars * rate) ** e))``, settled after the
verdicts out of the absolute verdict reward (``net = |gross| - bill``). Every such
episode records a ``message_cost`` block, and this script reads only that block and the
verdicts. Runs without it are skipped.

One *communication round* here is one message by one agent, so "per round" below means
per message. Counted per condition (the trajectories of one model pair and launch),
overall, per agent (Alice, Bob) and per trajectory:

* characters and cost per message, per episode, and per episode combined (Alice + Bob);
* the same split by verdict pair (AA, RR, AR, RA; Alice first), and within each pair by
  verdict accuracy (both verdicts right, only Alice's, only Bob's, neither);
* who paid more in each of those, how often, and by how much (Alice minus Bob);
* per communication round, and per episode index across trajectories;
* free messages, and whether the raw log was sent verbatim, so mutual ACCEPT can be
  told apart as evidenced (both logs sent whole) or unsupported;
* accuracy of an agent's verdict against what its peer spent on it;
* gross, counted and net reward.

Standard library only. Usage::

    python analysis/message_cost_metrics.py --runs 'results/*/*/rep*/run.json'
    python analysis/message_cost_metrics.py --runs '...' --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_layout import describe as describe_layout  # noqa: E402

AGENTS = ("alice", "bob")
PAIRS = ("AA", "RR", "AR", "RA", "other")
LETTER = {"accept": "A", "reject": "R"}
# Which of the two verdicts matched its peer's ground truth.
ACCURACY = ("both_right", "alice_only", "bob_only", "both_wrong", "unknown")
ACCURACY_LABEL = {
    "both_right": "both verdicts right",
    "alice_only": "only Alice's verdict right",
    "bob_only": "only Bob's verdict right",
    "both_wrong": "both verdicts wrong",
    "unknown": "accuracy unknown",
}
# What the peer spent on an agent's evidence, in reward units, as half-open intervals.
SPEND_BUCKETS = (
    ("free (0)", 0.0, 0.0),
    ("(0, 5]", 0.0, 5.0),
    ("(5, 10]", 5.0, 10.0),
    ("above 10", 10.0, math.inf),
)


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _sd(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _verdicts(episode: dict) -> dict:
    meta = episode.get("analysis_metadata") or {}
    verdicts = dict(meta.get("verdict_by_agent") or {})
    for name, agent in (episode.get("agents") or {}).items():
        verdicts.setdefault(name, agent.get("verdict"))
    return verdicts


def _correct(episode: dict) -> dict:
    meta = episode.get("analysis_metadata") or {}
    correct = dict(meta.get("verdict_correct_by_agent") or {})
    for name, agent in (episode.get("agents") or {}).items():
        correct.setdefault(name, agent.get("verdict_correct"))
    return correct


def pair_of(verdicts: dict) -> str:
    code = "".join(LETTER.get(verdicts.get(agent), "?") for agent in AGENTS)
    return code if code in PAIRS else "other"


def accuracy_of(correct: dict) -> str:
    flags = [correct.get(agent) for agent in AGENTS]
    if any(flag not in (True, False) for flag in flags):
        return "unknown"
    return {
        (True, True): "both_right",
        (True, False): "alice_only",
        (False, True): "bob_only",
        (False, False): "both_wrong",
    }[(flags[0], flags[1])]


# ------------------------------------------------------------------ records


def episode_record(episode: dict) -> dict | None:
    """One episode's messages, bills, verdicts and rewards, or None when not priced."""
    block = episode.get("message_cost") or {}
    if not block.get("enabled"):
        return None
    verdicts = _verdicts(episode)
    correct = _correct(episode)
    free = float(block.get("free_characters") or 0)
    per_agent = {}
    for agent in AGENTS:
        facts = (block.get("by_agent") or {}).get(agent) or {}
        messages = facts.get("messages") or []
        per_agent[agent] = {
            "chars": int(facts.get("chars") or 0),
            "cost": float(facts.get("cost") or 0.0),
            "messages": len(messages),
            "message_chars": [int(m.get("chars") or 0) for m in messages],
            "message_costs": [float(m.get("cost") or 0.0) for m in messages],
            "message_rounds": [m.get("round") for m in messages],
            "free_messages": sum(1 for m in messages if int(m.get("chars") or 0) <= free),
            "verdict": verdicts.get(agent),
            "verdict_correct": correct.get(agent),
            "gross_reward": facts.get("gross_reward"),
            "reward_counted": facts.get("reward_counted"),
            "net_reward": facts.get("net_reward"),
            "raw_log_sent_verbatim": bool(facts.get("raw_log_sent_verbatim")),
        }
    cost = {agent: per_agent[agent]["cost"] for agent in AGENTS}
    diff = round(cost["alice"] - cost["bob"], 4)
    return {
        "index": episode.get("episode_index"),
        "rate": block.get("rate"),
        "free_characters": block.get("free_characters"),
        "pair": pair_of(verdicts),
        "accuracy": accuracy_of(correct),
        "agents": per_agent,
        "combined_chars": sum(per_agent[a]["chars"] for a in AGENTS),
        "combined_cost": round(sum(cost.values()), 4),
        # Alice's bill minus Bob's: positive when Alice paid more.
        "paid_diff": diff,
        "paid_more": "tie" if diff == 0 else ("alice" if diff > 0 else "bob"),
    }


def run_records(run: dict) -> list[dict] | None:
    """The episode records of one run in episode order, or None when it priced nothing."""
    episodes = sorted(
        run.get("results") or [],
        key=lambda ep: (ep.get("episode_index") is None, ep.get("episode_index") or 0),
    )
    records = [episode_record(ep) for ep in episodes]
    if not any(records):
        return None
    return [record for record in records if record]


# ------------------------------------------------------------------ summaries


def _per_agent_means(records: list[dict], field: str) -> dict:
    """Mean of one per-agent episode field for Alice, Bob and the two combined."""
    values = {a: [r["agents"][a][field] for r in records] for a in AGENTS}
    combined = [sum(r["agents"][a][field] for a in AGENTS) for r in records]
    return {**{a: _mean(values[a]) for a in AGENTS}, "combined": _mean(combined)}


def _per_message_means(records: list[dict], field: str) -> dict:
    """Mean of one per-message field (a message is one agent's turn in one round)."""
    values = {a: [v for r in records for v in r["agents"][a][field]] for a in AGENTS}
    combined = values["alice"] + values["bob"]
    return {**{a: _mean(values[a]) for a in AGENTS}, "combined": _mean(combined)}


def block(records: list[dict]) -> dict:
    """Length, price, who paid more, accuracy and reward over a set of episodes."""
    n = len(records)
    paid_more = {who: sum(1 for r in records if r["paid_more"] == who) for who in ("alice", "bob", "tie")}
    alice_more = [r["paid_diff"] for r in records if r["paid_more"] == "alice"]
    bob_more = [-r["paid_diff"] for r in records if r["paid_more"] == "bob"]
    verdicts = {
        a: [r["agents"][a] for r in records if r["agents"][a]["verdict"] in LETTER] for a in AGENTS
    }
    right = {a: sum(1 for v in verdicts[a] if v["verdict_correct"] is True) for a in AGENTS}
    both_right = sum(1 for r in records if r["accuracy"] == "both_right")
    messages = {a: sum(r["agents"][a]["messages"] for r in records) for a in AGENTS}
    free = {a: sum(r["agents"][a]["free_messages"] for r in records) for a in AGENTS}
    return {
        "episodes": n,
        "mean_chars_per_message": _per_message_means(records, "message_chars"),
        "mean_cost_per_message": _per_message_means(records, "message_costs"),
        "mean_chars_per_episode": _per_agent_means(records, "chars"),
        "mean_cost_per_episode": _per_agent_means(records, "cost"),
        "total_chars": {**{a: sum(r["agents"][a]["chars"] for r in records) for a in AGENTS},
                        "combined": sum(r["combined_chars"] for r in records)},
        "total_cost": {**{a: round(sum(r["agents"][a]["cost"] for r in records), 4) for a in AGENTS},
                       "combined": round(sum(r["combined_cost"] for r in records), 4)},
        "paid_more": paid_more,
        # Alice minus Bob over every episode of the set, and over only those each paid more in.
        "mean_paid_diff": _mean([r["paid_diff"] for r in records]),
        "total_paid_diff": round(sum(r["paid_diff"] for r in records), 4),
        "mean_diff_when_alice_paid_more": _mean(alice_more),
        "mean_diff_when_bob_paid_more": _mean(bob_more),
        "free_message_share": {
            **{a: _rate(free[a], messages[a]) for a in AGENTS},
            "combined": _rate(sum(free.values()), sum(messages.values())),
        },
        "verdict_accuracy": {
            **{a: _rate(right[a], len(verdicts[a])) for a in AGENTS},
            "both_right": _rate(both_right, n),
        },
        "mean_gross_reward": {a: _mean([r["agents"][a]["gross_reward"] for r in records]) for a in AGENTS},
        "mean_net_reward": {a: _mean([r["agents"][a]["net_reward"] for r in records]) for a in AGENTS},
        "negative_net_share": {
            a: _rate(sum(1 for r in records if (r["agents"][a]["net_reward"] or 0) < 0), n)
            for a in AGENTS
        },
    }


def _pair_blocks(records: list[dict]) -> dict:
    """Each verdict pair's block, and inside it the same by verdict accuracy."""
    out = {}
    for pair in PAIRS:
        rows = [r for r in records if r["pair"] == pair]
        if not rows and pair == "other":
            continue
        entry = block(rows)
        entry["share"] = _rate(len(rows), len(records))
        entry["by_accuracy"] = {}
        for accuracy in ACCURACY:
            sub = [r for r in rows if r["accuracy"] == accuracy]
            if sub or accuracy != "unknown":
                entry["by_accuracy"][accuracy] = block(sub)
        out[pair] = entry
    return out


def _accuracy_blocks(records: list[dict]) -> dict:
    return {
        accuracy: block([r for r in records if r["accuracy"] == accuracy])
        for accuracy in ACCURACY
        if accuracy != "unknown" or any(r["accuracy"] == "unknown" for r in records)
    }


def _rounds(records: list[dict]) -> list[dict]:
    """Mean length and price of the message each agent sent in communication round 1, 2, ..."""
    by_round: dict[int, dict] = defaultdict(lambda: {a: {"chars": [], "cost": []} for a in AGENTS})
    for record in records:
        for agent in AGENTS:
            facts = record["agents"][agent]
            for rnd, chars, cost in zip(
                facts["message_rounds"], facts["message_chars"], facts["message_costs"]
            ):
                if isinstance(rnd, int):
                    by_round[rnd][agent]["chars"].append(chars)
                    by_round[rnd][agent]["cost"].append(cost)
    rows = []
    for rnd in sorted(by_round):
        slot = by_round[rnd]
        rows.append({
            "round": rnd + 1,
            "chars": {a: _mean(slot[a]["chars"]) for a in AGENTS},
            "cost": {a: _mean(slot[a]["cost"]) for a in AGENTS},
            "messages": {a: len(slot[a]["chars"]) for a in AGENTS},
        })
    return rows


def _by_episode_index(trajectories: list[tuple[str, list[dict]]]) -> list[dict]:
    """Mean bill and length at each episode index across trajectories: the spending trend."""
    by_index: dict[int, list[dict]] = defaultdict(list)
    for _, records in trajectories:
        for record in records:
            if isinstance(record["index"], int):
                by_index[record["index"]].append(record)
    return [
        {
            "index": index,
            "trajectories": len(rows),
            "cost": _per_agent_means(rows, "cost"),
            "chars": _per_agent_means(rows, "chars"),
            "mutual_accept": _rate(sum(1 for r in rows if r["pair"] == "AA"), len(rows)),
            "mean_net_reward": {a: _mean([r["agents"][a]["net_reward"] for r in rows]) for a in AGENTS},
        }
        for index, rows in sorted(by_index.items())
    ]


def _spend_vs_accuracy(records: list[dict]) -> list[dict]:
    """An agent's verdict accuracy and accept rate against what its peer spent on it.

    One row per bucket of the peer's episode bill; both agents' verdicts are pooled.
    """
    observations = []
    for record in records:
        for agent in AGENTS:
            peer = "bob" if agent == "alice" else "alice"
            facts = record["agents"][agent]
            if facts["verdict"] in LETTER:
                observations.append((record["agents"][peer]["cost"], facts))
    rows = []
    for label, low, high in SPEND_BUCKETS:
        if high == 0.0:
            chosen = [f for cost, f in observations if cost == 0.0]
        else:
            chosen = [f for cost, f in observations if low < cost <= high]
        rows.append({
            "bucket": label,
            "verdicts": len(chosen),
            "accept_rate": _rate(sum(1 for f in chosen if f["verdict"] == "accept"), len(chosen)),
            "accuracy": _rate(sum(1 for f in chosen if f["verdict_correct"] is True), len(chosen)),
        })
    return rows


def _evidence(records: list[dict]) -> dict:
    """Raw log sent verbatim, and mutual ACCEPT with and without both logs sent whole."""
    aa = [r for r in records if r["pair"] == "AA"]
    backed = [r for r in aa if all(r["agents"][a]["raw_log_sent_verbatim"] for a in AGENTS)]
    verbatim = [r for r in records if any(r["agents"][a]["raw_log_sent_verbatim"] for a in AGENTS)]
    return {
        "episodes_with_a_verbatim_log": len(verbatim),
        "verbatim_log_by_agent": {
            a: sum(1 for r in records if r["agents"][a]["raw_log_sent_verbatim"]) for a in AGENTS
        },
        "mutual_accept": len(aa),
        "mutual_accept_evidenced": len(backed),
        "mutual_accept_unsupported": len(aa) - len(backed),
        "mean_combined_cost_evidenced": _mean([r["combined_cost"] for r in backed]),
        "mean_combined_cost_unsupported": _mean([r["combined_cost"] for r in aa if r not in backed]),
    }


def _reward(records: list[dict]) -> dict:
    out = {}
    for agent in AGENTS:
        gross = [r["agents"][agent]["gross_reward"] or 0 for r in records]
        counted = [r["agents"][agent]["reward_counted"] or 0 for r in records]
        net = [r["agents"][agent]["net_reward"] or 0 for r in records]
        paid = [r["agents"][agent]["cost"] for r in records]
        out[agent] = {
            "gross_total": sum(gross),
            "counted_total": sum(counted),
            "cost_total": round(sum(paid), 4),
            "net_total": round(sum(net), 4),
            "cost_share_of_counted": _rate(sum(paid), sum(counted)) if sum(counted) else None,
        }
    return out


def trajectory_summary(name: str, records: list[dict]) -> dict:
    """The block of one trajectory, its verdict pairs, and per-episode bills for a sparkline."""
    return {
        "rep": name,
        "episodes": len(records),
        "overall": block(records),
        "by_pair": {pair: entry for pair, entry in _pair_blocks(records).items()},
        "pairs": [r["pair"] for r in records],
        "cost_by_episode": {a: [r["agents"][a]["cost"] for r in records] for a in AGENTS},
        "chars_by_episode": {a: [r["agents"][a]["chars"] for r in records] for a in AGENTS},
        "paid_more_by_episode": [r["paid_more"] for r in records],
        "evidence": _evidence(records),
        "reward": _reward(records),
    }


def _spread(values: list[float | None]) -> dict:
    vals = [v for v in values if v is not None]
    return {
        "mean": _mean(vals),
        "sd": _sd(vals),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def summarize(trajectories: list[tuple[str, list[dict]]]) -> dict:
    """Overall, per-agent, per-trajectory and trajectory-level metrics for ``(name, records)``."""
    trajectories = [(name, records) for name, records in trajectories if records]
    episodes = [record for _, records in trajectories for record in records]
    rows = [trajectory_summary(name, records) for name, records in trajectories]
    rates = sorted({r["rate"] for r in episodes if r["rate"] is not None})

    traj_level = {
        "trajectories": len(rows),
        # Spread across trajectories of each trajectory's own mean, so one long run cannot dominate.
        "chars_per_message": {
            a: _spread([row["overall"]["mean_chars_per_message"][a] for row in rows]) for a in (*AGENTS, "combined")
        },
        "cost_per_message": {
            a: _spread([row["overall"]["mean_cost_per_message"][a] for row in rows]) for a in (*AGENTS, "combined")
        },
        "cost_per_episode": {
            a: _spread([row["overall"]["mean_cost_per_episode"][a] for row in rows]) for a in (*AGENTS, "combined")
        },
        "alice_paid_more_overall": sum(1 for row in rows if row["overall"]["total_paid_diff"] > 0),
        "bob_paid_more_overall": sum(1 for row in rows if row["overall"]["total_paid_diff"] < 0),
        "net_reward_total": {
            a: _spread([row["reward"][a]["net_total"] for row in rows]) for a in AGENTS
        },
    }
    return {
        "rate": rates,
        "free_characters": sorted({r["free_characters"] for r in episodes if r["free_characters"]}),
        "trajectories": len(rows),
        "episodes": len(episodes),
        "overall": block(episodes),
        "by_pair": _pair_blocks(episodes),
        "by_accuracy": _accuracy_blocks(episodes),
        "by_round": _rounds(episodes),
        "by_episode_index": _by_episode_index(trajectories),
        "spend_vs_accuracy": _spend_vs_accuracy(episodes),
        "evidence": _evidence(episodes),
        "reward": _reward(episodes),
        "trajectory_level": traj_level,
        "per_trajectory": rows,
    }


# ------------------------------------------------------------------ loading


def load_groups(paths: list[Path]) -> tuple[list[dict], int]:
    """Group priced runs by model pair and launch, then summarize each group."""
    grouped: dict[tuple, list[tuple[str, list[dict]]]] = defaultdict(list)
    skipped = 0
    for path in paths:
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records = run_records(run)
        if records is None:
            skipped += 1
            continue
        layout = describe_layout(path)
        name = f"rep{layout['rep']}" if layout["rep"] is not None else path.parent.name
        grouped[(layout["pair"], layout["condition"])].append((name, records))
    groups = []
    for (pair, condition), members in sorted(grouped.items()):
        members.sort(key=lambda item: (len(item[0]), item[0]))
        groups.append({"pair": pair, "condition": condition, **summarize(members)})
    return groups, skipped


# ------------------------------------------------------------------ report


def _n(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _who(entry: dict) -> str:
    more = entry["paid_more"]
    if more["alice"] == more["bob"]:
        return f"tie ({more['alice']}-{more['bob']}, ties {more['tie']})"
    leader = "alice" if more["alice"] > more["bob"] else "bob"
    return f"{leader} {more[leader]}-{more['bob' if leader == 'alice' else 'alice']} (ties {more['tie']})"


def _line(label: str, entry: dict, indent: str = "    ") -> str:
    if not entry["episodes"]:
        return f"{indent}{label:<22} no episodes"
    chars, cost = entry["mean_chars_per_episode"], entry["mean_cost_per_episode"]
    diff = entry["mean_diff_when_alice_paid_more"], entry["mean_diff_when_bob_paid_more"]
    return (
        f"{indent}{label:<22} n={entry['episodes']:<3} chars a/b/both "
        f"{_n(chars['alice'], 0)}/{_n(chars['bob'], 0)}/{_n(chars['combined'], 0)}  "
        f"cost a/b/both {_n(cost['alice'], 2)}/{_n(cost['bob'], 2)}/{_n(cost['combined'], 2)}  "
        f"paid more: {_who(entry)}; by {_n(diff[0], 2)} (alice) / {_n(diff[1], 2)} (bob)"
    )


def format_report(groups: list[dict], skipped: int = 0) -> str:
    lines = ["=" * 78, "Message cost  (max(0, ln((chars * rate) ** e)) per delivered message)", "=" * 78]
    if not groups:
        lines.append("no run.json priced its messages")
    for g in groups:
        o = g["overall"]
        lines += ["", g["pair"], f"  {g['condition']}"]
        lines.append(
            f"  rate {'/'.join(f'{r:g}' for r in g['rate'])}, free up to "
            f"{'/'.join(f'{c:g}' for c in g['free_characters'])} characters   "
            f"N={g['trajectories']} trajectories, {g['episodes']} episodes"
        )
        cpm, kpm = o["mean_chars_per_message"], o["mean_cost_per_message"]
        lines.append(
            f"  per message (one agent, one round): chars alice {_n(cpm['alice'], 0)}, bob {_n(cpm['bob'], 0)}, "
            f"both {_n(cpm['combined'], 0)};  cost alice {_n(kpm['alice'], 2)}, bob {_n(kpm['bob'], 2)}, "
            f"both {_n(kpm['combined'], 2)};  free messages {_pct(o['free_message_share']['combined'])}"
        )
        lines.append("  per episode (all rounds):")
        lines.append(_line("overall", o))
        lines.append("  by verdict pair (Alice then Bob):")
        for pair, entry in g["by_pair"].items():
            lines.append(_line(f"{pair} ({_pct(entry['share'])})", entry))
        lines.append("  by verdict pair and verdict accuracy:")
        for pair, entry in g["by_pair"].items():
            if not entry["episodes"]:
                continue
            for accuracy, sub in entry["by_accuracy"].items():
                if sub["episodes"]:
                    lines.append(_line(f"{pair} / {ACCURACY_LABEL[accuracy]}", sub))
        lines.append("  by verdict accuracy only:")
        for accuracy, sub in g["by_accuracy"].items():
            lines.append(_line(ACCURACY_LABEL[accuracy], sub))
        lines.append("  by communication round (mean chars / cost, alice | bob):")
        for row in g["by_round"]:
            lines.append(
                f"    round {row['round']}: alice {_n(row['chars']['alice'], 0)} / {_n(row['cost']['alice'], 2)}"
                f"  |  bob {_n(row['chars']['bob'], 0)} / {_n(row['cost']['bob'], 2)}"
            )
        ev = g["evidence"]
        lines.append(
            f"  raw log sent verbatim: alice {ev['verbatim_log_by_agent']['alice']}, "
            f"bob {ev['verbatim_log_by_agent']['bob']} episodes;  mutual ACCEPT "
            f"{ev['mutual_accept']}: evidenced (both logs whole) {ev['mutual_accept_evidenced']}, "
            f"unsupported {ev['mutual_accept_unsupported']}"
        )
        lines.append("  peer spend vs the agent's verdict (bucket of the peer's bill):")
        for row in g["spend_vs_accuracy"]:
            lines.append(
                f"    {row['bucket']:<10} verdicts {row['verdicts']:<4} accept {_pct(row['accept_rate']):<7} "
                f"accuracy {_pct(row['accuracy'])}"
            )
        lines.append("  reward (gross -> counted |gross|, minus bill = net):")
        for agent in AGENTS:
            r = g["reward"][agent]
            lines.append(
                f"    {agent:<5} gross {_n(r['gross_total'], 1)}, counted {_n(r['counted_total'], 1)}, "
                f"bill {_n(r['cost_total'], 2)}, net {_n(r['net_total'], 2)}; bill is "
                f"{_pct(r['cost_share_of_counted'])} of counted"
            )
        t = g["trajectory_level"]
        lines.append("  trajectory level (mean of each trajectory's own mean, +- sd across trajectories):")
        for label, key in (("chars/message", "chars_per_message"), ("cost/message", "cost_per_message"),
                           ("cost/episode", "cost_per_episode")):
            parts = []
            for agent in (*AGENTS, "combined"):
                s = t[key][agent]
                parts.append(f"{agent} {_n(s['mean'], 2)}+-{_n(s['sd'], 2)}")
            lines.append(f"    {label:<14} " + "  ".join(parts))
        lines.append(
            f"    paid more over the whole trajectory: alice {t['alice_paid_more_overall']}, "
            f"bob {t['bob_paid_more_overall']} of {t['trajectories']}"
        )
        lines.append("  trajectories (verdicts Alice then Bob per episode):")
        for row in g["per_trajectory"]:
            ov = row["overall"]
            aa = row["by_pair"].get("AA") or {"episodes": 0}
            aa_text = (
                f"AA n={aa['episodes']} paid more {_who(aa)}, combined cost "
                f"{_n(aa['mean_cost_per_episode']['combined'], 2)}"
                if aa["episodes"] else "AA none"
            )
            lines.append(
                f"    {row['rep']:<6} {' '.join(row['pairs'])}  chars/msg a{_n(ov['mean_chars_per_message']['alice'], 0)}"
                f"/b{_n(ov['mean_chars_per_message']['bob'], 0)}  cost/ep a{_n(ov['mean_cost_per_episode']['alice'], 2)}"
                f"/b{_n(ov['mean_cost_per_episode']['bob'], 2)}  paid more {_who(ov)}  {aa_text}"
            )
    if skipped:
        lines += ["", f"({skipped} run(s) without priced messages skipped)"]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", required=True,
                        help="glob(s) matching run.json files, relative to the repo root")
    parser.add_argument("--json", default="", help="also write the full result to this JSON file")
    args = parser.parse_args(argv)

    paths = sorted({Path(p) for pattern in args.runs for p in glob.glob(str(ROOT / pattern))})
    if not paths:
        print("no run.json matched", file=sys.stderr)
        return 2
    groups, skipped = load_groups(paths)
    print(format_report(groups, skipped))
    if args.json:
        Path(args.json).write_text(json.dumps({"groups": groups}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
