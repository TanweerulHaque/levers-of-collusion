"""Run the checked-out branch's compare_runs on named sets of run.json files.

usage: python harness.py <branch-tag> <out.json> <set> [<set> ...] [--cmp A:B ...]

Each set name is defined in SETS below as a list of globs relative to results/.
Sets ending in '+inc' keep incomplete trajectories.
"""
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path("/Users/tanweerulhaque/Desktop/NTU/NAIS_hackathon/joint-task-collusion")
RES = REPO / "results"
sys.path.insert(0, str(REPO / "analysis"))
import compare_runs  # noqa: E402  (from whichever branch is checked out)

LG = "openai-gpt-6-luna-floor__z-ai-glm-5.3-flash-floor"
GL = "z-ai-glm-5.3-flash-floor__openai-gpt-6-luna-floor"
DD = "deepseek-deepseek-v4.1-flash-floor"
DM = "deepseek-deepseek-v4.1-flash-floor__xiaomi-mimo-v2.6-flash-floor"
HM = "anthropic-claude-haiku-5.5-floor__xiaomi-mimo-v2.6-flash-floor_forecast-before-verdict"


def reps(launch_glob, which=None):
    files = sorted(glob.glob(str(RES / launch_glob / "rep*" / "run.json")),
                   key=lambda p: int(re.search(r"/rep(\d+)/", p).group(1)))
    if which is not None:
        files = [p for p in files if int(re.search(r"/rep(\d+)/", p).group(1)) in which]
    return files


SETS = {
    # baselines (exp-2)
    "LG_base25": lambda: reps(f"{LG}/*_reward_verdict-review_reflection_20261024_004155"),
    "LG_base5": lambda: reps(f"{LG}/*_reward_verdict-review_reflection_20261024_004155", range(1, 6)),
    "LG_base_oct20": lambda: reps(f"{LG}/*_20261020_004835"),
    "GL_base25": lambda: reps(f"{GL}/*accuracy_reward_verdict-review_reflection_20261024_150426"),
    "GL_base5": lambda: reps(f"{GL}/*accuracy_reward_verdict-review_reflection_20261024_150426", range(1, 6)),
    "DD_base": lambda: reps(f"{DD}/*_20261024_032115"),
    "DM_base": lambda: (reps(f"{DM}/*_20261011_012330", {1, 2, 4, 5}) + reps(f"{DM}/*_20261011_162251", {3})),
    # reward penalty (exp-2)
    "LG_pen25": lambda: reps(f"{LG}-reward-penalty-per-term/*"),
    "LG_pen5": lambda: reps(f"{LG}-reward-penalty-per-term/*", range(1, 6)),
    "GL_pen": lambda: (reps(f"{GL}/*per-term*_20261024_225034", range(1, 12))
                       + reps(f"{GL}/*per-term*_resume_20261025_003555")
                       + reps(f"{GL}/*per-term*_20261025_003629")),
    # forecast (exp-3)
    "LG_forecast": lambda: reps(f"{LG}_forecast-before-communication/*"),
    "HM_forecast": lambda: reps(f"{HM}/*"),
    # announcement (exp-5)
    "LG_countdown": lambda: reps(f"{LG}_episode_announcement_all/*"),
    "LG_countdown_pen": lambda: reps(f"{LG}_episode_announcement_all_with_reward_penalty_per_term/*"),
    # restore tool (exp-6)
    "LG_restore50": lambda: reps(f"{LG}_restore_channel_link_cost50/*"),
    "LG_restore110": lambda: reps(f"{LG}_restore_channel_link_cost110/*"),
    "GL_restore110": lambda: reps(f"{GL}_restore_channel_link_cost110/*"),
    "DM_restore50": lambda: reps(f"{DM}_restore_channel_link_cost50/*"),
    # message cost (exp-7)
    "LG_msgcost": lambda: reps(f"{LG}_message_cost_rate0_005/*"),
    # sequential verdict (exp-8)
    "LG_seq": lambda: reps(f"{LG}-verdict-order_sequential/*"),
    "GL_seq": lambda: reps(f"{GL}-verdict-order_sequential/*"),
    "DM_seq": lambda: (reps(f"{DM}-verdict-order_sequential/*_20261017_221559", {1, 2, 5})
                       + reps(f"{DM}-verdict-order_sequential/*_20261017_235313")),
    # difficulty label (exp-9)
    "LG_diff": lambda: reps(f"{LG}-difficulty-label_true/*"),
    "GL_diff": lambda: reps(f"{GL}-difficulty-label_true/*"),
    "DM_diff": lambda: reps(f"{DM}-difficulty-label_true/*"),
}


def files_for(name):
    return SETS[name.removesuffix("+inc")]()


def main():
    tag, out = sys.argv[1], sys.argv[2]
    names, cmps = [], []
    args = sys.argv[3:]
    while args:
        a = args.pop(0)
        if a == "--cmp":
            cmps.append(args.pop(0).split(":"))
        else:
            names.append(a)
    for a, b in cmps:
        names += [x for x in (a, b) if x not in names]

    current = {}
    compare_runs.find_run_files = lambda directory: [Path(p) for p in current["files"]]
    sides = {}
    for name in names:
        current["files"] = files_for(name)
        side = compare_runs.summarize_side(RES, name, "raw-only", name.endswith("+inc"))
        side["files"] = [str(Path(p).relative_to(RES)) for p in current["files"]]
        sides[name] = side
    comparisons = {}
    for a, b in cmps:
        rows = [compare_runs.compare_metric(m, sides[a]["metrics"][m], sides[b]["metrics"][m])
                for m in sides[a]["metrics"] if m in sides[b]["metrics"]]
        comparisons[f"{a}|{b}"] = rows
    for side in sides.values():
        for metric in side["metrics"].values():
            metric.pop("values", None)
    Path(out).write_text(json.dumps({"branch": tag, "sides": sides, "comparisons": comparisons}, indent=1, default=str))
    for name, s in sides.items():
        m = s["metrics"]
        g = lambda k: m[k]["value"]
        fmt = lambda v: "  n/a" if v is None else f"{100*v:5.1f}"
        print(f"{name:18s} N={s['trajectories']:2d}/{s['found']:2d} eps={s['episodes']:3d} EC={fmt(g('EC'))} TC={fmt(g('TC'))} "
              f"CC={fmt(g('CC'))} A_acc={fmt(g('alice_accept_rate'))} B_acc={fmt(g('bob_accept_rate'))} "
              f"vacc={fmt(g('verdict_accuracy'))} R={g('joint_reward_mean') if g('joint_reward_mean') is None else round(g('joint_reward_mean'),2)}")
    for key, rows in comparisons.items():
        print("--", key)
        for r in rows:
            if r["metric"] in ("EC", "TC", "CC", "onset_episode", "EC_final_episode", "alice_accept_rate", "bob_accept_rate",
                               "verdict_accuracy", "alice_verdict_accuracy", "bob_verdict_accuracy",
                               "alice_false_accept_rate", "bob_false_accept_rate", "alice_false_reject_rate",
                               "bob_false_reject_rate", "joint_reward_mean", "mutual_reject_rate", "task_fail_share") and r["diff"] is not None:
                d = f"{100*r['diff']:+6.1f}pp" if r["kind"] == "rate" else f"{r['diff']:+6.2f}"
                print(f"   {r['metric']:26s} {d}  p={r['p'] if r['p'] is None else round(r['p'], 3)}  ci={r['diff_ci']}")


if __name__ == "__main__":
    main()
