"""Collect every number the report and slides use into numbers.json and numbers.tex.

Inputs are the JSON files in data/, written by analysis/harness.py (one per branch),
analysis/evidence.py and analysis/extras.py. Comparisons the branches did not run
(task-matched baselines, unsupported mutual ACCEPT) are recomputed here with the same
trajectory bootstrap and permutation test as analysis/compare_runs.py.

    python make_numbers.py
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP = {tag: json.loads((DATA / f"{tag}.json").read_text()) for tag in
       ("exp2", "exp3", "exp5", "exp6", "exp7", "exp8", "exp9")}
EVID = json.loads((DATA / "evidence_out.json").read_text())
EXTRA = json.loads((DATA / "extras.json").read_text())

MODELS = {"LG": ("luna", "glm"), "GL": ("glm", "luna"), "DD": ("deepseek", "deepseek"),
          "DM": ("deepseek", "mimo"), "HM": ("haiku", "mimo")}
PAIR_LABEL = {"LG": "Luna $\\to$ GLM", "GL": "GLM $\\to$ Luna", "DD": "DeepSeek $\\times$ DeepSeek",
              "DM": "DeepSeek $\\times$ MiMo", "HM": "Haiku $\\times$ MiMo"}

# key, lever, pair, branch json, side, (baseline json, baseline side), branch name
CONDITIONS = [
    ("LG_base", "Baseline", "LG", "exp2", "LG_base25", None, "exp-2"),
    ("GL_base", "Baseline", "GL", "exp2", "GL_base25", None, "exp-2"),
    ("DD_base", "Baseline", "DD", "exp2", "DD_base", None, "exp-2"),
    ("DM_base", "Baseline", "DM", "exp2", "DM_base", None, "exp-2"),
    ("HM_fore", "Forecast before verdict", "HM", "exp2", "HM_forecast+inc", None, "exp-3"),
    ("LG_pen", "Per-term penalty", "LG", "exp2", "LG_pen25", ("exp2", "LG_base25"), "exp-2"),
    ("GL_pen", "Per-term penalty", "GL", "exp2", "GL_pen", ("exp2", "GL_base25"), "exp-2"),
    ("LG_count", "Episode countdown", "LG", "exp5", "LG_countdown", ("exp5", "LG_base25"), "exp-5"),
    ("LG_countpen", "Countdown + penalty", "LG", "exp5", "LG_countdown_pen", ("exp5", "LG_pen25"), "exp-5"),
    ("LG_fore", "Private forecast", "LG", "exp3", "LG_forecast", ("exp3", "LG_base25"), "exp-3"),
    ("LG_seq", "Sequential verdict", "LG", "exp8", "LG_seq", ("exp8", "LG_base25"), "exp-8"),
    ("GL_seq", "Sequential verdict", "GL", "exp8", "GL_seq", ("exp8", "GL_base25"), "exp-8"),
    ("DM_seq", "Sequential verdict", "DM", "exp8", "DM_seq", ("exp8", "DM_base"), "exp-8"),
    ("LG_diff", "Difficulty label", "LG", "exp9", "LG_diff", ("exp9", "LG_base25"), "exp-9"),
    ("GL_diff", "Difficulty label", "GL", "exp9", "GL_diff", ("exp9", "GL_base25"), "exp-9"),
    ("DM_diff", "Difficulty label", "DM", "exp9", "DM_diff+inc", ("exp9", "DM_base+inc"), "exp-9"),
    ("LG_rest50", "Paid restore, 50\\%", "LG", "exp6", "LG_restore50", ("exp6", "LG_base25"), "exp-6"),
    ("LG_rest110", "Paid restore, 110\\%", "LG", "exp6", "LG_restore110", ("exp6", "LG_base25"), "exp-6"),
    ("GL_rest110", "Paid restore, 110\\%", "GL", "exp6", "GL_restore110", ("exp6", "GL_base25"), "exp-6"),
    ("DM_rest50", "Paid restore, 50\\%", "DM", "exp6", "DM_restore50+inc", ("exp6", "DM_base+inc"), "exp-6"),
    ("LG_price", "Message price", "LG", "exp7", "LG_msgcost+inc", ("exp7", "LG_base25"), "exp-7"),
]
EVIDENCE_KEYS = {"LG_rest50": "LG_restore50", "LG_rest110": "LG_restore110", "GL_rest110": "GL_restore110",
                 "DM_rest50": "DM_restore50", "LG_price": "LG_msgcost"}
METRICS = ("EC", "TC", "CC", "onset_episode", "EC_final_episode", "alice_accept_rate", "bob_accept_rate",
           "verdict_accuracy", "alice_verdict_accuracy", "bob_verdict_accuracy", "alice_false_accept_rate",
           "bob_false_accept_rate", "alice_false_reject_rate", "bob_false_reject_rate", "joint_reward_mean",
           "mutual_reject_rate", "split_verdict_rate", "task_fail_share", "message_chars_mean")


# ---------------------------------------------------------------- statistics (as compare_runs.py)
def ratio(pairs):
    den = sum(d for _, d in pairs)
    return sum(n for n, _ in pairs) / den if den else None


def _draw_ci(draws):
    draws = sorted(d for d in draws if d is not None)
    if len(draws) < 20:
        return None
    return [draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]]


def diff_ci(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    rng = random.Random(2)
    draws = []
    for _ in range(2000):
        ra = ratio([a[rng.randrange(len(a))] for _ in a])
        rb = ratio([b[rng.randrange(len(b))] for _ in b])
        draws.append(None if ra is None or rb is None else rb - ra)
    return _draw_ci(draws)


def perm_p(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    ra, rb = ratio(a), ratio(b)
    observed = abs(rb - ra)
    pool = a + b
    rng = random.Random(3)
    hits = 0
    for _ in range(4000):
        rng.shuffle(pool)
        pa, pb = ratio(pool[:len(a)]), ratio(pool[len(a):])
        if pa is not None and pb is not None and abs(pb - pa) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / 4001


def compare(a, b):
    a, b = [tuple(x) for x in a], [tuple(x) for x in b]
    va, vb = ratio(a), ratio(b)
    return {"a": va, "b": vb, "diff": None if va is None or vb is None else vb - va,
            "ci": diff_ci(a, b), "p": perm_p(a, b)}


# ---------------------------------------------------------------- collect
def side(tag, name):
    return EXP[tag]["sides"][name]


def metric(s, m):
    x = s["metrics"].get(m) or {}
    return {"value": x.get("value"), "ci": x.get("ci"), "num": x.get("numerator"), "den": x.get("denominator"),
            "per_trajectory": x.get("per_trajectory") or []}


def model_view(pair, s):
    """Accept rate and accuracy per model, so a seat swap reads as the same model."""
    a, b = MODELS[pair]
    out = {}
    for seat, model in (("alice", a), ("bob", b)):
        for kind in ("accept_rate", "verdict_accuracy", "false_accept_rate", "false_reject_rate"):
            out.setdefault(model, {})[kind] = metric(s, f"{seat}_{kind}")["value"]
        out[model]["seat"] = seat
    return out


numbers = {"conditions": {}, "extras": EXTRA}
for key, lever, pair, tag, name, base, branch in CONDITIONS:
    s = side(tag, name)
    entry = {"lever": lever, "pair": pair, "branch": branch, "side": name, "trajectories": s["trajectories"],
             "found": s["found"], "episodes": s["episodes"], "partial": name.endswith("+inc"),
             "metrics": {m: metric(s, m) for m in METRICS}, "models": model_view(pair, s),
             "verdict_pairs": {p["pair"]: p["share"] for p in s.get("verdict_pairs", [])},
             "curves": {k: v["values"] for k, v in s["curves"].items()}}
    if base:
        b = side(*base)
        entry["baseline"] = base[1]
        entry["delta"] = {}
        for m in METRICS:
            entry["delta"][m] = compare(metric(b, m)["per_trajectory"], metric(s, m)["per_trajectory"])
        # task-matched: the first five baseline trajectories use the same task sequences (rep001-rep005)
        if s["trajectories"] <= 5 and base[1].endswith("25"):
            entry["delta_matched"] = {m: compare(metric(b, m)["per_trajectory"][:5], metric(s, m)["per_trajectory"])
                                      for m in ("EC", "alice_accept_rate", "bob_accept_rate", "verdict_accuracy")}
    if key in EVIDENCE_KEYS:
        ev = EVID[EVIDENCE_KEYS[key]]
        entry["evidence"] = ev
        base_ec = metric(side(*base), "EC")["per_trajectory"]
        entry["delta"]["unsupported_EC"] = compare(base_ec, ev["per_trajectory_lenient"])
    numbers["conditions"][key] = entry

# pooled accept rate per model over the baselines, both seats
pooled = {}
for key in ("LG_base", "GL_base", "DD_base", "DM_base", "HM_fore"):
    s = side("exp2", numbers["conditions"][key]["side"])
    a, b = MODELS[numbers["conditions"][key]["pair"]]
    for seat, model in (("alice", a), ("bob", b)):
        m = metric(s, f"{seat}_accept_rate")
        acc = metric(s, f"{seat}_verdict_accuracy")
        row = pooled.setdefault(model, {"num": 0, "den": 0, "acc_num": 0, "acc_den": 0, "by_seat": {}})
        row["num"] += m["num"]
        row["den"] += m["den"]
        row["acc_num"] += acc["num"]
        row["acc_den"] += acc["den"]
        row["by_seat"][f"{key}:{seat}"] = m["value"]
for row in pooled.values():
    row["rate"] = row["num"] / row["den"]
    row["accuracy"] = row["acc_num"] / row["acc_den"]
numbers["model_accept"] = pooled

# does acceptance depend on whether the work is correct? accept|fail and accept|pass, pooled over seats
discrim = {}
for key in ("LG_base", "GL_base", "DD_base", "DM_base"):
    s = side("exp2", numbers["conditions"][key]["side"])
    a, b = MODELS[numbers["conditions"][key]["pair"]]
    for seat, model in (("alice", a), ("bob", b)):
        fa, fr = metric(s, f"{seat}_false_accept_rate"), metric(s, f"{seat}_false_reject_rate")
        row = discrim.setdefault(model, {"acc_fail": 0, "n_fail": 0, "acc_pass": 0, "n_pass": 0})
        row["acc_fail"] += fa["num"]
        row["n_fail"] += fa["den"]
        row["acc_pass"] += fr["den"] - fr["num"]
        row["n_pass"] += fr["den"]
for row in discrim.values():
    row["accept_given_fail"] = row["acc_fail"] / row["n_fail"]
    row["accept_given_pass"] = row["acc_pass"] / row["n_pass"]
numbers["discrimination"] = discrim
# episode-1 verdicts, before any feedback, over every baseline
first = [0, 0]
for key in ("LG_base", "GL_base", "DD_base", "DM_base", "HM_fore"):
    c, n = numbers["conditions"][key]["curves"], numbers["conditions"][key]["trajectories"]
    first[0] += round(c["alice_accept"][0] * n) + round(c["bob_accept"][0] * n)
    first[1] += 2 * n
numbers["episode1_accepts"] = first

# replicate batch and the 5-trajectory pilot of the penalty
rows = {r["metric"]: r for r in EXP["exp2"]["comparisons"]["LG_base5|LG_base_oct20"]}
numbers["replicate"] = {m: {"a": rows[m]["a"], "b": rows[m]["b"], "diff": rows[m]["diff"], "p": rows[m]["p"]}
                        for m in ("EC", "bob_accept_rate", "alice_accept_rate", "verdict_accuracy", "CC")}
rows = {r["metric"]: r for r in EXP["exp2"]["comparisons"]["LG_base5|LG_pen5"]}
numbers["pilot_penalty"] = {m: {"a": rows[m]["a"], "b": rows[m]["b"], "diff": rows[m]["diff"], "p": rows[m]["p"]}
                            for m in ("EC", "EC_final_episode", "joint_reward_mean")}

# sequential-verdict follow metrics (collusion_metrics.py on tanweer/exp-8)
seq = json.loads((DATA / "seq_collusion.json").read_text())
follow = {}
for g in seq["groups"]:
    inf = g.get("influence") or {}
    label = g["pair"] + ("/" + g["condition"][-15:] if "deepseek" in g["pair"] else "")
    mets = inf.get("metrics") or {}
    follow[label] = {k: {"rate": mets[k]["rate"], "k": mets[k]["k"], "n": mets[k]["n"]} for k in
                     ("follower_matches_leader", "follower_accept_given_leader_accept",
                      "follower_accept_given_leader_reject", "follower_cites_leader_rate") if k in mets}
    gap = (inf.get("gaps") or {}).get("reciprocity_gap") or {}
    follow[label]["reciprocity_gap"] = {"value": gap.get("value"), "ci": gap.get("ci")}
numbers["sequential_follow"] = follow

# restore-tool usage (restore_channel_metrics.py on tanweer/exp-6)
restore = {}
for path in DATA.glob("restore_*.json"):
    r = json.loads(path.read_text())
    for g in r.get("groups", []):
        restore[path.stem[len("restore_"):]] = g.get("restore") or g
numbers["restore"] = restore

(DATA / "numbers.json").write_text(json.dumps(numbers, indent=1))


# ---------------------------------------------------------------- LaTeX macros
def pct(v, signed=False):
    if v is None:
        return "n/a"
    x = 100 * v
    s = f"{x:+.1f}" if signed else f"{x:.1f}"
    if abs(round(x, 1) - round(x)) < 1e-9:
        s = f"{round(x):+d}" if signed else f"{round(x):d}"
    return s.replace("-", "$-$") + "\\%"


def pp(v):
    return "n/a" if v is None else (f"{100 * v:+.1f}".replace("-", "$-$") + "\\,pp")


def pval(p):
    if p is None:
        return "n/a"
    if p < 0.001:
        return "$p<0.001$"
    return f"$p={p:.3f}$" if p < 0.01 else f"$p={p:.2f}$"


def ci(c):
    if not c:
        return "n/a"
    return f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]".replace("-", "$-$")


def num(v, d=1):
    return "n/a" if v is None else f"{v:.{d}f}".replace("-", "$-$")


macros = {}
for key, e in numbers["conditions"].items():
    m = e["metrics"]
    macros[f"{key}.N"] = str(e["trajectories"])
    macros[f"{key}.eps"] = str(e["episodes"])
    for k in ("EC", "TC", "CC", "verdict_accuracy", "EC_final_episode", "mutual_reject_rate", "task_fail_share"):
        macros[f"{key}.{k}"] = pct(m[k]["value"])
    macros[f"{key}.onset"] = num(m["onset_episode"]["value"])
    macros[f"{key}.reward"] = num(m["joint_reward_mean"]["value"])
    macros[f"{key}.chars"] = num(m["message_chars_mean"]["value"], 0)
    for model, row in e["models"].items():
        tag = model if MODELS[e["pair"]][0] != MODELS[e["pair"]][1] else row["seat"]
        for kind, value in row.items():
            if kind != "seat":
                macros[f"{key}.{tag}.{kind}"] = pct(value)
    for kind in ("accept_rate", "verdict_accuracy", "false_accept_rate", "false_reject_rate"):
        for seat in ("alice", "bob"):
            macros[f"{key}.{seat}.{kind}"] = pct(m[f"{seat}_{kind}"]["value"])
    for d_key, d in (e.get("delta") or {}).items():
        macros[f"{key}.d.{d_key}"] = pp(d["diff"]) if d_key not in ("onset_episode", "joint_reward_mean") else num(d["diff"], 2)
        macros[f"{key}.d.{d_key}.p"] = pval(d["p"])
        macros[f"{key}.d.{d_key}.ci"] = ci(d["ci"]) if d_key not in ("onset_episode", "joint_reward_mean") else "n/a"
    for d_key, d in (e.get("delta_matched") or {}).items():
        macros[f"{key}.dm.{d_key}"] = pp(d["diff"])
        macros[f"{key}.dm.{d_key}.p"] = pval(d["p"])
    if "evidence" in e:
        ev = e["evidence"]
        n = ev["episodes"]
        macros[f"{key}.unsup"] = pct(ev["unsupported_mutual_both"]["lenient"] / n)
        macros[f"{key}.unsupStrict"] = pct(ev["unsupported_mutual_both"]["strict"] / n)
        macros[f"{key}.coverage"] = pct(ev["mean_log_coverage"])
        for seat in ("alice", "bob"):
            macros[f"{key}.{seat}.unsupAcc"] = f"{ev['unsupported_accepts']['lenient'][seat]}/{ev['accepts'][seat]}"
for model, row in pooled.items():
    macros[f"model.{model}.accept"] = pct(row["rate"])
    macros[f"model.{model}.n"] = str(row["den"])
    macros[f"model.{model}.accuracy"] = pct(row["accuracy"])
for model, row in discrim.items():
    macros[f"disc.{model}.fail"] = pct(row["accept_given_fail"])
    macros[f"disc.{model}.pass"] = pct(row["accept_given_pass"])
    macros[f"disc.{model}.nfail"] = str(int(row["n_fail"]))
macros["ep1.accepts"], macros["ep1.verdicts"] = str(first[0]), str(first[1])
cv = {k: numbers["conditions"][k]["curves"] for k in ("LG_base", "GL_base", "DD_base")}
pooled_curve = {
    "deepseek": [(a + b) / 2 for a, b in zip(cv["DD_base"]["alice_accept"], cv["DD_base"]["bob_accept"])],
    "glm": [(a + b) / 2 for a, b in zip(cv["LG_base"]["bob_accept"], cv["GL_base"]["alice_accept"])],
    "luna": [(a + b) / 2 for a, b in zip(cv["LG_base"]["alice_accept"], cv["GL_base"]["bob_accept"])],
}
for model, ys in pooled_curve.items():
    for t, y in enumerate(ys, 1):
        macros[f"curve.{model}.e{t}"] = pct(y)
for side_name, band in EXTRA["noise_band"].items():
    for m, b in band.items():
        macros[f"band.{side_name}.{m}"] = f"{pct(b['p2.5'])}--{pct(b['p97.5'])}"
for m, r in numbers["replicate"].items():
    macros[f"rep.{m}.a"], macros[f"rep.{m}.b"] = pct(r["a"]), pct(r["b"])
    macros[f"rep.{m}.d"], macros[f"rep.{m}.p"] = pp(r["diff"]), pval(r["p"])
for m, r in numbers["pilot_penalty"].items():
    if m == "joint_reward_mean":
        macros[f"pilot.{m}.a"], macros[f"pilot.{m}.b"] = num(r["a"]), num(r["b"])
    else:
        macros[f"pilot.{m}.a"], macros[f"pilot.{m}.b"] = pct(r["a"]), pct(r["b"])
    macros[f"pilot.{m}.p"] = pval(r["p"])
for name, d in EXTRA["deals"].items():
    macros[f"deal.{name}.eps"] = pct(d["announce_episodes"] / d["episodes"])
    macros[f"deal.{name}.mutual"] = pct(d["announce_and_mutual"] / d["mutual"]) if d["mutual"] else "n/a"
for name, f in EXTRA["forecast"].items():
    for seat, r in f.items():
        macros[f"fc.{name}.{seat}.forecast"] = pct(r["mean_forecast_accept"])
        macros[f"fc.{name}.{seat}.actual"] = pct(r["actual_accept"])
        macros[f"fc.{name}.{seat}.brier"] = num(r["brier"], 2)
t = EXTRA["totals"]
macros["tot.traj"], macros["tot.complete"] = str(t["trajectories"]), str(t["complete"])
macros["tot.eps"] = f"{t['episodes']:,}".replace(",", "{,}")
macros["tot.cost"] = f"{t['cost_usd']:.0f}"
for label, f in follow.items():
    tag = {"openai-gpt-6-luna-floor__z-ai-glm-5.3-flash-floor-verdict-order_sequential": "LG_seq",
           "z-ai-glm-5.3-flash-floor__openai-gpt-6-luna-floor-verdict-order_sequential": "GL_seq",
           "openai-gpt-6-luna-floor__z-ai-glm-5.3-flash-floor": "LG_base",
           "z-ai-glm-5.3-flash-floor__openai-gpt-6-luna-floor": "GL_base"}.get(label)
    if not tag:
        continue
    for k, v in f.items():
        if k == "reciprocity_gap":
            macros[f"follow.{tag}.gap"] = pp(v["value"])
        else:
            macros[f"follow.{tag}.{k}"] = pct(v["rate"])
            macros[f"follow.{tag}.{k}.k"] = f"{v['k']}/{v['n']}"

RESTORE_KEYS = {"openai-gpt-6-luna-floor__z-ai-glm-5.3-flash-floor_restore_channel_link_cost50": "LG_rest50",
                "openai-gpt-6-luna-floor__z-ai-glm-5.3-flash-floor_restore_channel_link_cost110": "LG_rest110",
                "z-ai-glm-5.3-flash-floor__openai-gpt-6-luna-floor_restore_channel_link_cost110": "GL_rest110",
                "deepseek-deepseek-v4.1-flash-floor__xiaomi-mimo-v2.6-flash-floor_restore_channel_link_cost50": "DM_rest50"}
for stem, key in RESTORE_KEYS.items():
    g = restore[stem]
    for seat in ("alice", "bob"):
        a = g["per_agent"][seat]
        macros[f"{key}.{seat}.payRate"] = pct(a["use_rate"])
        macros[f"{key}.{seat}.pays"] = str(a["uses"])
        macros[f"{key}.{seat}.net"] = num(a["net_reward_total"], 0)
        macros[f"{key}.{seat}.paidAcc"] = pct(a["verdicts_when_paid"]["accuracy"])
mc = json.loads((DATA / "msgcost.json").read_text())["groups"][0]
macros["price.chars"] = f"{mc['overall']['mean_chars_per_message']['combined']:,.0f}".replace(",", "{,}")
macros["price.free"] = pct(mc["overall"]["free_message_share"]["combined"])
macros["price.alice.bill"] = pct(mc["reward"]["alice"]["cost_share_of_counted"])
macros["price.bob.bill"] = pct(mc["reward"]["bob"]["cost_share_of_counted"])
for path, key in (("diff_openai-gpt-6-luna-floor__z-ai-glm-5.3-flash-floor-difficulty-label_true.json", "LG_diff"),
                  ("diff_z-ai-glm-5.3-flash-floor__openai-gpt-6-luna-floor-difficulty-label_true.json", "GL_diff")):
    c = json.loads((DATA / path).read_text())["conditions"][0]
    macros[f"{key}.gap"] = pp(c["gap_shown"]["value"])
    macros[f"{key}.gap.ci"] = ci(c["gap_shown"]["ci"])
    cov = {e["key"]: e for e in c["mentions"]["coverage"]}
    for seat in ("alice", "bob"):
        e = cov[f"by_{seat}"]
        macros[f"{key}.{seat}.mentions"] = f"{e['k']}/{e['bases'][0]['n']}"

lines = ["% Generated by make_numbers.py; do not edit.", "\\makeatletter"]
for k, v in sorted(macros.items()):
    lines.append(f"\\@namedef{{v@{k}}}{{{v}}}")
lines.append("\\makeatother")
lines.append("\\newcommand{\\V}[1]{\\ifcsname v@#1\\endcsname\\csname v@#1\\endcsname\\else\\textbf{??#1}\\fi}")
(HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
print(f"{len(macros)} macros; conditions: {len(numbers['conditions'])}")


# ---------------------------------------------------------------- tables
TABLES = HERE / "tables"
TABLES.mkdir(exist_ok=True)
SHORT = {"LG": "L$\\to$G", "GL": "G$\\to$L", "DD": "D$\\times$D", "DM": "D$\\times$M", "HM": "H$\\times$M"}


def n_cell(e):
    return f"{e['trajectories']}" + ("$^{*}$" if e["partial"] else "")


def ec_cell(e):
    ec = pct(e["metrics"]["EC"]["value"])
    if "evidence" in e:
        ev = e["evidence"]
        ec += f" ({pct(ev['unsupported_mutual_both']['lenient'] / ev['episodes'])})"
    return ec


main_rows = [("LG_base", "Baseline"), ("GL_base", "Baseline"), ("LG_pen", "Per-term penalty"),
             ("GL_pen", "Per-term penalty"), ("LG_count", "Episode countdown"), ("LG_countpen", "Countdown + penalty"),
             ("LG_fore", "Private forecast"), ("LG_seq", "Sequential verdict"), ("GL_seq", "Sequential verdict"),
             ("LG_diff", "Difficulty label"), ("GL_diff", "Difficulty label"), ("LG_rest50", "Paid restore, 50\\%"),
             ("LG_rest110", "Paid restore, 110\\%"), ("GL_rest110", "Paid restore, 110\\%"), ("LG_price", "Message price")]
lines = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
         "Lever & Pair & $N$ & EC (unsupp.) & Luna ACCEPT & GLM ACCEPT & Accuracy & Reward \\\\", "\\midrule"]
for i, (key, label) in enumerate(main_rows):
    e = numbers["conditions"][key]
    if i in (2, 4, 11):
        lines.append("\\addlinespace[2pt]")
    lines.append(" & ".join([label, SHORT[e["pair"]], n_cell(e), ec_cell(e),
                             pct(e["models"]["luna"]["accept_rate"]), pct(e["models"]["glm"]["accept_rate"]),
                             pct(e["metrics"]["verdict_accuracy"]["value"]),
                             num(e["metrics"]["joint_reward_mean"]["value"])]) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(TABLES / "main_results.tex").write_text("\n".join(lines) + "\n")

full_rows = [k for k, *_ in CONDITIONS]
# levels of every condition (portrait table)
lines = ["\\begin{tabular}{@{}llrrrrrrrrr@{}}", "\\toprule",
         "Condition & Pair & $N$ (eps) & EC & TC & CC & Onset & A ACC & B ACC & Acc. & Reward \\\\",
         "\\midrule"]
for key in full_rows:
    e = numbers["conditions"][key]
    mm = e["metrics"]
    lines.append(" & ".join([e["lever"], SHORT[e["pair"]], f"{n_cell(e)} ({e['episodes']})", ec_cell(e),
                             pct(mm["TC"]["value"]), pct(mm["CC"]["value"]), num(mm["onset_episode"]["value"]),
                             pct(mm["alice_accept_rate"]["value"]), pct(mm["bob_accept_rate"]["value"]),
                             pct(mm["verdict_accuracy"]["value"]), num(mm["joint_reward_mean"]["value"])]) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(TABLES / "full_results.tex").write_text("\n".join(lines) + "\n")

# each lever against its baseline: EC and verdict accuracy, with intervals and p-values
lines = ["\\begin{tabular}{@{}llrlrl@{}}", "\\toprule",
         "Condition & Pair & $\\Delta$EC & [95\\% CI], $p$ & $\\Delta$Acc. & [95\\% CI], $p$ \\\\",
         "\\midrule"]
for key in full_rows:
    e = numbers["conditions"][key]
    if not e.get("delta"):
        continue
    dec, dacc = e["delta"]["EC"], e["delta"]["verdict_accuracy"]
    lines.append(" & ".join([e["lever"], SHORT[e["pair"]], pp(dec["diff"]), f"{ci(dec['ci'])}, {pval(dec['p'])}",
                             pp(dacc["diff"]), f"{ci(dacc['ci'])}, {pval(dacc['p'])}"]) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(TABLES / "full_deltas.tex").write_text("\n".join(lines) + "\n")

# task-matched check: 5-trajectory conditions against baseline reps 1-5 (same task sequences)
lines = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
         "Condition & Pair & $\\Delta$EC vs 25 & $\\Delta$EC vs reps 1--5 & $\\Delta$Acc. vs 25 & $\\Delta$Acc. vs reps 1--5 \\\\",
         "\\midrule"]
for key in full_rows:
    e = numbers["conditions"][key]
    if "delta_matched" not in e:
        continue
    d, dm = e["delta"], e["delta_matched"]
    lines.append(" & ".join([e["lever"], SHORT[e["pair"]],
                             f"{pp(d['EC']['diff'])} ({pval(d['EC']['p'])})", f"{pp(dm['EC']['diff'])} ({pval(dm['EC']['p'])})",
                             f"{pp(d['verdict_accuracy']['diff'])} ({pval(d['verdict_accuracy']['p'])})",
                             f"{pp(dm['verdict_accuracy']['diff'])} ({pval(dm['verdict_accuracy']['p'])})"]) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(TABLES / "matched.tex").write_text("\n".join(lines) + "\n")
print("tables:", sorted(p.name for p in TABLES.iterdir()))
