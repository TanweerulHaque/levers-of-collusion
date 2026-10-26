"""Figures for the report and the slides, from data/numbers.json (run make_numbers.py first).

Colours come from the dataviz reference palette, used unchanged and in its validated
order; each model keeps one colour in every figure. Writes figures/*.pdf and *.png.

    python make_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
N = json.loads((HERE / "data" / "numbers.json").read_text())
C = N["conditions"]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
MODEL_COLOR = {"luna": "#2a78d6", "glm": "#eb6834", "deepseek": "#1baf7a", "haiku": "#6250d6", "mimo": "#e87ba4"}
MODEL_NAME = {"luna": "GPT-6-Luna", "glm": "GLM-5.3-Flash", "deepseek": "DeepSeek V4.1 Flash",
              "haiku": "Claude Haiku 5.5", "mimo": "MiMo-V2.6-Flash"}
RED = "#e34948"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def m(key, metric):
    return C[key]["metrics"][metric]


# ---------------------------------------------------------------- Figure 2: the model lever
def fig_models():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.05), gridspec_kw={"width_ratios": [1.0, 1], "wspace": 0.95})
    rows = [("HM_fore", "Haiku $\\times$ MiMo"), ("LG_base", "Luna $\\to$ GLM"), ("GL_base", "GLM $\\to$ Luna"),
            ("DD_base", "DeepSeek $\\times$ DeepSeek"), ("DM_base", "DeepSeek $\\times$ MiMo")]
    for i, (key, label) in enumerate(rows):
        e = m(key, "EC")
        v = e["value"] or 0.0
        ax1.barh(i, 100 * v, height=0.42, color="#2a78d6", zorder=2)
        if e["ci"]:
            ax1.plot([100 * e["ci"][0], 100 * e["ci"][1]], [i, i], color=INK, linewidth=0.9, zorder=3)
        ax1.text(100 * max(v, (e["ci"] or [0, v])[1]) + 2.5, i, f"{100 * v:.0f}%", va="center", fontsize=7.5, color=INK)
    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels([f"{lab}  (N={C[k]['trajectories']})" for k, lab in rows])
    ax1.set_xlim(0, 105)
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax1.set_xlabel("Episodes with mutual ACCEPT (EC)")
    ax1.set_title("(a) Collusion by model pair", loc="left")
    grid(ax1)
    ax1.tick_params(axis="y", length=0)

    seats = {
        "haiku": [("alice", m("HM_fore", "alice_accept_rate")["value"])],
        "luna": [("alice", m("LG_base", "alice_accept_rate")["value"]), ("bob", m("GL_base", "bob_accept_rate")["value"])],
        "glm": [("alice", m("GL_base", "alice_accept_rate")["value"]), ("bob", m("LG_base", "bob_accept_rate")["value"])],
        "mimo": [("bob", m("DM_base", "bob_accept_rate")["value"]), ("bob", m("HM_fore", "bob_accept_rate")["value"])],
        "deepseek": [("alice", m("DD_base", "alice_accept_rate")["value"]), ("bob", m("DD_base", "bob_accept_rate")["value"])],
    }
    order = ["haiku", "luna", "glm", "mimo", "deepseek"]
    for i, model in enumerate(order):
        xs = [100 * v for _, v in seats[model]]
        if len(xs) > 1:
            ax2.plot([min(xs), max(xs)], [i, i], color=AXIS, linewidth=2, zorder=1, solid_capstyle="round")
        for seat, v in seats[model]:
            if seat == "bob":  # ring drawn first and larger, so a filled Alice dot at the same value stays visible
                ax2.scatter(100 * v, i, s=70, zorder=3, color="white", edgecolors=MODEL_COLOR[model], linewidths=1.6)
        for seat, v in seats[model]:
            if seat == "alice":
                ax2.scatter(100 * v, i, s=24, zorder=4, color=MODEL_COLOR[model], edgecolors=MODEL_COLOR[model])
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([MODEL_NAME[x] for x in order])
    ax2.set_xlim(-4, 104)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax2.set_xlabel("Share of verdicts that ACCEPT")
    ax2.set_title("(b) Acceptance by model and seat", loc="left")
    ax2.tick_params(axis="y", length=0)
    grid(ax2)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=5.5, markerfacecolor=INK2, markeredgecolor=INK2, label="as Alice"),
               Line2D([], [], marker="o", linestyle="", markersize=5.5, markerfacecolor="white", markeredgecolor=INK2,
                      markeredgewidth=1.4, label="as Bob")]
    ax2.legend(handles=handles, loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.1)
    save(fig, "fig_models")


# ---------------------------------------------------------------- Figure 3: learning, and the penalty
def mean_curves(*curves):
    return [sum(c[t] for c in curves) / len(curves) for t in range(10)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fig_dynamics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.15), gridspec_kw={"wspace": 0.38})
    cur = {k: C[k]["curves"] for k in ("LG_base", "GL_base", "DD_base", "HM_fore")}
    lines = {
        "deepseek": mean_curves(cur["DD_base"]["alice_accept"], cur["DD_base"]["bob_accept"]),
        "glm": mean_curves(cur["LG_base"]["bob_accept"], cur["GL_base"]["alice_accept"]),
        "luna": mean_curves(cur["LG_base"]["alice_accept"], cur["GL_base"]["bob_accept"]),
        "haiku": [v if v is not None else float("nan") for v in cur["HM_fore"]["alice_accept"]],
    }
    x = list(range(1, 11))
    for model, ys in lines.items():
        ax1.plot(x, [100 * y for y in ys], color=MODEL_COLOR[model], linewidth=2, solid_capstyle="round",
                 solid_joinstyle="round", label=MODEL_NAME[model])
        ax1.scatter([10], [100 * ys[-1]], s=16, color=MODEL_COLOR[model], edgecolors="white", linewidths=1, zorder=3)
    ax1.set_ylim(-3, 103)
    ax1.set_xticks(x)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("ACCEPT rate")
    ax1.set_title("(a) Agents start compliant, then accept", loc="left")
    grid(ax1, "y")
    ax1.legend(loc="center right", frameon=False, handlelength=1.4, borderaxespad=0.2, bbox_to_anchor=(1.0, 0.25),
               labelspacing=0.3)

    for key, color, label in (("LG_base", INK2, "Baseline"), ("LG_pen", RED, "Per-term penalty")):
        ys = C[key]["curves"]["ec"]
        n = C[key]["trajectories"]
        lo, hi = zip(*[wilson(round(y * n), n) for y in ys])
        ax2.fill_between(x, [100 * v for v in lo], [100 * v for v in hi], color=color, alpha=0.10, linewidth=0)
        ax2.plot(x, [100 * y for y in ys], color=color, linewidth=2, solid_capstyle="round", label=f"{label} (N={n})")
    ax2.set_ylim(-3, 103)
    ax2.set_xticks(x)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Mutual ACCEPT (EC)")
    ax2.set_title("(b) The penalty leaves EC unchanged", loc="left")
    grid(ax2, "y")
    ax2.legend(loc="upper left", frameon=False, handlelength=1.4, borderaxespad=0.2)
    save(fig, "fig_dynamics")


# ---------------------------------------------------------------- Figure 4: every lever against its baseline
LEVER_ROWS = [
    ("Incentive", [("LG_pen", "Per-term penalty", "L$\\to$G"), ("GL_pen", "Per-term penalty", "G$\\to$L"),
                   ("LG_countpen", "Countdown + penalty$^{\\dagger}$", "L$\\to$G")]),
    ("Information", [("LG_count", "Episode countdown", "L$\\to$G"), ("LG_fore", "Private forecast", "L$\\to$G"),
                     ("LG_seq", "Sequential verdict", "L$\\to$G"), ("GL_seq", "Sequential verdict", "G$\\to$L"),
                     ("LG_diff", "Difficulty label", "L$\\to$G"), ("GL_diff", "Difficulty label", "G$\\to$L")]),
    ("Evidence", [("LG_rest50", "Paid restore, 50%", "L$\\to$G"), ("LG_rest110", "Paid restore, 110%", "L$\\to$G"),
                  ("GL_rest110", "Paid restore, 110%", "G$\\to$L"), ("LG_price", "Message price$^{\\ddagger}$", "L$\\to$G")]),
]


def fig_levers(name="fig_levers", width=5.5):
    rows, labels, sections = [], [], []
    y = 0
    for section, items in LEVER_ROWS:
        sections.append((section, y))
        y += 0.9
        for key, label, pair in items:
            rows.append((key, y))
            labels.append((y, f"{label}  {pair}  (N={C[key]['trajectories']})"))
            y += 1
        y += 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, 0.19 * y + 0.7), sharey=True, gridspec_kw={"wspace": 0.08})
    band = N["extras"]["noise_band"]["LG_base25"]
    base = C["LG_base"]["metrics"]
    for ax, metric, title, bkey in ((ax1, "EC", "$\\Delta$ unsupported mutual ACCEPT (pp)", "EC"),
                                    (ax2, "verdict_accuracy", "$\\Delta$ verdict accuracy (pp)", "verdict_accuracy")):
        lo = 100 * (band[bkey]["p2.5"] - base[bkey]["value"])
        hi = 100 * (band[bkey]["p97.5"] - base[bkey]["value"])
        ax.axvspan(lo, hi, color=GRID, alpha=0.55, linewidth=0, zorder=0)
        ax.axvline(0, color=AXIS, linewidth=0.9, zorder=1)
        for key, yy in rows:
            d = C[key]["delta"]["unsupported_EC" if (metric == "EC" and "unsupported_EC" in C[key]["delta"]) else metric]
            v, ci = 100 * d["diff"], d["ci"]
            sig = ci is not None and (ci[0] > 0 or ci[1] < 0)
            if ci:
                ax.plot([100 * ci[0], 100 * ci[1]], [yy, yy], color=INK, linewidth=1.0, zorder=2, solid_capstyle="round")
            ax.scatter(v, yy, s=26, zorder=3, color=INK if sig else "white", edgecolors=INK, linewidths=1.2)
        ax.set_xlim(-62, 62)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xticks([-50, -25, 0, 25, 50])
        ax.set_title(title, loc="left")
        grid(ax)
        ax.tick_params(axis="y", length=0)
    ax1.set_yticks([yy for yy, _ in labels])
    ax1.set_yticklabels([t for _, t in labels])
    for section, yy in sections:
        ax1.text(-0.02, yy, section, transform=ax1.get_yaxis_transform(), ha="right", va="center",
                 fontsize=8, fontweight="bold", color=INK)
    ax1.set_ylim(y - 0.2, -0.6)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor=INK, markeredgecolor=INK,
                      label="95% CI excludes 0"),
               Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor="white", markeredgecolor=INK,
                      label="95% CI includes 0"),
               Patch(facecolor=GRID, alpha=0.55, edgecolor="none",
                     label="where 95% of 5-trajectory samples of the L$\\to$G baseline fall")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.3, columnspacing=1.4, fontsize=7)
    save(fig, name)


if __name__ == "__main__":
    fig_models()
    fig_dynamics()
    fig_levers()
    print(sorted(p.name for p in OUT.iterdir()))
