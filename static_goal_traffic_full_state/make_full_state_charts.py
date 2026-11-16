"""Charts from the actual full-state result files.

    media/training_curves_by_seed.png   training success and crash rate per seed (sampled actions), LiDAR run as reference
    media/validation_grid.png           validation success per seed, checkpoint and action mode (selection data)
    media/test_outcomes.png             held-out test outcomes of the compared policies

    python static_goal_traffic_full_state/make_full_state_charts.py
"""
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "media")
RESULTS = os.path.join(HERE, "results")
INK, MUTED, GRID, GREY = "#1F2933", "#52616B", "#E1E0D9", "#9AA5B1"
SEED_COLORS = {0: "#0072B2", 1: "#E69F00", 2: "#009E73"}  # Okabe-Ito, colour-blind safe
OUTCOMES = [("Success", "goal_reached", "#0CA30C"), ("Vehicle collision", "crash_vehicle", "#D03B3B"),
            ("Off-road", "out_of_road", "#EC835A"), ("Timeout", "timeout", "#FAB219")]
plt.rcParams.update({"font.family": "Arial", "font.size": 15, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})


def smooth(y, w=5):
    y = np.asarray(y, dtype=float)
    return np.array([y[max(0, i - w + 1):i + 1].mean() for i in range(len(y))])


def progress(trial_glob):
    rows = []
    for p in glob.glob(trial_glob):
        with open(p) as f:
            rows += list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["timesteps_total"]))
    return rows


def style(ax):
    ax.grid(axis="y", color=GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def training_curves():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.4), dpi=200, sharex=True)
    lidar = progress(os.path.join(ROOT, "static_goal_traffic", "runs", "static_goal_traffic_ppo", "*", "progress.csv"))
    a1.plot([int(r["timesteps_total"]) / 1e6 for r in lidar],
            smooth([100 * float(r["custom_metrics/success_rate_mean"]) for r in lidar]), color=GREY, lw=2, ls="--",
            label="LiDAR run (previous experiment, seed 0)")
    for trial in sorted(glob.glob(os.path.join(HERE, "runs", "full_state_ppo", "PPO_*"))):
        seed = json.load(open(os.path.join(trial, "params.json")))["seed"]
        rows = progress(os.path.join(trial, "progress.csv"))
        x = [int(r["timesteps_total"]) / 1e6 for r in rows]
        a1.plot(x, smooth([100 * float(r["custom_metrics/success_rate_mean"]) for r in rows]), color=SEED_COLORS[seed],
                lw=2, label=f"Full state, seed {seed}")
        a2.plot(x, smooth([100 * float(r["custom_metrics/crash_rate_mean"]) for r in rows]), color=SEED_COLORS[seed],
                lw=2, label=f"Full state, seed {seed}")
    a2.plot([int(r["timesteps_total"]) / 1e6 for r in lidar],
            smooth([100 * float(r["custom_metrics/crash_rate_mean"]) for r in lidar]), color=GREY, lw=2, ls="--",
            label="LiDAR run")
    a1.set_title("Success rate in training episodes (sampled actions)", loc="left", fontsize=16, fontweight="bold")
    a2.set_title("Crash rate in training episodes (any crash type)", loc="left", fontsize=16, fontweight="bold")
    for a in (a1, a2):
        a.set_ylim(0, 100)
        a.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        style(a)
    a1.legend(frameon=False, fontsize=12, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.08))
    a2.set_xlim(0, 1.0)
    a2.set_xlabel("Training steps (millions)")
    fig.text(0.02, 0.01, "Actual training logs; each point is a 5-iteration moving average (about 75-200 episodes). "
             "Training rates use sampled actions and are not the evaluation result.", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "training_curves_by_seed.png"))
    plt.close(fig)


def validation_grid():
    val = json.load(open(os.path.join(RESULTS, "summary.json")))["validation"]
    sel = json.load(open(os.path.join(RESULTS, "selection.json")))
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), dpi=200, sharey=True)
    panels = [(f"Full state, seed {s}", lambda k, m, s=s: f"fs_s{s}_{k}k_{m}", (250, 500, 750, 1000), SEED_COLORS[s])
              for s in (0, 1, 2)]
    panels.append(("LiDAR (previous run)", lambda k, m: f"lidar_{k}k_{m}", (240, 500, 760, 1000), GREY))
    for ax, (title, name, ks, color) in zip(axes, panels):
        for m, ls, mk in (("det", "-", "o"), ("stoch", "--", "s")):
            ys = [val[name(k, m)]["goal_reached"] for k in ks]
            ax.plot([k / 1000 for k in ks], ys, ls=ls, marker=mk, ms=8, lw=2, color=color,
                    label="deterministic" if m == "det" else "stochastic")
            for k, y in zip(ks, ys):
                if name(k, m) in (sel["full_state"], sel["lidar"]):
                    ax.plot(k / 1000, y, marker="*", ms=20, color=INK, zorder=5)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
        ax.set_xticks([0.25, 0.5, 0.75, 1.0], ["250k", "500k", "750k", "1M"])
        ax.set_ylim(0, 100)
        style(ax)
    axes[0].set_ylabel("Validation successes (of 100)")
    axes[0].legend(frameon=False, fontsize=12, loc="lower left")
    fig.suptitle("Validation set (seeds 100, 101): success by checkpoint and action mode", x=0.01, ha="left",
                 fontsize=17, fontweight="bold")
    fig.text(0.01, 0.01, "Star = checkpoint and mode selected on validation for the test set. LiDAR checkpoints are the "
             "saved ones nearest each milestone (240k, 500k, 760k, 1M).", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(OUT, "validation_grid.png"))
    plt.close(fig)


def label(name, sel):
    if name == "original_det":
        return "No-traffic policy"
    if name == "script":
        return "Constant-throttle script"
    mode = "sampled" if "_stoch" in name else "mean action"
    off = ", traffic OFF" if name.endswith("_off") else ""
    if name.startswith("lidar"):
        return f"LiDAR policy ({name.split('_')[1]}, {mode}){off}"
    parts = name.split("_")
    best = " [selected]" if name in (sel["full_state"], sel["full_state"] + "_off") else ""
    return f"Full state seed {parts[1][1:]} ({parts[2]}, {mode}){off}{best}"


def test_chart():
    s = json.load(open(os.path.join(RESULTS, "summary.json")))
    sel = s["selection"]
    order = ["original_det", "script", sel["lidar"], sel["full_state"]]
    order += [g for g in sel["full_state_per_seed"].values() if g not in order]
    order += [sel["full_state"] + "_off"]
    rows = [(label(g, sel), s["test"][g]) for g in order]
    fig, ax = plt.subplots(figsize=(12.5, 0.75 * len(rows) + 2.2), dpi=200)
    for i, (_, t) in enumerate(rows):
        left = 0
        for _, key, color in OUTCOMES:
            v = t[key]
            if v:
                ax.barh(i, v, left=left, color=color, edgecolor="white", linewidth=2, height=0.62)
                if v >= 3:
                    ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
            left += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=13, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Test episodes (out of 100)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in OUTCOMES]
    ax.legend(handles, [o[0] for o in OUTCOMES], ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=13)
    fig.suptitle("Held-out test set: 100 episodes, same starts and traffic for every policy", x=0.01, ha="left",
                 fontsize=17, fontweight="bold")
    fig.text(0.01, 0.01, "Test seeds 0 and 1 (50 episodes each), 6 IDM vehicles per episode. Checkpoints and action "
             "modes were chosen on the validation set only.", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "test_outcomes.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    training_curves()
    validation_grid()
    test_chart()
    print("Saved media/training_curves_by_seed.png, media/validation_grid.png, media/test_outcomes.png")
