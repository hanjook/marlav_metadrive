"""Charts from the actual traffic result files (does not touch meeting/media).

    media/evaluation_results.png   outcome counts per group from results/eval_*_seed0.json
    media/training_curve.png       training success / crash / off-road rate from the traffic run's progress.csv,
                                   with the no-traffic run's success rate for reference

    python static_goal_traffic/make_traffic_charts.py
"""
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "media")
INK, MUTED, GRID, GREY = "#1F2933", "#52616B", "#E1E0D9", "#9AA5B1"
OUTCOMES = [("Success", "goal_reached", "#0CA30C"), ("Vehicle collision", "crash_vehicle", "#D03B3B"),
            ("Off-road", "out_of_road", "#EC835A"), ("Timeout", "timeout", "#FAB219")]
GROUPS = [("No-traffic policy (original)", "original_det"), ("Traffic-trained, deterministic", "traffic_det"),
          ("Traffic-trained, stochastic", "traffic_stoch"), ("Constant-throttle script", "script")]
plt.rcParams.update({"font.family": "Arial", "font.size": 16, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})


def evaluation_chart(seed=0):
    rows = []
    for label, name in GROUPS:
        with open(os.path.join(HERE, "results", f"eval_{name}_seed{seed}.json")) as f:
            data = json.load(f)
        eps = data["episodes"]
        rows.append((label, len(eps), {key: sum(e[key] for e in eps) for _, key, _ in OUTCOMES}))
        assert sum(e["crash_other"] for e in eps) == 0  # no other crash types to show
    cfg = data["env_config"]
    n = rows[0][1]
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=200)
    for i, (label, _, counts) in enumerate(rows):
        left = 0
        for name, key, color in OUTCOMES:
            v = counts[key]
            if v:
                ax.barh(i, v, left=left, color=color, edgecolor="white", linewidth=2, height=0.6)
                if v >= 2:
                    ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
            left += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=16, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, n)
    ax.set_xticks(range(0, n + 1, 10))
    ax.set_xlabel(f"Episodes (out of {n})")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in OUTCOMES]
    ax.legend(handles, [o[0] for o in OUTCOMES], ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              fontsize=15)
    fig.suptitle(f"Outcomes with background traffic, {n} episodes per group", x=0.02, ha="left", fontsize=20,
                 fontweight="bold")
    fig.text(0.02, 0.015, f"Same {n} ego starts and traffic for every group (spawn seed {cfg['spawn_seed']}, traffic seed "
             f"{cfg['traffic_seed']}; density {cfg['traffic_density']}, {cfg['traffic_mode']} mode, IDM vehicles).",
             fontsize=12, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, "evaluation_results.png"))
    plt.close(fig)
    return rows


def read_progress(pattern):
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    rows = []
    for p in paths:  # a resumed run writes a new trial directory; concatenate by timesteps
        with open(p) as f:
            rows += list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["timesteps_total"]))
    return rows


def training_curve():
    rows = read_progress(os.path.join(HERE, "runs", "static_goal_traffic_ppo", "*", "progress.csv"))
    base = read_progress(os.path.join(ROOT, "static_goal", "runs", "static_goal_ppo", "*", "progress.csv"))
    steps = [int(r["timesteps_total"]) / 1e6 for r in rows]
    last = steps[-1]
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=200)
    base = [r for r in base if int(r["timesteps_total"]) / 1e6 <= last + 1e-9]
    ax.plot([int(r["timesteps_total"]) / 1e6 for r in base],
            [100 * float(r["custom_metrics/success_rate_mean"]) for r in base], color=GREY, lw=2, ls="--",
            label="Success, no-traffic run (reference)")
    for label, key, color in (("Success", "success_rate_mean", "#0CA30C"),
                              ("Crash (any: vehicle, sidewalk, object)", "crash_rate_mean", "#D03B3B"),
                              ("Off-road", "out_of_road_rate_mean", "#EC835A")):
        ax.plot(steps, [100 * float(r[f"custom_metrics/{key}"]) for r in rows], color=color, lw=2, label=label)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlim(0, last)
    ax.set_xlabel("Training steps (millions)")
    ax.grid(axis="y", color=GRID)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(0.505, 97, "resumed from the 500k checkpoint", color=MUTED, fontsize=12, va="top")
    ax.legend(frameon=False, fontsize=13, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    fig.suptitle("Training episodes with traffic (sampled actions)", x=0.02, ha="left", fontsize=20,
                 fontweight="bold")
    fig.text(0.02, 0.012, f"Actual training log: {len(rows)} PPO iterations, {int(rows[-1]['timesteps_total']):,} steps. "
             "Rates are per training iteration (about 15-40 episodes each).", fontsize=12, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "training_curve.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for label, n, counts in evaluation_chart():
        print(label, n, counts)
    training_curve()
    print("Saved media/evaluation_results.png and media/training_curve.png")
