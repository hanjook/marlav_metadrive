"""Top-down replay GIF: intersection, ego, fixed target region, distance to target, status.

`goal_reached` drives the simplest task DFA, WAITING -> DONE (this is not DFA-conditioned learning).

    python static_goal/replay.py                         # latest checkpoint, first evaluation episode
    python static_goal/replay.py --episode 3 --seed 0    # 4th episode of the same spawn sequence as evaluate.py
    python static_goal/replay.py --untrained
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import RESULTS, load_policy, run_episode  # noqa: E402
from static_goal_env import StaticGoalIntersectionEnv  # noqa: E402
from train_static_goal import latest_checkpoint  # noqa: E402


def dfa_state(goal_reached):
    """Simplest task DFA: WAITING until goal_reached, then DONE (absorbing)."""
    return "DONE" if goal_reached else "WAITING"


def draw_roads(ax, env):
    graph = env.current_map.road_network.graph
    for start in graph:
        for end in graph[start]:
            for lane in graph[start][end]:
                s = np.linspace(0, lane.length, 30)
                left = [lane.position(si, -lane.width / 2) for si in s]
                right = [lane.position(si, lane.width / 2) for si in s[::-1]]
                ax.add_patch(Polygon(left + right, closed=True, fc="#d9d9d9", ec="#bdbdbd", lw=0.4, zorder=0))


def car_corners(x, y, heading, length, width):
    c, s = np.cos(heading), np.sin(heading)
    pts = np.array([[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2],
                    [-length / 2, width / 2]])
    return pts @ np.array([[c, s], [-s, c]]) + [x, y]


def render(env, traj, summary, title, out_path, every=1, fps=20):
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=90)
    draw_roads(ax, env)
    region = env.goal_region()
    ax.add_patch(Polygon(region, closed=True, fc="#2ca02c", alpha=0.25, ec="#1a7a1a", lw=1.5, ls="--", zorder=1))
    center = region.mean(axis=0)
    ax.plot(*center, marker="*", color="#1a7a1a", ms=14, zorder=2)
    ax.text(region[:, 0].min(), region[:, 1].max() + 1.5, "fixed target", color="#1a7a1a", fontsize=9)
    ax.set_xlim(-5, 135)
    ax.set_ylim(-28, 46)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m] (east)")
    ax.set_ylabel("y [m]")
    length, width = env.agent.LENGTH, env.agent.WIDTH
    trail, = ax.plot([], [], color="#1f77b4", lw=1.2, zorder=3)
    car = Polygon(car_corners(traj[0]["x"], traj[0]["y"], traj[0]["heading"], length, width), fc="#1f77b4",
                  ec="k", zorder=4)
    ax.add_patch(car)
    hud = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9,
                  bbox=dict(fc="white", alpha=0.85, ec="#999999"))
    end_status = "SUCCESS" if summary["goal_reached"] else ("COLLISION" if summary["crash"] else
                                                             "OFF ROAD" if summary["out_of_road"] else "TIMEOUT")
    frames = []
    idx = list(range(0, len(traj), every))
    if idx[-1] != len(traj) - 1:
        idx.append(len(traj) - 1)
    for i in idx:
        p = traj[i]
        last = i == len(traj) - 1
        trail.set_data([q["x"] for q in traj[:i + 1]], [q["y"] for q in traj[:i + 1]])
        car.set_xy(car_corners(p["x"], p["y"], p["heading"], length, width))
        status = end_status if last else "driving"
        hud.set_text(
            f"step {i + 1:3d}  t={0.1 * (i + 1):4.1f}s  speed {p['speed']:4.1f} m/s\n"
            f"distance to target {p['dist_to_goal']:6.1f} m\n"
            f"goal_reached={p['goal_reached']}  DFA: {dfa_state(p['goal_reached'])}\n"
            f"status: {status}"
        )
        hud.get_bbox_patch().set_edgecolor("#1a7a1a" if p["goal_reached"] else ("#c0392b" if last else "#999999"))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    frames += [frames[-1]] * fps  # hold the final frame for 1 s
    imageio.mimsave(out_path, frames, duration=1000 / fps, loop=0)  # duration is ms per frame
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0, help="index into the seeded spawn sequence")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    label = "untrained" if args.untrained else "trained"
    ckpt = None if args.untrained else (args.checkpoint or latest_checkpoint())
    env = StaticGoalIntersectionEnv(dict(spawn_seed=args.seed))
    policy = load_policy(ckpt, untrained=args.untrained, env=env, seed=args.seed)
    for _ in range(args.episode):  # advance the spawn sequence so episode k matches evaluate.py's episode k
        env.reset()
    summary, traj = run_episode(env, policy)
    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(RESULTS, f"replay_{label}_ep{args.episode}.gif")
    render(env, traj, summary, f"{label} PPO policy, episode {args.episode} (seed {args.seed})", out)
    env.close()
    print(json.dumps(dict(goal_reached=summary["goal_reached"], dfa_state=dfa_state(summary["goal_reached"]),
                          **{k: v for k, v in summary.items() if k != "goal_reached"}), indent=2))
    print("Saved", out)
