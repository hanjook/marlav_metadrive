"""Evaluate policies on the traffic task with reproducible ego-start and traffic seeds.

Every group sees the same episodes: ego start k comes from the --seed spawn sequence and the traffic of episode k
from the --seed traffic sequence, so episode k is the same situation for every policy.

Groups:
  original_det        no-traffic baseline checkpoint (static_goal/checkpoint_final), deterministic actions
  traffic_det         traffic-trained checkpoint, deterministic (mean) actions
  traffic_stoch       traffic-trained checkpoint, sampled actions (torch seeded with --seed)
  script              constant-throttle script: steering 0, throttle 0.5
  traffic_det_empty   traffic-trained checkpoint, deterministic, same ego starts with traffic switched off
                      (used only for the interaction check: same policy, same start, with vs without traffic)

    python static_goal_traffic/evaluate_traffic.py --episodes 50 --seed 0
Writes results/eval_<group>_seed<seed><tag>.json and results/summary_seed<seed><tag>.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "static_goal"))
from evaluate import load_policy  # noqa: E402
from traffic_env import StaticGoalTrafficEnv  # noqa: E402
from train_traffic import latest_checkpoint  # noqa: E402

RESULTS = os.path.join(HERE, "results")
ORIGINAL_CKPT = os.path.join(ROOT, "static_goal", "checkpoint_final")
TRAFFIC_CKPT = os.path.join(HERE, "checkpoint_final")
SCRIPT_ACTION = np.array([0.0, 0.5])
OUTCOMES = ("goal_reached", "crash_vehicle", "out_of_road", "crash_other", "timeout")


def episode_outcome(info):
    """Exactly one outcome per episode, from the last step's info. Vehicle collisions are kept apart from off-road."""
    goal = bool(info["goal_reached"])
    vehicle = bool(info["crash_vehicle"]) and not goal
    off_road = (bool(info["out_of_road"]) or bool(info["crash_sidewalk"]) or bool(info["crash_building"])) \
        and not (goal or vehicle)
    other = (bool(info["crash_object"]) or bool(info["crash_human"])) and not (goal or vehicle or off_road)
    return dict(goal_reached=goal, crash_vehicle=vehicle, out_of_road=off_road, crash_other=other,
                timeout=not (goal or vehicle or off_road or other))


def run_episode(env, act):
    obs, info = env.reset()
    dt = env.config["physics_world_step_size"] * env.config["decision_repeat"]
    total, traj = 0.0, []
    n_vehicles, traffic_seed, min_dist = info["num_traffic_vehicles"], info["traffic_seed"], np.inf
    while True:
        action = np.clip(act(obs), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        # The action the ego used must be the policy's action (no other controller in the loop).
        assert info["policy"] == "EnvInputPolicy" and np.allclose(info["raw_action"], action, atol=1e-6)
        total += reward
        min_dist = min(min_dist, info["min_dist_to_traffic"])
        v = env.agent
        traj.append((float(v.position[0]), float(v.position[1]), float(v.speed)))
        if terminated or truncated:
            break
    out = episode_outcome(info)
    summary = dict(
        out,
        reward=float(total),
        steps=len(traj),
        seconds=round(len(traj) * dt, 2),
        time_to_goal=round(len(traj) * dt, 2) if out["goal_reached"] else None,
        final_dist_to_goal=info["dist_to_goal"],
        traffic_seed=traffic_seed,
        num_traffic_vehicles=n_vehicles,
        min_dist_to_traffic=None if np.isinf(min_dist) else round(float(min_dist), 2),
        max_slowdown=round(max_slowdown([s for _, _, s in traj]), 2),
        mean_speed=round(float(np.mean([s for _, _, s in traj])), 2),
    )
    return summary, traj


def max_slowdown(speeds):
    """Largest drop below the running peak speed (m/s): how much the ego braked at some point after speeding up."""
    peak, drop = 0.0, 0.0
    for v in speeds:
        peak = max(peak, v)
        drop = max(drop, peak - v)
    return drop


def policy_actor(policy, explore):
    return lambda obs: policy.compute_single_action(obs, explore=explore)[0]


def evaluate_group(env, act, episodes, seed, traffic=True, explore=False):
    import torch
    torch.manual_seed(seed)  # makes sampled actions reproducible
    env.config["traffic_density"] = env.default_config()["traffic_density"] if traffic else 0.0
    env.seed_spawns(seed)
    env.seed_traffic(seed)
    rows, trajs = [], []
    for ep in range(episodes):
        summary, traj = run_episode(env, act)
        rows.append(dict(episode=ep, **summary))
        trajs.append(traj)
    return rows, trajs


def summarize(rows):
    def rate(key):
        return float(np.mean([r[key] for r in rows]))

    times = [r["time_to_goal"] for r in rows if r["goal_reached"]]
    return dict(
        episodes=len(rows),
        success_rate=rate("goal_reached"),
        vehicle_collision_rate=rate("crash_vehicle"),
        off_road_rate=rate("out_of_road"),
        other_crash_rate=rate("crash_other"),
        timeout_rate=rate("timeout"),
        counts={k: int(sum(r[k] for r in rows)) for k in OUTCOMES},
        mean_reward=float(np.mean([r["reward"] for r in rows])),
        std_reward=float(np.std([r["reward"] for r in rows])),
        mean_time_to_goal=float(np.mean(times)) if times else None,
        mean_steps=float(np.mean([r["steps"] for r in rows])),
        mean_traffic_vehicles=float(np.mean([r["num_traffic_vehicles"] for r in rows])),
    )


def interaction_check(with_traffic, without_traffic, rows_with, rows_without):
    """Same policy, same start, with vs without traffic: does traffic change what the ego does?"""
    per_episode = []
    for k, (a, b) in enumerate(zip(with_traffic, without_traffic)):
        n = min(len(a), len(b))
        divergence = max(float(np.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1])) for i in range(n))
        per_episode.append(dict(
            episode=k,
            max_position_difference_m=round(divergence, 2),
            max_slowdown_with=rows_with[k]["max_slowdown"],
            max_slowdown_without=rows_without[k]["max_slowdown"],
            outcome_with=next(o for o in OUTCOMES if rows_with[k][o]),
            outcome_without=next(o for o in OUTCOMES if rows_without[k][o]),
            seconds_with=rows_with[k]["seconds"],
            seconds_without=rows_without[k]["seconds"],
        ))
    div = [p["max_position_difference_m"] for p in per_episode]
    return dict(
        episodes_trajectory_changed_over_1m=int(sum(d > 1.0 for d in div)),
        median_max_position_difference_m=float(np.median(div)),
        episodes_braked_over_2mps_with=int(sum(p["max_slowdown_with"] > 2.0 for p in per_episode)),
        episodes_braked_over_2mps_without=int(sum(p["max_slowdown_without"] > 2.0 for p in per_episode)),
        episodes_outcome_changed=int(sum(p["outcome_with"] != p["outcome_without"] for p in per_episode)),
        per_episode=per_episode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="ego-start and traffic sequence seed (same for every group)")
    parser.add_argument("--traffic-checkpoint", type=str, default=None)
    parser.add_argument("--groups", type=str, default="original_det,traffic_det,traffic_stoch,script,traffic_det_empty")
    parser.add_argument("--tag", type=str, default="", help="suffix for result file names, e.g. _ckpt500k")
    args = parser.parse_args()

    traffic_ckpt = args.traffic_checkpoint or (TRAFFIC_CKPT if os.path.isdir(TRAFFIC_CKPT) else latest_checkpoint())
    env = StaticGoalTrafficEnv()
    original = load_policy(ORIGINAL_CKPT)
    trained = load_policy(traffic_ckpt)
    groups = {
        "original_det": (policy_actor(original, False), True, ORIGINAL_CKPT),
        "traffic_det": (policy_actor(trained, False), True, traffic_ckpt),
        "traffic_stoch": (policy_actor(trained, True), True, traffic_ckpt),
        "script": (lambda obs: SCRIPT_ACTION, True, None),
        "traffic_det_empty": (policy_actor(trained, False), False, traffic_ckpt),
    }
    os.makedirs(RESULTS, exist_ok=True)
    cfg = env.config
    env_config = dict(map=cfg["map"], traffic_density=env.default_config()["traffic_density"],
                      traffic_mode=cfg["traffic_mode"], random_traffic=cfg["random_traffic"], horizon=cfg["horizon"],
                      spawn_seed=args.seed, traffic_seed=args.seed)
    summary, trajs, rows_by_group = {}, {}, {}
    try:
        for name in args.groups.split(","):
            act, traffic, ckpt = groups[name]
            rows, trajs[name] = evaluate_group(env, act, args.episodes, args.seed, traffic, explore=name == "traffic_stoch")
            rows_by_group[name] = rows
            summary[name] = dict(checkpoint=ckpt, traffic=traffic, **summarize(rows))
            with open(os.path.join(RESULTS, f"eval_{name}_seed{args.seed}{args.tag}.json"), "w") as f:
                json.dump(dict(group=name, checkpoint=ckpt, env_config=env_config if traffic else
                               dict(env_config, traffic_density=0.0), summary=summary[name], episodes=rows), f, indent=2)
            s = summary[name]
            print(f"{name:18s} success {s['counts']['goal_reached']:2d}/{s['episodes']}  vehicle-collision "
                  f"{s['counts']['crash_vehicle']:2d}  off-road {s['counts']['out_of_road']:2d}  other "
                  f"{s['counts']['crash_other']}  timeout {s['counts']['timeout']:2d}  reward {s['mean_reward']:6.1f}  "
                  f"time-to-goal {s['mean_time_to_goal']}  vehicles {s['mean_traffic_vehicles']:.1f}")
    finally:
        env.close()
    out = dict(seed=args.seed, env_config=env_config, groups=summary)
    if "traffic_det" in trajs and "traffic_det_empty" in trajs:
        out["interaction_check"] = interaction_check(trajs["traffic_det"], trajs["traffic_det_empty"],
                                                     rows_by_group["traffic_det"], rows_by_group["traffic_det_empty"])
        ic = out["interaction_check"]
        print(f"interaction check: trajectory changed >1 m in {ic['episodes_trajectory_changed_over_1m']}/"
              f"{args.episodes} episodes, outcome changed in {ic['episodes_outcome_changed']}")
    with open(os.path.join(RESULTS, f"summary_seed{args.seed}{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved", os.path.join(RESULTS, f"summary_seed{args.seed}{args.tag}.json"))
