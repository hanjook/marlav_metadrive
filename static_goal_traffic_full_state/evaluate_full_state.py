"""Evaluation for the full-state experiment: validation grid, checkpoint selection, held-out test, report.

Episode sets (ego-start seed = traffic seed, 50 episodes per seed; every group sees the same episodes):
  validation  seeds 100, 101   used only to choose checkpoints and action modes
  test        seeds 0, 1       held out; the same 100 test episodes as static_goal_traffic/TRAFFIC_RESULTS.md

All groups run in StaticGoalTrafficFullStateEnv with extra_lidar_obs=True, so LiDAR-trained checkpoints get the
previous 259-dim observation (info["lidar_obs"]) on exactly the same simulated episodes.

Group names:
  fs_s<seed>_<k>k_<det|stoch>[_off]   full-state checkpoint (k = 250, 500, 750, 1000); _off = traffic disabled
  lidar_<k>k_<det|stoch>[_off]        LiDAR traffic checkpoint from static_goal_traffic (k = 240, 500, 760, 1000)
  original_det                        no-traffic checkpoint from static_goal
  script                              constant throttle 0.5, steering 0

    python evaluate_full_state.py grid                         # list validation groups
    python evaluate_full_state.py run --set validation --groups fs_s0_250k_det,...
    python evaluate_full_state.py select                       # results/selection.json from validation only
    python evaluate_full_state.py test                         # run the selected groups on the test set
    python evaluate_full_state.py report                       # results/*.csv, results/summary.json
"""
import argparse
import csv
import glob
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "static_goal_traffic"))
sys.path.insert(0, os.path.join(ROOT, "static_goal"))
from evaluate_traffic import ORIGINAL_CKPT, SCRIPT_ACTION, episode_outcome, max_slowdown  # noqa: E402
from full_state_env import StaticGoalTrafficFullStateEnv  # noqa: E402

RESULTS = os.path.join(HERE, "results")
SETS = {"validation": (100, 101), "test": (0, 1), "equivalence_check": (0, 1)}  # last: reproduce earlier results
EPISODES_PER_SEED = 50
FS_MILESTONES = (250, 500, 750, 1000)
LIDAR_MILESTONES = (240, 500, 760, 1000)  # saved LiDAR checkpoints (every 10 iterations) closest to the milestones
OUTCOMES = ("goal_reached", "crash_vehicle", "out_of_road", "crash_other", "timeout")
JUNCTION_X = 50.0  # ego enters the junction area at x = 50 m
BRAKE_MPS = 2.0  # "meaningful braking": speed fell more than 2 m/s below its running peak at some point
IMPACT_STEPS = 2  # steps excluded at the end of collision / off-road episodes (impact, not braking)


def run_checkpoints(exp_dir):
    """{timesteps: checkpoint_dir} for every checkpoint under a Tune experiment directory (all trials)."""
    out = {}
    for trial in glob.glob(os.path.join(exp_dir, "PPO_*")):
        with open(os.path.join(trial, "progress.csv")) as f:
            steps = {int(r["training_iteration"]): int(r["timesteps_total"]) for r in csv.DictReader(f)}
        for ckpt in glob.glob(os.path.join(trial, "checkpoint_*")):
            out[steps[int(ckpt.rsplit("_", 1)[1])]] = ckpt
    return out


def fs_checkpoint(seed, k):
    for trial in glob.glob(os.path.join(HERE, "runs", "full_state_ppo", "PPO_*")):
        with open(os.path.join(trial, "params.json")) as f:
            if json.load(f)["seed"] != seed:
                continue
        with open(os.path.join(trial, "progress.csv")) as f:
            steps = {int(r["training_iteration"]): int(r["timesteps_total"]) for r in csv.DictReader(f)}
        for ckpt in glob.glob(os.path.join(trial, "checkpoint_*")):
            ts = steps[int(ckpt.rsplit("_", 1)[1])]
            if k * 1000 <= ts < k * 1000 + 4000:  # first iteration at or after the milestone
                return ckpt
    raise FileNotFoundError(f"full-state seed {seed} checkpoint {k}k")


def lidar_checkpoint(k):
    return run_checkpoints(os.path.join(ROOT, "static_goal_traffic", "runs", "static_goal_traffic_ppo"))[k * 1000]


def parse_group(name):
    """-> dict(kind, checkpoint, explore, traffic)."""
    traffic = not name.endswith("_off")
    base = name[:-4] if not traffic else name
    if base == "script":
        return dict(kind="script", checkpoint=None, explore=False, traffic=traffic)
    if base == "original_det":
        return dict(kind="lidar", checkpoint=ORIGINAL_CKPT, explore=False, traffic=traffic)
    parts = base.split("_")
    explore = parts[-1] == "stoch"
    if parts[0] == "fs":
        seed, k = int(parts[1][1:]), int(parts[2][:-1])
        return dict(kind="fs", checkpoint=fs_checkpoint(seed, k), explore=explore, traffic=traffic, seed=seed, k=k)
    if parts[0] == "lidar":
        k = int(parts[1][:-1])
        return dict(kind="lidar", checkpoint=lidar_checkpoint(k), explore=explore, traffic=traffic, k=k)
    raise ValueError(name)


def validation_groups():
    groups = [f"fs_s{s}_{k}k_{m}" for s in (0, 1, 2) for k in FS_MILESTONES for m in ("det", "stoch")]
    groups += [f"lidar_{k}k_{m}" for k in LIDAR_MILESTONES for m in ("det", "stoch")]
    return groups + ["original_det", "script"]


def noise_seed(seed, episode):
    return seed * 1000 + episode


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def run_episode(env, act, lidar):
    env.want_lidar = lidar
    obs, info = env.reset()
    dt = env.config["physics_world_step_size"] * env.config["decision_repeat"]
    n_vehicles, traffic_seed, min_dist = info["num_traffic_vehicles"], info["traffic_seed"], info["min_dist_to_traffic"]
    total, traj = 0.0, []
    while True:
        action = np.clip(act(info["lidar_obs"] if lidar else obs), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        # the ego must be driven by the evaluated policy's action, not by another controller
        assert info["policy"] == "EnvInputPolicy" and np.allclose(info["raw_action"], action, atol=1e-6)
        total += reward
        min_dist = min(min_dist, info["min_dist_to_traffic"])
        v = env.agent
        traj.append((float(v.position[0]), float(v.position[1]), float(v.speed), float(action[0]), float(action[1])))
        if terminated or truncated:
            break
    out = episode_outcome(info)
    t = np.array(traj)
    # Speed drops caused by the impact itself must not count as braking: for collision / off-road endings the
    # last IMPACT_STEPS steps are excluded from the slowdown measures.
    ended_by_impact = not (out["goal_reached"] or out["timeout"])
    tb = t[:-IMPACT_STEPS] if ended_by_impact and len(t) > IMPACT_STEPS else t
    speeds = tb[:, 2]
    before = tb[:, 0] < JUNCTION_X
    brake = t[:, 4] < 0  # the policy commanded braking (throttle/brake action below zero)
    entry = int(np.argmax(~before)) if (~before).any() else None
    summary = dict(
        out,
        reward=float(total),
        steps=len(traj),
        time_to_goal=round(len(traj) * dt, 2) if out["goal_reached"] else None,
        traffic_seed=traffic_seed,
        num_traffic_vehicles=n_vehicles,
        min_dist_to_traffic=None if np.isinf(min_dist) else round(float(min_dist), 2),
        max_slowdown=round(max_slowdown(speeds), 2),
        braked=bool(max_slowdown(speeds) > BRAKE_MPS),
        brake_command_steps=int(brake.sum()),
        brake_commanded=bool(longest_run(brake) >= 3),
        brake_action_share=round(float(np.mean(brake)), 3),
        approach_slowdown=round(max_slowdown(speeds[before]), 2) if before.any() else None,
        junction_entry_time=round((entry + 1) * dt, 2) if entry is not None else None,
        junction_entry_speed=round(float(speeds[entry]), 2) if entry is not None else None,
        mean_speed=round(float(t[:, 2].mean()), 2),
    )
    return summary, t


def evaluate(env, name, seeds):
    import torch
    from ray.rllib.policy.policy import Policy
    g = parse_group(name)
    if g["kind"] == "script":
        act = lambda obs: SCRIPT_ACTION  # noqa: E731
    else:
        policy = Policy.from_checkpoint(os.path.join(g["checkpoint"], "policies", "default_policy"))
        act = lambda obs: policy.compute_single_action(obs, explore=g["explore"])[0]  # noqa: E731
    env.config["traffic_density"] = env.default_config()["traffic_density"] if g["traffic"] else 0.0
    rows, trajs = [], []
    for seed in seeds:
        env.seed_spawns(seed)
        env.seed_traffic(seed)
        for ep in range(EPISODES_PER_SEED):
            # Sampling noise seeded per episode: episode k always gets the same noise sequence, so traffic-on and
            # traffic-off runs of a stochastic policy differ only through what the policy observes.
            torch.manual_seed(noise_seed(seed, ep))
            s, t = run_episode(env, act, lidar=g["kind"] == "lidar")
            rows.append(dict(seed=seed, episode=ep, **s))
            trajs.append(t)
    env.config["traffic_density"] = env.default_config()["traffic_density"]
    return g, rows, trajs


def summarize(rows):
    def mean(key, only=None):
        vals = [r[key] for r in rows if r[key] is not None and (only is None or only(r))]
        return float(np.mean(vals)) if vals else None

    return dict(
        episodes=len(rows),
        **{k: int(sum(r[k] for r in rows)) for k in OUTCOMES},
        success_rate=mean("goal_reached"),
        vehicle_collision_rate=mean("crash_vehicle"),
        mean_reward=mean("reward"),
        mean_time_to_goal=mean("time_to_goal"),
        mean_min_dist_to_traffic=mean("min_dist_to_traffic"),
        braked_episodes=int(sum(r["braked"] for r in rows)),
        brake_commanded_episodes=int(sum(r["brake_commanded"] for r in rows)),
        mean_brake_action_share=mean("brake_action_share"),
        mean_junction_entry_speed=mean("junction_entry_speed"),
        mean_junction_entry_time=mean("junction_entry_time"),
        mean_traffic_vehicles=mean("num_traffic_vehicles"),
    )


def cmd_run(set_name, groups):
    out_dir = os.path.join(RESULTS, set_name)
    os.makedirs(os.path.join(out_dir, "traj"), exist_ok=True)
    env = StaticGoalTrafficFullStateEnv(dict(extra_lidar_obs=True))
    env.want_lidar = False
    try:
        for name in groups:
            g, rows, trajs = evaluate(env, name, SETS[set_name])
            s = summarize(rows)
            with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
                json.dump(dict(group=name, set=set_name, seeds=SETS[set_name], **{k: v for k, v in g.items()},
                               summary=s, episodes=rows), f, indent=2)
            np.savez_compressed(os.path.join(out_dir, "traj", f"{name}.npz"), *trajs)
            print(f"{set_name:10s} {name:22s} success {s['goal_reached']:3d}  vehicle {s['crash_vehicle']:3d}  "
                  f"off-road {s['out_of_road']:3d}  timeout {s['timeout']:3d}  reward {s['mean_reward']:6.1f}  "
                  f"braked {s['braked_episodes']:3d}  brake-cmd {s['brake_commanded_episodes']:3d}", flush=True)
    finally:
        env.close()


def load(set_name, name):
    with open(os.path.join(RESULTS, set_name, f"{name}.json")) as f:
        return json.load(f)


def rank_key(s):
    """Selection rule (validation only): most successes, then fewest vehicle collisions, then highest reward."""
    return (s["goal_reached"], -s["crash_vehicle"], s["mean_reward"])


def cmd_select():
    val = {n: load("validation", n)["summary"] for n in validation_groups()}
    fs = [n for n in val if n.startswith("fs_")]
    sel = dict(
        rule="validation set only (seeds 100, 101): most successes, then fewest vehicle collisions, then reward",
        full_state=max(fs, key=lambda n: rank_key(val[n])),
        full_state_per_seed={s: max([n for n in fs if n.startswith(f"fs_s{s}_")], key=lambda n: rank_key(val[n]))
                             for s in (0, 1, 2)},
        lidar=max([n for n in val if n.startswith("lidar_")], key=lambda n: rank_key(val[n])),
    )
    with open(os.path.join(RESULTS, "selection.json"), "w") as f:
        json.dump(sel, f, indent=2)
    print(json.dumps(sel, indent=2))
    return sel


def test_groups(sel):
    groups = ["original_det", sel["lidar"], sel["full_state"], "script", sel["full_state"] + "_off", sel["lidar"] + "_off"]
    groups += [g for g in sel["full_state_per_seed"].values() if g not in groups]
    return groups


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired booleans. Returns (only_a, only_b, p)."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return only_a, only_b, round(p, 4)


def interaction(on_name, off_name):
    """Same policy, same ego starts, traffic on vs off (test set)."""
    on, off = load("test", on_name)["episodes"], load("test", off_name)["episodes"]
    t_on = list(np.load(os.path.join(RESULTS, "test", "traj", f"{on_name}.npz")).values())
    t_off = list(np.load(os.path.join(RESULTS, "test", "traj", f"{off_name}.npz")).values())
    per = []
    for a, b, ra, rb in zip(t_on, t_off, on, off):
        n = min(len(a), len(b))
        pre = (a[:n, 0] < JUNCTION_X) & (b[:n, 0] < JUNCTION_X)
        per.append(dict(
            max_position_difference=float(np.max(np.hypot(a[:n, 0] - b[:n, 0], a[:n, 1] - b[:n, 1]))),
            mean_abs_throttle_difference=float(np.mean(np.abs(a[:n, 4] - b[:n, 4]))),
            mean_abs_steering_difference=float(np.mean(np.abs(a[:n, 3] - b[:n, 3]))),
            throttle_lower_before_junction=float(np.mean(a[:n][pre, 4] - b[:n][pre, 4])) if pre.any() else None,
            braked_on=ra["braked"], braked_off=rb["braked"],
            brake_cmd_on=ra["brake_commanded"], brake_cmd_off=rb["brake_commanded"],
            approach_slowdown_on=ra["approach_slowdown"], approach_slowdown_off=rb["approach_slowdown"],
            entry_time_on=ra["junction_entry_time"], entry_time_off=rb["junction_entry_time"],
            entry_speed_on=ra["junction_entry_speed"], entry_speed_off=rb["junction_entry_speed"],
        ))

    def paired(key_on, key_off):
        d = [p[key_on] - p[key_off] for p in per if p[key_on] is not None and p[key_off] is not None]
        return dict(mean_difference=float(np.mean(d)), episodes=len(d))

    slowed = [p for p in per if p["approach_slowdown_on"] is not None and p["approach_slowdown_off"] is not None]
    return dict(
        policy=on_name,
        trajectory_changed_over_1m=int(sum(p["max_position_difference"] > 1.0 for p in per)),
        median_max_position_difference_m=float(np.median([p["max_position_difference"] for p in per])),
        mean_abs_throttle_difference=float(np.mean([p["mean_abs_throttle_difference"] for p in per])),
        mean_abs_steering_difference=float(np.mean([p["mean_abs_steering_difference"] for p in per])),
        mean_throttle_change_before_junction=float(np.mean([p["throttle_lower_before_junction"] for p in per
                                                            if p["throttle_lower_before_junction"] is not None])),
        braked_episodes_on=int(sum(p["braked_on"] for p in per)),
        braked_episodes_off=int(sum(p["braked_off"] for p in per)),
        brake_commanded_episodes_on=int(sum(p["brake_cmd_on"] for p in per)),
        brake_commanded_episodes_off=int(sum(p["brake_cmd_off"] for p in per)),
        slowed_before_junction_on=int(sum(p["approach_slowdown_on"] > BRAKE_MPS for p in slowed)),
        slowed_before_junction_off=int(sum(p["approach_slowdown_off"] > BRAKE_MPS for p in slowed)),
        junction_entry_time_change_s=paired("entry_time_on", "entry_time_off"),
        junction_entry_speed_change_mps=paired("entry_speed_on", "entry_speed_off"),
        vehicle_collisions_on=int(sum(r["crash_vehicle"] for r in on)),
        vehicle_collisions_off=int(sum(r["crash_vehicle"] for r in off)),
        success_on=int(sum(r["goal_reached"] for r in on)),
        success_off=int(sum(r["goal_reached"] for r in off)),
    )


def cmd_report():
    with open(os.path.join(RESULTS, "selection.json")) as f:
        sel = json.load(f)
    # Per-group summary CSV for both sets
    fields = ["set", "group", "episodes", *OUTCOMES, "success_rate", "vehicle_collision_rate", "mean_reward",
              "mean_time_to_goal", "mean_min_dist_to_traffic", "braked_episodes", "brake_commanded_episodes", "mean_brake_action_share",
              "mean_junction_entry_speed", "mean_junction_entry_time", "mean_traffic_vehicles", "checkpoint"]
    rows = []
    for set_name in ("validation", "test"):
        for path in sorted(glob.glob(os.path.join(RESULTS, set_name, "*.json"))):
            d = json.load(open(path))
            rows.append(dict(set=set_name, group=d["group"], checkpoint=d.get("checkpoint"), **d["summary"]))
    with open(os.path.join(RESULTS, "summary_by_group.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # Per-episode CSV for the test set
    ep_fields = ["group", "seed", "episode", *OUTCOMES, "reward", "steps", "time_to_goal", "traffic_seed",
                 "num_traffic_vehicles", "min_dist_to_traffic", "max_slowdown", "braked", "brake_commanded",
                 "brake_command_steps", "brake_action_share",
                 "approach_slowdown", "junction_entry_time", "junction_entry_speed", "mean_speed"]
    with open(os.path.join(RESULTS, "test_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ep_fields, extrasaction="ignore")
        w.writeheader()
        for g in test_groups(sel):
            for r in load("test", g)["episodes"]:
                w.writerow(dict(group=g, **r))
    # Paired tests on the test set
    fs, li = sel["full_state"], sel["lidar"]
    test = {g: load("test", g)["episodes"] for g in test_groups(sel)}
    succ = {g: [r["goal_reached"] for r in e] for g, e in test.items()}
    coll = {g: [r["crash_vehicle"] for r in e] for g, e in test.items()}
    pairs = [(fs, li), (fs, "script"), (fs, "original_det"), (li, "script"), (li, "original_det")]
    paired = [dict(a=a, b=b, success=dict(zip(("only_a", "only_b", "p"), mcnemar(succ[a], succ[b]))),
                   vehicle_collision=dict(zip(("only_a", "only_b", "p"), mcnemar(coll[a], coll[b]))))
              for a, b in pairs]
    summary = dict(
        selection=sel,
        test={g: load("test", g)["summary"] for g in test_groups(sel)},
        paired_tests_test_set=paired,
        interaction_test_set=[interaction(fs, fs + "_off"), interaction(li, li + "_off")],
        validation={n: load("validation", n)["summary"] for n in validation_groups()},
    )
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("selection", "paired_tests_test_set", "interaction_test_set")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["grid", "run", "select", "test", "report"])
    parser.add_argument("--set", choices=list(SETS), default="validation")
    parser.add_argument("--groups", type=str, default=None)
    args = parser.parse_args()
    if args.command == "grid":
        print(",".join(validation_groups()))
    elif args.command == "run":
        cmd_run(args.set, args.groups.split(","))
    elif args.command == "select":
        cmd_select()
    elif args.command == "test":
        with open(os.path.join(RESULTS, "selection.json")) as f:
            groups = test_groups(json.load(f))
        cmd_run("test", groups if args.groups is None else args.groups.split(","))
    elif args.command == "report":
        cmd_report()
