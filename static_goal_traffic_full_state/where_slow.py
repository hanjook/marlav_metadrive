"""Where along the route does each policy drive slowly? Share of steps below 3 m/s by zone (test set, traffic on).

    python static_goal_traffic_full_state/where_slow.py
Writes results/where_slow.json.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(HERE, "results", "test")
ZONES = {"approach (x < 50 m)": (-1e9, 50.0), "junction (50-87.5 m)": (50.0, 87.5), "exit road (> 87.5 m)": (87.5, 1e9)}
SLOW = 3.0  # m/s; excludes the standing start (the first 10 steps)

if __name__ == "__main__":
    sel = json.load(open(os.path.join(HERE, "results", "selection.json")))
    out = {}
    for g in [sel["full_state"], sel["lidar"], "script", sel["full_state"] + "_off", sel["lidar"] + "_off"]:
        trajs = list(np.load(os.path.join(TEST, "traj", f"{g}.npz")).values())
        steps = np.concatenate([t[10:] for t in trajs])
        out[g] = {}
        for zone, (a, b) in ZONES.items():
            z = steps[(steps[:, 0] >= a) & (steps[:, 0] < b)]
            out[g][zone] = dict(slow_steps=int((z[:, 2] < SLOW).sum()), steps=int(len(z)),
                                episodes_with_slow_steps=int(sum(((t[10:, 0] >= a) & (t[10:, 0] < b) & (t[10:, 2] < SLOW)).any()
                                                                 for t in trajs)))
        print(g, {z: f"{v['slow_steps']}/{v['steps']} steps slow, {v['episodes_with_slow_steps']} episodes" for z, v in out[g].items()})
    with open(os.path.join(HERE, "results", "where_slow.json"), "w") as f:
        json.dump(out, f, indent=2)
