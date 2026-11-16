"""Render and check several seeded resets before training (no learning involved).

The ego drives the constant-throttle script (steering 0, throttle 0.5) so that every reset is driven the same way.
For each episode it checks: number of non-ego vehicles, how many actually move, how many drive through the junction,
how many are in the junction at the same time as the ego, closest approach to the ego, that the ego is driven by
EnvInputPolicy, and that the destination and success region equal the no-traffic environment's.

    python static_goal_traffic/inspect_traffic.py            # 6 episodes, traffic seed 0
Writes media/inspect_resets.png (frames at 0 s, 3 s, 6 s per episode) and results/inspect_traffic.json.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "meeting"))
from record_comparison import font, map_frame  # noqa: E402
from static_goal_env import TARGET_NODE  # noqa: E402
from traffic_env import StaticGoalTrafficEnv  # noqa: E402

JUNCTION_X, JUNCTION_Y = (50.0, 87.5), (-10.0, 27.5)  # map "X" junction area (from the road graph)
SNAP_STEPS = (0, 30, 60)
SCRIPT_ACTION = [0.0, 0.5]


def in_junction(p):
    return JUNCTION_X[0] <= p[0] <= JUNCTION_X[1] and JUNCTION_Y[0] <= p[1] <= JUNCTION_Y[1]


REFERENCE_REGION = ((117.5, 127.5), (-1.75, 8.75))  # x, y bounds of the no-traffic baseline's success region


def region_unchanged(region):
    (x0, x1), (y0, y1) = REFERENCE_REGION
    return bool(np.allclose([region[:, 0].min(), region[:, 0].max(), region[:, 1].min(), region[:, 1].max()],
                            [x0, x1, y0, y1], atol=1e-3))


def outcome_label(info):
    if info["goal_reached"]:
        return "SUCCESS"
    if info["crash_vehicle"]:
        return "VEHICLE COLLISION"
    if info["out_of_road"]:
        return "OFF ROAD"
    return "TIMEOUT"


def inspect(episodes=6, spawn_seed=0, traffic_seed=0):
    env = StaticGoalTrafficEnv(dict(spawn_seed=spawn_seed, traffic_seed=traffic_seed))
    rows, sheet = [], []
    try:
        for ep in range(episodes):
            obs, info = env.reset()
            assert env.agent.navigation.checkpoints[-1] == TARGET_NODE
            region_same = region_unchanged(env.goal_region())
            first = {v.name: np.asarray(v.position) for v in env.traffic_vehicles()}
            moved, crossed, both_in_junction, frames, step = set(), set(), 0, [], 0
            min_dist = info["min_dist_to_traffic"]
            while True:
                if step in SNAP_STEPS:
                    frames.append(map_frame(env))
                obs, reward, terminated, truncated, info = env.step(SCRIPT_ACTION)
                assert info["policy"] == "EnvInputPolicy" and np.allclose(info["raw_action"], SCRIPT_ACTION)
                step += 1
                ego_in = in_junction(np.asarray(env.agent.position))
                any_in = False
                for v in env.traffic_vehicles():
                    p = np.asarray(v.position)
                    first.setdefault(v.name, p)
                    travelled = np.linalg.norm(p - first[v.name])
                    if travelled > 5.0:
                        moved.add(v.name)
                    if in_junction(p) and travelled > 1.0:
                        crossed.add(v.name)
                        any_in = True
                both_in_junction += int(ego_in and any_in)
                min_dist = min(min_dist, info["min_dist_to_traffic"])
                if terminated or truncated:
                    break
            while len(frames) < len(SNAP_STEPS):
                frames.append(map_frame(env))
            rows.append(
                dict(
                    episode=ep,
                    traffic_seed=info["traffic_seed"],
                    num_traffic_vehicles=len(first),
                    vehicles_moved=len(moved),
                    vehicles_through_junction=len(crossed),
                    steps_ego_and_traffic_in_junction=both_in_junction,
                    min_dist_to_traffic=round(float(min_dist), 2),
                    ego_policy=info["policy"],
                    goal_region_unchanged=region_same,
                    scripted_outcome=outcome_label(info),
                    steps=step,
                )
            )
            sheet.append((rows[-1], frames))
            print(rows[-1])
    finally:
        env.close()
    return rows, sheet


def contact_sheet(sheet, path, scale=0.5):
    h, w = sheet[0][1][0].shape[:2]
    tw, th, label_h, title_h = int(w * scale), int(h * scale), 46, 40
    im = Image.new("RGB", (tw * len(SNAP_STEPS), title_h + (th + label_h) * len(sheet)), "white")
    d = ImageDraw.Draw(im)
    d.text((10, 8), "Seeded resets with traffic (density 0.2, trigger mode). Columns: t = 0 s, 3 s, 6 s. "
           "Ego: constant-throttle script.", fill=(20, 20, 20), font=font(18, True))
    for r, (row, frames) in enumerate(sheet):
        y = title_h + r * (th + label_h)
        d.text((10, y + 6), f"Episode {row['episode']}  traffic seed {row['traffic_seed']}  "
               f"vehicles {row['num_traffic_vehicles']}  moved {row['vehicles_moved']}  "
               f"through junction {row['vehicles_through_junction']}  closest {row['min_dist_to_traffic']} m  "
               f"script: {row['scripted_outcome']}", fill=(40, 40, 40), font=font(15))
        for c, f in enumerate(frames):
            im.paste(Image.fromarray(f).resize((tw, th)), (c * tw, y + label_h - 6))
    im.save(path)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "media"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rows, sheet = inspect()
    contact_sheet(sheet, os.path.join(HERE, "media", "inspect_resets.png"))
    with open(os.path.join(HERE, "results", "inspect_traffic.json"), "w") as f:
        json.dump(dict(spawn_seed=0, traffic_seed=0, script_action=SCRIPT_ACTION, episodes=rows), f, indent=2)
    print("Saved media/inspect_resets.png and results/inspect_traffic.json")
