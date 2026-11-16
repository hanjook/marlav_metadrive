"""Record evaluation episodes of the traffic task with MetaDrive's top-down renderer.

Each clip replays episode k of a deterministic evaluation group exactly (same ego start and traffic seeds as
evaluate_traffic.py) and checks that the replayed outcome matches the evaluation file. Drawn on top of the
renderer's frame: the destination region and a text panel (time, speed, distance to the region, closest vehicle,
status). Background vehicles are drawn by the renderer itself.

    python static_goal_traffic/record_traffic.py            # picks clips from results/eval_*_seed0.json
Writes media/success.gif, media/failure.gif, media/side_by_side.gif/.mp4, stills and media/recordings_meta.json.
"""
import argparse
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "meeting"))
from evaluate_traffic import RESULTS, SCRIPT_ACTION, TRAFFIC_CKPT, ORIGINAL_CKPT, episode_outcome  # noqa: E402
from record_comparison import GREEN, HEAD, MAP_H, MAP_W, font, map_frame, to_screen  # noqa: E402
from evaluate import load_policy  # noqa: E402
from traffic_env import StaticGoalTrafficEnv  # noqa: E402

OUT = os.path.join(HERE, "media")
FOOT = 104
STATUS_COLOR = {
    "DRIVING": (90, 90, 90),
    "SUCCESS": (20, 130, 50),
    "OFF ROAD": (200, 60, 30),
    "VEHICLE COLLISION": (200, 30, 30),
    "TIMEOUT": (180, 120, 0)
}
GROUP_TITLE = {
    "traffic_det": "Traffic-trained policy",
    "original_det": "No-traffic policy (original)",
    "script": "Constant-throttle script",
}


def status_label(out):
    for key, label in (("goal_reached", "SUCCESS"), ("crash_vehicle", "VEHICLE COLLISION"), ("out_of_road", "OFF ROAD")):
        if out[key]:
            return label
    return "TIMEOUT"


def panel(img, region_px, rec, title, subtitle):
    im = Image.new("RGB", (MAP_W, HEAD + MAP_H + FOOT), "white")
    im.paste(Image.fromarray(img), (0, HEAD))
    d = ImageDraw.Draw(im)
    d.text((20, 12), title, fill=(20, 20, 20), font=font(28, True))
    d.text((MAP_W - 20, 20), subtitle, fill=(90, 90, 90), font=font(18), anchor="ra")
    x1 = max(p[0] for p in region_px)
    y1 = max(p[1] for p in region_px) + HEAD
    d.text((x1, y1 + 6), "Destination region", fill=GREEN, font=font(17, True), anchor="ra")
    y = HEAD + MAP_H + 14
    d.text((20, y), f"Time {rec['t']:4.1f} s     Speed {rec['speed']:4.1f} m/s", fill=(20, 20, 20), font=font(24))
    d.text((20, y + 40), f"To destination {rec['dist']:5.1f} m     Closest vehicle {rec['closest']:5.1f} m",
           fill=(20, 20, 20), font=font(24))
    d.text((MAP_W - 20, y + 20), rec["status"], fill=STATUS_COLOR[rec["status"]], font=font(30, True), anchor="ra")
    return np.asarray(im)


def record(env, act, seed, episode, title):
    env.seed_spawns(seed)
    env.seed_traffic(seed)
    for _ in range(episode):  # advance both seed sequences so this is evaluation episode `episode`
        env.reset()
    obs, info = env.reset()
    region = Polygon(env.goal_region())
    dt = env.config["physics_world_step_size"] * env.config["decision_repeat"]
    subtitle = f"episode {episode}  traffic seed {info['traffic_seed']}  {info['num_traffic_vehicles']} vehicles"
    frames, recs = [], []

    def snapshot(status, closest):
        v = env.agent
        rec = dict(t=dt * len(recs), speed=float(v.speed), closest=closest, status=status,
                   dist=float(region.distance(Point(float(v.position[0]), float(v.position[1])))))
        recs.append(rec)
        img = map_frame(env)
        frames.append(panel(img, [to_screen(env, x, y) for x, y in env.goal_region()], rec, title, subtitle))

    snapshot("DRIVING", info["min_dist_to_traffic"])
    while True:
        action = np.clip(act(obs), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["policy"] == "EnvInputPolicy" and np.allclose(info["raw_action"], action, atol=1e-6)
        done = terminated or truncated
        snapshot(status_label(episode_outcome(info)) if done else "DRIVING", info["min_dist_to_traffic"])
        if done:
            break
    return frames, dict(status=recs[-1]["status"], seconds=round(recs[-1]["t"], 1), traffic_seed=info["traffic_seed"],
                        num_traffic_vehicles=info["num_traffic_vehicles"])


def save_gif(path, frames, fps=10, hold=1.5):
    frames = list(frames) + [frames[-1]] * int(fps * hold)
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)  # duration is ms per frame


def pad(frames, n):
    return list(frames) + [frames[-1]] * (n - len(frames))


def side_by_side(left, right, banner):
    n = max(len(left), len(right))
    left, right = pad(left, n), pad(right, n)
    bh = 50
    out = []
    for a, b in zip(left, right):
        im = Image.new("RGB", (a.shape[1] + b.shape[1] + 12, a.shape[0] + bh), "white")
        im.paste(Image.fromarray(a), (0, bh))
        im.paste(Image.fromarray(b), (a.shape[1] + 12, bh))
        ImageDraw.Draw(im).text((20, 12), banner, fill=(20, 20, 20), font=font(22, True))
        out.append(np.asarray(im))
    return out


def load_eval(group, seed):
    with open(os.path.join(RESULTS, f"eval_{group}_seed{seed}.json")) as f:
        return json.load(f)["episodes"]


def pick(seed):
    """Success clip: first episode where traffic_det succeeds and the original policy fails (falls back to the first
    traffic_det success), so the side-by-side shows a contrast; it is an illustrative example, not a typical one.
    Failure clip: first traffic_det failure of its most common failure type; if traffic_det never fails, the
    original policy's first failure."""
    det = load_eval("traffic_det", seed)
    orig = load_eval("original_det", seed)
    keys = ("crash_vehicle", "out_of_road", "timeout", "crash_other")
    contrast = [d["episode"] for d, o in zip(det, orig) if d["goal_reached"] and not o["goal_reached"]]
    success = contrast[0] if contrast else next(e["episode"] for e in det if e["goal_reached"])
    failures = [e for e in det if not e["goal_reached"]]
    if failures:
        common = max(keys, key=lambda k: sum(e[k] for e in failures))
        return ("traffic_det", success), ("traffic_det", next(e["episode"] for e in failures if e[common]))
    return ("traffic_det", success), ("original_det", next(e["episode"] for e in orig if not e["goal_reached"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    (sg, se), (fg, fe) = pick(args.seed)
    clips = [("success", sg, se), ("failure", fg, fe)]
    if fg == "traffic_det":  # also show the original policy on the failure episode for comparison
        clips.append(("original_on_failure_episode", "original_det", fe))
    # the original policy on the success episode, for a same-situation side-by-side
    clips.append(("original_on_success_episode", "original_det", se))

    env = StaticGoalTrafficEnv()
    policies = {"traffic_det": load_policy(TRAFFIC_CKPT), "original_det": load_policy(ORIGINAL_CKPT)}
    actors = {g: (lambda p: (lambda obs: p.compute_single_action(obs, explore=False)[0]))(p) for g, p in policies.items()}
    actors["script"] = lambda obs: SCRIPT_ACTION
    meta, frames = dict(seed=args.seed, clips={}), {}
    try:
        for name, group, ep in clips:
            frames[name], m = record(env, actors[group], args.seed, ep, GROUP_TITLE[group])
            expected = status_label(load_eval(group, args.seed)[ep])
            assert m["status"] == expected, (name, m["status"], expected)  # replay matches the evaluation file
            meta["clips"][name] = dict(group=group, episode=ep, **m)
            print(name, meta["clips"][name])
    finally:
        env.close()

    save_gif(os.path.join(OUT, "success.gif"), frames["success"])
    save_gif(os.path.join(OUT, "failure.gif"), frames["failure"])
    Image.fromarray(frames["success"][-1]).save(os.path.join(OUT, "still_success_end.png"))
    Image.fromarray(frames["failure"][-1]).save(os.path.join(OUT, "still_failure_end.png"))
    s = meta["clips"]["success"]
    sbs = side_by_side(frames["original_on_success_episode"], frames["success"],
                       f"Same start and same traffic (episode {se}, traffic seed {s['traffic_seed']}): "
                       "no-traffic policy (left) vs traffic-trained policy (right)")
    save_gif(os.path.join(OUT, "side_by_side.gif"), sbs)
    imageio.mimsave(os.path.join(OUT, "side_by_side.mp4"), pad(sbs, len(sbs) + 15), fps=10, macro_block_size=1)
    Image.fromarray(sbs[-1]).save(os.path.join(OUT, "still_side_by_side_end.png"))
    if "original_on_failure_episode" in frames:
        f = meta["clips"]["failure"]
        sbs2 = side_by_side(frames["original_on_failure_episode"], frames["failure"],
                            f"Same start and same traffic (episode {fe}, traffic seed {f['traffic_seed']}): "
                            "no-traffic policy (left) vs traffic-trained policy (right)")
        save_gif(os.path.join(OUT, "side_by_side_failure_episode.gif"), sbs2)
    with open(os.path.join(OUT, "recordings_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("Saved recordings to", OUT)
