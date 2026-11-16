"""Recordings of held-out test episodes for the full-state experiment (MetaDrive top-down renderer).

A clip replays test episode (seed, k) exactly as evaluate_full_state.py ran it: same ego-start and traffic seeds and
the same per-episode sampling-noise seed. Each replayed outcome is checked against the evaluation file.

    python static_goal_traffic_full_state/record_full_state.py
Writes media/*.gif, media/*.mp4, stills and media/recordings_meta.json. Clips are chosen from the test results:
  success.gif             selected full-state policy succeeds
  failure.gif             selected full-state policy, its most common failure
  lidar_vs_full_state.gif same episode: selected LiDAR policy (left) vs selected full-state policy (right)
  traffic_on_vs_off.gif   same start: full-state policy with traffic (left) and without (right)
"""
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "static_goal_traffic"))
sys.path.insert(0, os.path.join(ROOT, "meeting"))
from evaluate_full_state import RESULTS, load, noise_seed, parse_group, EPISODES_PER_SEED  # noqa: E402
from evaluate_traffic import SCRIPT_ACTION, episode_outcome  # noqa: E402
from record_comparison import map_frame, to_screen  # noqa: E402
from record_traffic import FOOT, GREEN, HEAD, MAP_H, MAP_W, STATUS_COLOR, font  # noqa: E402
from record_traffic import pad, save_gif, side_by_side, status_label  # noqa: E402
from PIL import ImageDraw  # noqa: E402


def panel(img, region_px, rec, title, subtitle):
    """Same layout as static_goal_traffic/record_traffic.panel; shows 'none' when no traffic vehicle exists."""
    im = Image.new("RGB", (MAP_W, HEAD + MAP_H + FOOT), "white")
    im.paste(Image.fromarray(img), (0, HEAD))
    d = ImageDraw.Draw(im)
    d.text((20, 12), title, fill=(20, 20, 20), font=font(26, True))
    d.text((MAP_W - 20, 20), subtitle, fill=(90, 90, 90), font=font(17), anchor="ra")
    x1 = max(p[0] for p in region_px)
    y1 = max(p[1] for p in region_px) + HEAD
    d.text((x1, y1 + 6), "Destination region", fill=GREEN, font=font(17, True), anchor="ra")
    y = HEAD + MAP_H + 14
    closest = "none" if rec["closest"] is None else f"{rec['closest']:5.1f} m"
    d.text((20, y), f"Time {rec['t']:4.1f} s     Speed {rec['speed']:4.1f} m/s", fill=(20, 20, 20), font=font(24))
    d.text((20, y + 40), f"To destination {rec['dist']:5.1f} m     Closest vehicle {closest}", fill=(20, 20, 20),
           font=font(24))
    d.text((MAP_W - 20, y + 20), rec["status"], fill=STATUS_COLOR[rec["status"]], font=font(30, True), anchor="ra")
    return np.asarray(im)
from full_state_env import StaticGoalTrafficFullStateEnv  # noqa: E402

OUT = os.path.join(HERE, "media")


def title_for(name):
    g = parse_group(name)
    mode = "sampled" if g["explore"] else "mean"
    traffic = "" if g["traffic"] else ", NO TRAFFIC"
    if name.startswith("fs_"):
        return f"Full state (s{g['seed']}, {g['k']}k, {mode}{traffic})"
    if name.startswith("lidar_"):
        return f"LiDAR ({g['k']}k, {mode}{traffic})"
    return name


def make_actor(name):
    import torch  # noqa: F401
    from ray.rllib.policy.policy import Policy
    g = parse_group(name)
    if g["kind"] == "script":
        return g, (lambda obs: SCRIPT_ACTION)
    policy = Policy.from_checkpoint(os.path.join(g["checkpoint"], "policies", "default_policy"))
    return g, (lambda obs: policy.compute_single_action(obs, explore=g["explore"])[0])


def drive(env, g, act, render, title=None):
    """One episode; if render, returns frames. Mirrors evaluate_full_state.run_episode."""
    lidar = g["kind"] == "lidar"
    env.want_lidar = lidar
    obs, info = env.reset()
    region = Polygon(env.goal_region())
    dt = env.config["physics_world_step_size"] * env.config["decision_repeat"]
    frames, t = [], 0
    subtitle = f"traffic seed {info['traffic_seed']}  {info['num_traffic_vehicles']} vehicles"

    def snap(status, closest):
        v = env.agent
        rec = dict(t=dt * t, speed=float(v.speed), closest=None if np.isinf(closest) else closest, status=status,
                   dist=float(region.distance(Point(float(v.position[0]), float(v.position[1])))))
        frames.append(panel(map_frame(env), [to_screen(env, x, y) for x, y in env.goal_region()], rec, title, subtitle))

    if render:
        snap("DRIVING", info["min_dist_to_traffic"])
    while True:
        action = np.clip(act(info["lidar_obs"] if lidar else obs), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["policy"] == "EnvInputPolicy" and np.allclose(info["raw_action"], action, atol=1e-6)
        t += 1
        done = terminated or truncated
        if render:
            snap(status_label(episode_outcome(info)) if done else "DRIVING", info["min_dist_to_traffic"])
        if done:
            return status_label(episode_outcome(info)), frames, info


def record(env, name, seed, k):
    import torch
    g, act = make_actor(name)
    env.config["traffic_density"] = env.default_config()["traffic_density"] if g["traffic"] else 0.0
    env.seed_spawns(seed)
    env.seed_traffic(seed)
    for _ in range(k):  # advance the ego-start and traffic seed sequences to episode k
        env.reset()
    torch.manual_seed(noise_seed(seed, k))  # same sampling noise as in the evaluation
    status, frames, info = drive(env, g, act, render=True, title=title_for(name))
    env.config["traffic_density"] = env.default_config()["traffic_density"]
    expected = status_label(load("test", name)["episodes"][[0, 1].index(seed) * EPISODES_PER_SEED + k])
    assert status == expected, (name, seed, k, status, expected)
    return frames, dict(group=name, seed=seed, episode=k, status=status, seconds=round(0.1 * (len(frames) - 1), 1),
                        traffic_seed=info["traffic_seed"])


def pick(sel):
    fs, li = sel["full_state"], sel["lidar"]
    e_fs, e_li = load("test", fs)["episodes"], load("test", li)["episodes"]
    e_off = load("test", fs + "_off")["episodes"]
    idx = range(len(e_fs))
    key = lambda i: (e_fs[i]["seed"], e_fs[i]["episode"])  # noqa: E731
    contrast = [i for i in idx if e_fs[i]["goal_reached"] and not e_li[i]["goal_reached"]]
    success = contrast[0] if contrast else next(i for i in idx if e_fs[i]["goal_reached"])
    fails = [i for i in idx if not e_fs[i]["goal_reached"]]
    common = max(("crash_vehicle", "out_of_road", "timeout"), key=lambda o: sum(e_fs[i][o] for i in fails)) if fails else None
    failure = next(i for i in fails if e_fs[i][common]) if fails else None
    # traffic on vs off: the successful episode where traffic delays junction entry the most
    delays = [(e_fs[i]["junction_entry_time"] - e_off[i]["junction_entry_time"], i) for i in idx
              if e_fs[i]["goal_reached"] and e_fs[i]["junction_entry_time"] and e_off[i]["junction_entry_time"]]
    onoff = max(delays)[1] if delays else success
    return key(success), (key(failure) if failure is not None else None), common, key(onoff), bool(contrast)


if __name__ == "__main__":
    with open(os.path.join(RESULTS, "selection.json")) as f:
        sel = json.load(f)
    fs, li = sel["full_state"], sel["lidar"]
    (ss, sk), failure, common, (os_, ok), is_contrast = pick(sel)
    os.makedirs(OUT, exist_ok=True)
    env = StaticGoalTrafficFullStateEnv(dict(extra_lidar_obs=True))
    meta = dict(selection=sel, clips={})
    try:
        clips = {"success": (fs, ss, sk), "lidar_same_episode": (li, ss, sk), "onoff_on": (fs, os_, ok),
                 "onoff_off": (fs + "_off", os_, ok)}
        if failure:
            clips["failure"] = (fs, *failure)
        frames = {}
        for name, (group, seed, k) in clips.items():
            frames[name], meta["clips"][name] = record(env, group, seed, k)
            print(name, meta["clips"][name], flush=True)
    finally:
        env.close()
    save_gif(os.path.join(OUT, "success.gif"), frames["success"])
    Image.fromarray(frames["success"][-1]).save(os.path.join(OUT, "still_success_end.png"))
    if "failure" in frames:
        save_gif(os.path.join(OUT, "failure.gif"), frames["failure"])
        Image.fromarray(frames["failure"][-1]).save(os.path.join(OUT, "still_failure_end.png"))
    s = meta["clips"]["success"]
    banner = (f"Same test episode (seed {ss}, episode {sk}, traffic seed {s['traffic_seed']}): "
              f"LiDAR policy (left) vs full-state policy (right)" + ("; picked for contrast" if is_contrast else ""))
    sbs = side_by_side(frames["lidar_same_episode"], frames["success"], banner)
    save_gif(os.path.join(OUT, "lidar_vs_full_state.gif"), sbs)
    imageio.mimsave(os.path.join(OUT, "lidar_vs_full_state.mp4"), pad(sbs, len(sbs) + 15), fps=10, macro_block_size=1)
    Image.fromarray(sbs[-1]).save(os.path.join(OUT, "still_lidar_vs_full_state_end.png"))
    o = meta["clips"]["onoff_on"]
    sbs2 = side_by_side(frames["onoff_on"], frames["onoff_off"],
                        f"Same start (seed {os_}, episode {ok}): full-state policy with traffic (left) and with traffic "
                        "switched off (right)")
    save_gif(os.path.join(OUT, "traffic_on_vs_off.gif"), sbs2)
    imageio.mimsave(os.path.join(OUT, "traffic_on_vs_off.mp4"), pad(sbs2, len(sbs2) + 15), fps=10, macro_block_size=1)
    Image.fromarray(sbs2[-1]).save(os.path.join(OUT, "still_traffic_on_vs_off_end.png"))
    meta["failure_type"] = common
    meta["contrast_episode"] = is_contrast
    with open(os.path.join(OUT, "recordings_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved recordings to", OUT)
