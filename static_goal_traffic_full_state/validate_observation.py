"""Validate the full-state observation before training, and write OBSERVATION_SCHEMA.md.

    python static_goal_traffic_full_state/validate_observation.py
Prints PASS/FAIL per check and writes results/validation.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from full_state_env import (EGO_FEATURES, NUM_SLOTS, SLOT_FEATURES, StaticGoalTrafficFullStateEnv, ego_frame,  # noqa
                            is_idm, POS_SCALE, REL_VEL_SCALE)

STATE_DIM = 19
EGO0 = STATE_DIM
SLOT0 = STATE_DIM + len(EGO_FEATURES)
SLOT_W = len(SLOT_FEATURES)
F = {n: i for i, (n, _) in enumerate(SLOT_FEATURES)}
STATE_OBS_ROWS = [
    "distance to left road edge / total road width (MetaDrive dist_to_left_side)",
    "distance to right road edge / total road width (MetaDrive dist_to_right_side)",
    "heading difference to current reference lane (MetaDrive heading_diff)",
    "(speed km/h + 1) / (max speed km/h + 1)",
    "(steering / MAX_STEERING + 1) / 2",
    "last throttle/brake action, (a + 1) / 2",
    "last steering action, (a + 1) / 2",
    "yaw rate, clipped to [0, 1]",
    "lateral position on current lane, (2 * lat / max lane width + 1) / 2",
] + [f"checkpoint {k}: {d}" for k in (1, 2) for d in (
    "forward offset (ego frame), normalised", "side offset (ego frame), normalised",
    "bend radius of the lane ending at the checkpoint", "bend direction (clockwise 1)", "bend angle")]


def slot(obs, k):
    return obs[SLOT0 + k * SLOT_W:SLOT0 + (k + 1) * SLOT_W]


class Checks:
    def __init__(self):
        self.results = {}

    def record(self, name, ok, detail):
        self.results[name] = dict(passed=bool(ok), detail=detail)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def drive(env, policy, rng, checks_state, max_steps=300):
    """One episode with a per-step audit. policy(obs) -> action."""
    obs, info = env.reset()
    fso = env.full_state_obs
    names0 = list(fso.slot_names)
    seen_absent = set()
    for t in range(max_steps):
        audit_step(env, obs, names0, seen_absent, checks_state)
        action = np.asarray(policy(obs, rng), dtype=float)
        obs, r, term, trunc, info = env.step(action)
        checks_state["policy_ok"] &= info["policy"] == "EnvInputPolicy" and np.allclose(info["raw_action"], action)
        checks_state["idm_ok"] &= all(is_idm(p) for p in env.traffic_policies().values())
        if term or trunc:
            break
    audit_step(env, obs, names0, seen_absent, checks_state)
    return info


def audit_step(env, obs, names0, seen_absent, s):
    space = env.observation_space
    s["steps"] += 1
    s["shape_ok"] &= obs.shape == space.shape and obs.dtype == space.dtype and space.contains(obs)
    s["finite_ok"] &= bool(np.all(np.isfinite(obs)))
    raw = env.full_state_obs.raw
    s["bounds_ok"] &= bool(np.all(obs >= -1.0) and np.all(obs <= 1.0))
    s["clipped"] += (np.abs(raw) > 1.0 + 1e-9)
    fso = env.full_state_obs
    s["slots_ok"] &= fso.slot_names == names0
    ego = env.agent
    objects = env.engine.get_objects()
    for k, name in enumerate(names0):
        sv = slot(obs, k)
        present = name is not None and name in objects
        if name is not None and not present:
            seen_absent.add(k)
        if k in seen_absent and present:
            s["slots_ok"] = False  # a slot must not come back once its vehicle is gone
        if not present:
            s["slots_ok"] &= bool(np.all(sv == 0.0))
            continue
        v = objects[name]
        rel = ego_frame(np.asarray(v.position) - np.asarray(ego.position), ego.heading_theta) / POS_SCALE
        s["slots_ok"] &= sv[F["present"]] == 1.0 and np.allclose(sv[[F["rel_forward"], F["rel_left"]]],
                                                                    np.clip(rel, -1, 1), atol=1e-4)
    s["max_present"] = max(s["max_present"], sum(n is not None and n in objects for n in names0))
    s["traffic_moved"] |= any(objects[n].speed > 1.0 for n in names0 if n is not None and n in objects)


def throttle_policy(obs, rng):
    return [float(rng.uniform(-0.15, 0.15)), 0.5]


def random_policy(obs, rng):
    return list(rng.uniform(-1, 1, size=2))


def write_schema(env, path, clip_rates):
    n = env.observation_space.shape[0]
    lines = [
        "# Full-state observation schema", "",
        f"Dimension: **{n}**. Gym space: `Box(-1.0, 1.0, ({n},), float32)`. Every value is clipped to [-1, 1]; the",
        "\"clipped\" column is the share of validation steps on which the raw value fell outside that range.", "",
        "Built by `FullStateObservation` in `full_state_env.py` from MetaDrive 0.4.3 simulator state, every step:",
        "vehicle `position`, `velocity`, `speed`, `heading_theta`, `steering`, `navigation` (current lane, route",
        "checkpoints, final lane, `route_completion`) for the ego, and the same objects for each traffic vehicle,",
        "looked up by name in `engine.get_objects()`.", "",
        "**Ego frame:** forward = along the ego's heading, left = 90 degrees counter-clockwise from it.", "",
        "**Traffic slot order:** at each reset the traffic vehicles are sorted by their spawn location",
        "`(spawn road start node, spawn road end node, lane index, spawn longitude)` from each vehicle's",
        "`config[\"spawn_lane_index\"]` and `config[\"spawn_longitude\"]`. A vehicle keeps its slot for the whole",
        "episode. When MetaDrive's traffic manager removes a vehicle (it drove off the end of an exit road), its slot",
        "stays and every value in it, including `present`, becomes 0. Spawn points are 10 m apart, so there are no ties.",
        "With density 0.2 there are always exactly 6 traffic vehicles at reset.", "",
        "**Not included:** reward, `goal_reached`, episode outcome, future states, or anything computed from them.",
        "The traffic `route_*` flags are the IDM driver's current route plan (its navigation destination), which is",
        "part of the simulator state at that moment.", "",
        "**Removed relative to the previous LiDAR experiment:** the 240 LiDAR beams. In this map they detect only",
        "vehicles (validation check: with traffic off, every beam reads 1.0 = nothing detected for a whole episode),",
        "and every vehicle is described exactly by the traffic slots.", "",
        "| Index | Feature | Meaning | Clipped |", "|---|---|---|---|",
    ]
    rows = [(f"state[{i}]", d) for i, d in enumerate(STATE_OBS_ROWS)]
    rows += [(f"ego.{nm}", d) for nm, d in EGO_FEATURES]
    for k in range(NUM_SLOTS):
        rows += [(f"slot{k}.{nm}", d if k == 0 else "as slot0") for nm, d in SLOT_FEATURES]
    assert len(rows) == n
    for i, (nm, d) in enumerate(rows):
        lines.append(f"| {i} | `{nm}` | {d} | {100 * clip_rates[i]:.1f}% |")
    lines += ["", "Groups: indices 0-18 MetaDrive `StateObservation` (unchanged from the previous experiment: ego state",
              "9 + two route checkpoints 10), 19-34 ground-truth ego features, 35-136 six traffic slots of 17 values",
              f"(slot k starts at index {SLOT0} + 17 k)."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    c = Checks()
    env = StaticGoalTrafficFullStateEnv(dict(spawn_seed=0, traffic_seed=0, extra_lidar_obs=True))
    rng = np.random.default_rng(0)
    try:
        n = env.observation_space.shape[0]
        s = dict(steps=0, shape_ok=True, finite_ok=True, bounds_ok=True, slots_ok=True, policy_ok=True, idm_ok=True,
                 clipped=np.zeros(n), max_present=0, traffic_moved=False)
        # Checks 1, 2, 6, 7, 8, 9 over 20 episodes (scripted and random actions)
        for ep in range(20):
            drive(env, throttle_policy if ep % 2 == 0 else random_policy, rng, s)
        c.record("1 shape matches observation space", s["shape_ok"], f"{s['steps']} observations, shape {env.observation_space.shape}")
        c.record("2 no NaN or inf", s["finite_ok"], f"{s['steps']} observations")
        c.record("6 traffic slots stable within episodes", s["slots_ok"],
                 "slot names fixed per episode, slot features match the named vehicle every step, removed slots are all-zero and never return")
        clip_rate = s["clipped"] / s["steps"]
        c.record("7 features within declared bounds", s["bounds_ok"],
                 f"all values in [-1, 1]; raw values clipped on {100 * clip_rate.mean():.2f}% of feature-steps "
                 f"(max {100 * clip_rate.max():.1f}% for index {int(clip_rate.argmax())})")
        c.record("8 EnvInputPolicy applies the action to the ego", s["policy_ok"], "info['policy'] and raw_action checked every step")
        c.record("9 traffic vehicles controlled by IDM", s["idm_ok"] and s["traffic_moved"],
                 f"every traffic vehicle's policy is IDMPolicy every step; up to {s['max_present']} present; traffic moved: {s['traffic_moved']}")

        # Check 3: same ego and traffic seeds -> same initial observation (and same first 30 steps)
        def rollout(seed, steps=30):
            env.seed_spawns(seed)
            env.seed_traffic(seed)
            obs, _ = env.reset()
            out = [obs]
            for _ in range(steps):
                obs, *_ = env.step([0.0, 0.5])
                out.append(obs)
            return np.array(out)
        a, b = rollout(7), rollout(7)
        max_diff = float(np.abs(a - b).max())
        c.record("3 same seeds reproduce the observation", np.array_equal(a[0], b[0]) and max_diff < 1e-5,
                 f"initial observation bit-identical across two resets with seed 7; the next 30 steps agree to "
                 f"{max_diff:.1e} (Bullet physics float noise)")

        # Check 4: different traffic seeds -> different traffic features (same ego start)
        env.seed_spawns(3); env.seed_traffic(3); o1, _ = env.reset()
        env.seed_spawns(3); env.seed_traffic(4); o2, _ = env.reset()
        same_ego = np.array_equal(o1[:SLOT0], o2[:SLOT0])
        diff_traffic = not np.array_equal(o1[SLOT0:], o2[SLOT0:])
        c.record("4 different traffic seeds change traffic features", same_ego and diff_traffic,
                 f"same ego start: ego part identical={same_ego}; traffic part differs={diff_traffic}")

        # Check 5: perturbing one vehicle's position / velocity changes exactly its slot
        env.seed_spawns(0); env.seed_traffic(0); env.reset()
        for _ in range(40):
            env.step([0.0, 0.5])
        fso = env.full_state_obs
        k = next(i for i, v in enumerate(fso.slot_vehicles()) if v is not None)
        v = fso.slot_vehicles()[k]
        before = fso.observe(env.agent).copy()
        v.set_position(np.asarray(v.position) + np.array([4.0, 2.0]))
        after_pos = fso.observe(env.agent).copy()
        v.set_velocity([1.0, 0.0], value=12.0)
        after_vel = fso.observe(env.agent).copy()
        others = [i for i in range(len(before)) if not (SLOT0 + k * SLOT_W <= i < SLOT0 + (k + 1) * SLOT_W)]
        pos_idx = [SLOT0 + k * SLOT_W + F[x] for x in ("rel_forward", "rel_left", "distance")]
        vel_idx = [SLOT0 + k * SLOT_W + F[x] for x in ("rel_vforward", "rel_vleft", "speed")]
        ok_pos = np.all(before[pos_idx] != after_pos[pos_idx]) and np.array_equal(before[others], after_pos[others])
        ok_vel = np.any(after_pos[vel_idx] != after_vel[vel_idx]) and np.array_equal(after_pos[others], after_vel[others]) \
            and abs(after_vel[SLOT0 + k * SLOT_W + F["speed"]] - 12.0 / 30.0) < 1e-3
        c.record("5 moving a vehicle changes its slot only", ok_pos and ok_vel,
                 f"slot {k}: position change -> rel_forward/rel_left/distance changed, velocity set to 12 m/s -> "
                 f"speed feature {after_vel[SLOT0 + k * SLOT_W + F['speed']]:.3f} (expected 0.400); all other features unchanged")

        # Extra: with traffic off, the LiDAR beams of the old observation never detect anything in this map
        env.config["traffic_density"] = 0.0
        env.seed_spawns(0); env.seed_traffic(0)
        obs, info = env.reset()
        beams_empty = True
        for _ in range(300):
            beams_empty &= bool(np.all(info["lidar_obs"][STATE_DIM:] == 1.0))
            obs, r, term, trunc, info = env.step([0.0, 0.5])
            if term or trunc:
                break
        absent = bool(np.all(obs[SLOT0:] == 0.0))
        c.record("extra: LiDAR sees nothing without traffic", beams_empty and absent,
                 "with traffic_density 0, all 240 beams read 1.0 every step (no detection) and all traffic slots are 0")
        env.config["traffic_density"] = 0.2
        write_schema(env, os.path.join(HERE, "OBSERVATION_SCHEMA.md"), clip_rate)
    finally:
        env.close()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "validation.json"), "w") as f:
        json.dump(dict(observation_dim=int(n), checks=c.results), f, indent=2)
    print("ALL PASSED" if all(r["passed"] for r in c.results.values()) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
