"""Smoke check: reload a full-state checkpoint and drive one episode with it.

    python static_goal_traffic_full_state/smoke_reload.py <checkpoint_dir>
"""
import os
import sys

import numpy as np
from ray.rllib.policy.policy import Policy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from full_state_env import StaticGoalTrafficFullStateEnv, is_idm  # noqa: E402

JUNCTION_X, JUNCTION_Y = (50.0, 87.5), (-10.0, 27.5)

if __name__ == "__main__":
    policy = Policy.from_checkpoint(os.path.join(sys.argv[1], "policies", "default_policy"))
    assert policy.observation_space.shape == (137, ), policy.observation_space
    env = StaticGoalTrafficFullStateEnv(dict(spawn_seed=0, traffic_seed=0))
    try:
        obs, info = env.reset()
        in_junction, steps = set(), 0
        while True:
            action = np.clip(policy.compute_single_action(obs, explore=False)[0], -1, 1)
            obs, r, term, trunc, info = env.step(action)
            assert info["policy"] == "EnvInputPolicy" and np.allclose(info["raw_action"], action, atol=1e-6)
            assert all(is_idm(p) for p in env.traffic_policies().values())
            for v in env.traffic_vehicles():
                x, y = v.position
                if JUNCTION_X[0] <= x <= JUNCTION_X[1] and JUNCTION_Y[0] <= y <= JUNCTION_Y[1] and v.speed > 1:
                    in_junction.add(v.name)
            steps += 1
            if term or trunc:
                break
        print(f"reloaded {sys.argv[1]}: obs space {policy.observation_space.shape}, {steps} steps driven by "
              f"EnvInputPolicy with the PPO action, {len(in_junction)} traffic vehicles drove through the junction, "
              f"all traffic IDM-controlled")
    finally:
        env.close()
