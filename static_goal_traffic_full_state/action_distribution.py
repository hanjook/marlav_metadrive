"""Diagnostic: the Gaussian action distribution a checkpoint outputs along its own (sampled-action) test episodes.

    python static_goal_traffic_full_state/action_distribution.py <group> [<group> ...]
Writes results/action_distribution.json: per group, mean and std of the policy's Gaussian mean and std for steering
and throttle/brake, and how often a sampled action is clipped at the [-1, 1] bounds.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate_full_state import RESULTS, noise_seed, parse_group  # noqa: E402
from full_state_env import StaticGoalTrafficFullStateEnv  # noqa: E402

if __name__ == "__main__":
    from ray.rllib.policy.policy import Policy
    env = StaticGoalTrafficFullStateEnv(dict(extra_lidar_obs=True))
    out = {}
    try:
        for name in sys.argv[1:]:
            g = parse_group(name)
            policy = Policy.from_checkpoint(os.path.join(g["checkpoint"], "policies", "default_policy"))
            lidar = g["kind"] == "lidar"
            env.want_lidar = lidar
            env.seed_spawns(0)
            env.seed_traffic(0)
            means, stds, raw, speeds = [], [], [], []
            for ep in range(10):
                torch.manual_seed(noise_seed(0, ep))
                obs, info = env.reset()
                while True:
                    a, _, extra = policy.compute_single_action(info["lidar_obs"] if lidar else obs, explore=True)
                    d = extra["action_dist_inputs"]
                    means.append(d[:2]); stds.append(np.exp(d[2:])); raw.append(a); speeds.append(env.agent.speed)
                    obs, r, term, trunc, info = env.step(np.clip(a, -1, 1))
                    if term or trunc:
                        break
            means, stds, raw, speeds = map(np.array, (means, stds, raw, speeds))
            out[name] = dict(
                steps=len(raw),
                gaussian_mean_avg=dict(steering=float(means[:, 0].mean()), throttle=float(means[:, 1].mean())),
                gaussian_mean_std_over_steps=dict(steering=float(means[:, 0].std()), throttle=float(means[:, 1].std())),
                gaussian_std_avg=dict(steering=float(stds[:, 0].mean()), throttle=float(stds[:, 1].mean())),
                gaussian_std_median=dict(steering=float(np.median(stds[:, 0])), throttle=float(np.median(stds[:, 1]))),
                share_steps_std_over_1=dict(steering=float(np.mean(stds[:, 0] > 1)), throttle=float(np.mean(stds[:, 1] > 1))),
                share_steps_std_over_1_when_slow=float(np.mean(stds[speeds < 3, 1] > 1)) if (speeds < 3).any() else None,
                share_steps_std_over_1_when_fast=float(np.mean(stds[speeds >= 3, 1] > 1)) if (speeds >= 3).any() else None,
                sampled_clipped_share=dict(steering=float(np.mean(np.abs(raw[:, 0]) > 1)),
                                           throttle=float(np.mean(np.abs(raw[:, 1]) > 1))),
                sampled_brake_share=float(np.mean(np.clip(raw[:, 1], -1, 1) < 0)),
            )
            print(name, json.dumps(out[name]))
    finally:
        env.close()
    with open(os.path.join(RESULTS, "action_distribution.json"), "w") as f:
        json.dump(out, f, indent=2)
