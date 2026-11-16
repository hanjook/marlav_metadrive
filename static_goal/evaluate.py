"""Evaluate a saved PPO checkpoint (or an untrained policy) on N reproducible episodes.

Both policies see the same spawn sequence (seeded by --seed) and act deterministically unless --explore is given.

    python static_goal/evaluate.py                       # latest checkpoint of static_goal_ppo
    python static_goal/evaluate.py --checkpoint <dir>
    python static_goal/evaluate.py --untrained           # randomly initialised policy, same network
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from static_goal_env import StaticGoalIntersectionEnv  # noqa: E402
from train_static_goal import latest_checkpoint  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def load_policy(checkpoint=None, untrained=False, env=None, seed=0):
    """Trained: restore the policy weights from an RLlib checkpoint. Untrained: same network, random weights."""
    if not untrained:
        from ray.rllib.policy.policy import Policy
        return Policy.from_checkpoint(os.path.join(checkpoint, "policies", "default_policy"))
    import torch
    from metadrive.envs.gym_wrapper import gymnasium_to_gym
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
    torch.manual_seed(seed)
    config = PPOConfig().framework("torch").to_dict()
    return PPOTorchPolicy(gymnasium_to_gym(env.observation_space), gymnasium_to_gym(env.action_space), config)


def run_episode(env, policy, explore=False):
    """Run one episode. Returns the summary and the per-step trajectory (used by replay.py)."""
    obs, info = env.reset()
    total, traj = 0.0, []
    while True:
        action = np.clip(policy.compute_single_action(obs, explore=explore)[0], -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        # The action the vehicle used must be the policy's action (no other controller in the loop).
        assert info["policy"] == "EnvInputPolicy" and np.allclose(info["raw_action"], action, atol=1e-6)
        total += reward
        v = env.agent
        traj.append(
            dict(
                x=float(v.position[0]),
                y=float(v.position[1]),
                heading=float(v.heading_theta),
                speed=float(v.speed),
                dist_to_goal=info["dist_to_goal"],
                goal_reached=info["goal_reached"]
            )
        )
        if terminated or truncated:
            break
    summary = dict(
        episode_outcome(info), reward=float(total), steps=len(traj), final_dist_to_goal=info["dist_to_goal"]
    )
    return summary, traj


def episode_outcome(info):
    """Outcome flags from the last step's info."""
    goal = info["goal_reached"]
    crash, off_road = bool(info["crash"]), bool(info["out_of_road"]) and not goal
    return dict(goal_reached=goal, crash=crash, out_of_road=off_road, timeout=not (goal or crash or off_road))


def evaluate(env, policy, episodes=20, seed=0, explore=False):
    import torch
    torch.manual_seed(seed)  # makes sampled actions (--explore) reproducible too
    env.seed_spawns(seed)
    return [run_episode(env, policy, explore)[0] for _ in range(episodes)]


def summarize(rows):
    def rate(key):
        return float(np.mean([r[key] for r in rows]))

    return dict(
        episodes=len(rows),
        success_rate=rate("goal_reached"),
        collision_rate=rate("crash"),
        off_road_rate=rate("out_of_road"),
        timeout_rate=rate("timeout"),
        mean_reward=float(np.mean([r["reward"] for r in rows])),
        std_reward=float(np.std([r["reward"] for r in rows])),
        mean_steps=float(np.mean([r["steps"] for r in rows])),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="spawn sequence seed (same for every policy)")
    parser.add_argument("--explore", action="store_true", help="sample actions as during training")
    args = parser.parse_args()

    ckpt = None if args.untrained else (args.checkpoint or latest_checkpoint())
    label = ("untrained" if args.untrained else "trained") + ("_explore" if args.explore else "")
    env = StaticGoalIntersectionEnv()
    policy = load_policy(ckpt, untrained=args.untrained, env=env, seed=args.seed)
    rows = evaluate(env, policy, args.episodes, args.seed, args.explore)
    env.close()
    result = dict(policy=label, checkpoint=ckpt, seed=args.seed, explore=args.explore, summary=summarize(rows), episodes=rows)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"eval_{label}_seed{args.seed}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(policy=label, checkpoint=ckpt, **result["summary"]), indent=2))
    print("Saved", out)
