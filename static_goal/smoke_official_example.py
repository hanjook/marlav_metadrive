"""Step 2: run MetaDrive's official PPO example (train_generalization_experiment.py) on ONE grid point.

Reuses the example's own `train()` and `DrivingCallbacks` unchanged. Only the grid search is collapsed to a
single configuration and the batch/worker sizes are shrunk so it finishes quickly on a CPU laptop.

    python static_goal/smoke_official_example.py            # train 2 iterations, save checkpoints
    python static_goal/smoke_official_example.py --reload   # reload the latest checkpoint and drive 1 episode
"""
import argparse
import glob
import os

from metadrive import MetaDriveEnv
from metadrive.envs.gym_wrapper import createGymWrapper
from metadrive.examples.train_generalization_experiment import train

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs")
EXP = "official_example_smoke"

# One point of the official grid: num_scenarios=1, start_seed=5000, traffic_density=0.1
ENV_CONFIG = dict(num_scenarios=1, start_seed=5000, random_traffic=False, traffic_density=0.1, log_level=50)

CONFIG = dict(
    env=createGymWrapper(MetaDriveEnv),
    env_config=ENV_CONFIG,
    framework="torch",
    # Official values kept: horizon, rollout_fragment_length, sgd_minibatch_size, num_sgd_iter, lr, lambda.
    horizon=1000,
    rollout_fragment_length=200,
    sgd_minibatch_size=256,
    num_sgd_iter=10,
    lr=3e-4,
    **{"lambda": 0.95},
    # CPU-sized changes (official: train_batch_size=20000, num_workers=5, evaluation every 2 iters with 5 workers).
    train_batch_size=2000,
    num_workers=2,
    num_gpus=0,
    num_cpus_per_worker=1,
    num_cpus_for_driver=1,
    evaluation_interval=None,
)


def reload_and_drive():
    from ray.rllib.policy.policy import Policy
    ckpts = sorted(glob.glob(os.path.join(OUT, EXP, "*", "checkpoint_*")))
    assert ckpts, "no checkpoint found, run training first"
    ckpt = ckpts[-1]
    print("Reloading", ckpt)
    # Load only the policy: building a full PPO here would create a second MetaDrive engine in this process.
    policy = Policy.from_checkpoint(os.path.join(ckpt, "policies", "default_policy"))
    env = createGymWrapper(MetaDriveEnv)(ENV_CONFIG)
    obs, total, steps, done, info = env.reset(), 0.0, 0, False, {}
    while not done:
        obs, r, done, info = env.step(policy.compute_single_action(obs, explore=False)[0])
        total += r
        steps += 1
    env.close()
    print(f"Reloaded policy drove {steps} steps, reward {total:.2f}, arrive_dest={info['arrive_dest']}, "
          f"crash={info['crash']}, out_of_road={info['out_of_road']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--iters", type=int, default=2)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.chdir(OUT)  # the official train() writes results to local_dir="."
    if args.reload:
        reload_and_drive()
    else:
        from ray.rllib.algorithms.ppo import PPO  # class instead of "PPO": avoids RLlib importing TF-only algorithms
        train(PPO, CONFIG, stop={"training_iteration": args.iters}, exp_name=EXP, checkpoint_freq=1)
