"""Train PPO on the static-goal intersection task with MetaDrive's official training code.

Reuses `train()` and `DrivingCallbacks` from metadrive/examples/train_generalization_experiment.py and keeps its PPO
hyper-parameters (rollout_fragment_length, sgd_minibatch_size, num_sgd_iter, lr, lambda). Only the environment and
the CPU-sized batch/worker counts differ.

    python static_goal/train_static_goal.py --timesteps 200000
    python static_goal/train_static_goal.py --timesteps 400000 --restore <checkpoint_dir>   # resume
"""
import argparse
import glob
import os
import sys

from metadrive.envs.gym_wrapper import createGymWrapper
from metadrive.examples.train_generalization_experiment import train

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = HERE + os.pathsep + os.environ.get("PYTHONPATH", "")  # so Ray workers can import it
from static_goal_env import StaticGoalIntersectionEnv  # noqa: E402

RUNS = os.path.join(HERE, "runs")
EXP = "static_goal_ppo"


def ppo_config(num_workers=4, train_batch_size=4000):
    return dict(
        env=createGymWrapper(StaticGoalIntersectionEnv),
        env_config=dict(),
        framework="torch",
        seed=0,
        # Official PPO values
        rollout_fragment_length=200,
        sgd_minibatch_size=256,
        num_sgd_iter=10,
        lr=3e-4,
        **{"lambda": 0.95},
        # CPU-sized (official: train_batch_size=20000, num_workers=5, horizon=1000 with periodic evaluation).
        # The episode horizon (300 steps) is set inside the env.
        train_batch_size=train_batch_size,
        num_workers=num_workers,
        num_gpus=0,
        num_cpus_per_worker=1,
        num_cpus_for_driver=1,
        evaluation_interval=None,
    )


def latest_checkpoint(exp=EXP):
    ckpts = glob.glob(os.path.join(RUNS, exp, "*", "checkpoint_*"))
    return max(ckpts, key=os.path.getmtime) if ckpts else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4000)
    parser.add_argument("--checkpoint-freq", type=int, default=5, help="save every N PPO iterations")
    parser.add_argument("--restore", type=str, default=None, help="checkpoint dir to resume from")
    parser.add_argument("--exp-name", type=str, default=EXP)
    args = parser.parse_args()

    from ray.rllib.algorithms.ppo import PPO  # class instead of "PPO": avoids RLlib importing TF-only algorithms

    os.makedirs(RUNS, exist_ok=True)
    os.chdir(RUNS)  # the official train() writes to local_dir="."
    extra = dict(restore=args.restore) if args.restore else {}
    train(
        PPO,
        ppo_config(args.workers, args.batch),
        stop={"timesteps_total": args.timesteps},
        exp_name=args.exp_name,
        checkpoint_freq=args.checkpoint_freq,
        **extra
    )
    print("Latest checkpoint:", latest_checkpoint(args.exp_name))
