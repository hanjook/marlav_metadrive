"""Train PPO on the static-goal task with background traffic.

Same official training code and the same PPO settings as static_goal/train_static_goal.py (ppo_config() is imported
from there unchanged: torch, seed 0, 4 workers, train batch 4000, official PPO hyper-parameters); only the env is
swapped for StaticGoalTrafficEnv. Runs go to static_goal_traffic/runs/static_goal_traffic_ppo/.

    python static_goal_traffic/train_traffic.py --timesteps 20000 --exp-name smoke_traffic        # smoke test
    python static_goal_traffic/train_traffic.py --timesteps 500000
    python static_goal_traffic/train_traffic.py --timesteps 1000000 --restore <checkpoint_dir>    # resume / extend
"""
import argparse
import glob
import os
import sys

from metadrive.envs.gym_wrapper import createGymWrapper
from metadrive.examples.train_generalization_experiment import train

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "static_goal"))
from train_static_goal import ppo_config  # noqa: E402  (sets PYTHONPATH for static_goal/)
os.environ["PYTHONPATH"] = HERE + os.pathsep + os.environ["PYTHONPATH"]  # so Ray workers can import traffic_env
from traffic_env import StaticGoalTrafficEnv  # noqa: E402

RUNS = os.path.join(HERE, "runs")
EXP = "static_goal_traffic_ppo"


def traffic_ppo_config(num_workers=4, train_batch_size=4000):
    config = ppo_config(num_workers, train_batch_size)
    config["env"] = createGymWrapper(StaticGoalTrafficEnv)
    return config


def latest_checkpoint(exp=EXP):
    ckpts = glob.glob(os.path.join(RUNS, exp, "*", "checkpoint_*"))
    return max(ckpts, key=os.path.getmtime) if ckpts else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4000)
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="save every N PPO iterations")
    parser.add_argument("--restore", type=str, default=None, help="checkpoint dir to resume from")
    parser.add_argument("--exp-name", type=str, default=EXP)
    args = parser.parse_args()

    from ray.rllib.algorithms.ppo import PPO  # class instead of "PPO": avoids RLlib importing TF-only algorithms

    os.makedirs(RUNS, exist_ok=True)
    os.chdir(RUNS)  # the official train() writes to local_dir="."
    extra = dict(restore=args.restore) if args.restore else {}
    train(
        PPO,
        traffic_ppo_config(args.workers, args.batch),
        stop={"timesteps_total": args.timesteps},
        exp_name=args.exp_name,
        checkpoint_freq=args.checkpoint_freq,
        **extra
    )
    print("Latest checkpoint:", latest_checkpoint(args.exp_name))
