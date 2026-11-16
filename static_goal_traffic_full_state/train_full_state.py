"""Train PPO on the traffic task with the full-state observation.

Same official training code and PPO settings as the two earlier experiments (ppo_config() imported unchanged from
static_goal/train_static_goal.py: torch, 4 workers, train batch 4000, official PPO hyper-parameters). Only the env
(observation) differs. Several seeds run in parallel as one Tune grid; checkpoints are saved at step milestones
(milestone_callbacks.py) and at the end.

    python static_goal_traffic_full_state/train_full_state.py --timesteps 20000 --seeds 0 --exp-name smoke_full_state --milestones 8000,16000
    python static_goal_traffic_full_state/train_full_state.py --timesteps 1000000 --seeds 0,1,2
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
os.environ["PYTHONPATH"] = os.pathsep.join([HERE, os.path.join(ROOT, "static_goal_traffic"), os.environ["PYTHONPATH"]])
from full_state_env import StaticGoalTrafficFullStateEnv  # noqa: E402
from milestone_callbacks import DEFAULT_MILESTONES, MilestoneCallbacks  # noqa: E402

RUNS = os.path.join(HERE, "runs")
EXP = "full_state_ppo"


def full_state_ppo_config(seeds, num_workers=4, train_batch_size=4000):
    from ray import tune
    config = ppo_config(num_workers, train_batch_size)
    config["env"] = createGymWrapper(StaticGoalTrafficFullStateEnv)
    config["seed"] = tune.grid_search(seeds) if len(seeds) > 1 else seeds[0]
    return config


def checkpoints(exp=EXP):
    """{seed: {timesteps: checkpoint_dir}} for every saved checkpoint of an experiment."""
    import json
    out = {}
    for trial in glob.glob(os.path.join(RUNS, exp, "PPO_*")):
        with open(os.path.join(trial, "params.json")) as f:
            seed = json.load(f)["seed"]
        rows = open(os.path.join(trial, "progress.csv")).read().splitlines()
        header = rows[0].split(",")
        it_col, ts_col = header.index("training_iteration"), header.index("timesteps_total")
        steps = {int(r.split(",")[it_col]): int(r.split(",")[ts_col]) for r in rows[1:]}
        for ckpt in glob.glob(os.path.join(trial, "checkpoint_*")):
            out.setdefault(seed, {})[steps[int(ckpt.rsplit("_", 1)[1])]] = ckpt
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4000)
    parser.add_argument("--milestones", type=str, default=DEFAULT_MILESTONES)
    parser.add_argument("--exp-name", type=str, default=EXP)
    args = parser.parse_args()

    from ray.rllib.algorithms.ppo import PPO  # class instead of "PPO": avoids RLlib importing TF-only algorithms

    os.environ["FULL_STATE_MILESTONES"] = args.milestones  # read by MilestoneCallbacks in the trainer process
    os.makedirs(RUNS, exist_ok=True)
    os.chdir(RUNS)  # the official train() writes to local_dir="."
    train(
        PPO,
        full_state_ppo_config([int(s) for s in args.seeds.split(",")], args.workers, args.batch),
        stop={"timesteps_total": args.timesteps},
        exp_name=args.exp_name,
        checkpoint_freq=0,  # milestones + end only
        custom_callback=MilestoneCallbacks,
    )
    for seed, ck in sorted(checkpoints(args.exp_name).items()):
        print("seed", seed, "checkpoints at steps", sorted(ck))
