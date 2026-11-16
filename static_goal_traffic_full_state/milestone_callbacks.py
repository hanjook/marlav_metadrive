"""MetaDrive's official DrivingCallbacks plus checkpoints at fixed step milestones.

Tune saves a checkpoint after any iteration whose result has should_checkpoint=True. A milestone M is hit by the
first iteration whose timesteps_total reaches M (with 4000-step batches: 252k, 500k, 752k, 1M). Milestones come from
the FULL_STATE_MILESTONES environment variable (comma-separated steps) so the smoke test can use small ones.
"""
import os

from metadrive.examples.train_generalization_experiment import DrivingCallbacks

DEFAULT_MILESTONES = "250000,500000,750000,1000000"


class MilestoneCallbacks(DrivingCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)
        milestones = [int(m) for m in os.environ.get("FULL_STATE_MILESTONES", DEFAULT_MILESTONES).split(",")]
        now = result["timesteps_total"]
        before = now - result["num_env_steps_sampled_this_iter"]
        if any(before < m <= now for m in milestones):
            result["should_checkpoint"] = True
