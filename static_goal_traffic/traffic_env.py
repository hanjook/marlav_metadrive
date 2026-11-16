"""Static-goal intersection task with background traffic.

Same task as static_goal/static_goal_env.py (same map, same randomized west-side start, same fixed east-exit
destination and success region, same observation, action and reward). The only change is background traffic from
MetaDrive's own traffic manager (PGTrafficManager), whose vehicles are driven by MetaDrive's IDMPolicy. The ego is
still the only learner and is still driven by EnvInputPolicy.

Traffic configuration (all documented MetaDriveEnv config keys):
  * traffic_mode="trigger"  MetaDrive's default. Vehicles are placed on the intersection block (inside the junction,
                            on the exit roads and on the three other incoming arms) and start driving when the ego
                            enters the last road before the junction. Each picks a random exit, so many cross or
                            merge into the ego's path. "respawn" mode was rejected: it also spawns on the ego's own
                            approach road and can place a car on top of the ego at reset.
  * traffic_density=0.2     0.1 gave 3 vehicles per episode, 0.15 gave 4, 0.2 gives 6.
  * random_traffic=True     the traffic manager is not reseeded from the (single) map seed on every reset, so traffic
                            differs between episodes. Each episode's traffic is reproducible: before every reset the
                            traffic manager is seeded with a per-episode traffic seed drawn from `traffic_seed`.
"""
import os
import sys

import numpy as np

from metadrive.base_class.randomizable import Randomizable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static_goal"))
from static_goal_env import StaticGoalIntersectionEnv  # noqa: E402

TRAFFIC_DENSITY = 0.2
TRAFFIC_MODE = "trigger"


class StaticGoalTrafficEnv(StaticGoalIntersectionEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(
            dict(
                traffic_density=TRAFFIC_DENSITY,
                traffic_mode=TRAFFIC_MODE,
                random_traffic=True,
                traffic_seed=None,  # None: random traffic; int: reproducible sequence of per-episode traffic seeds
            ),
            allow_add_new_key=True,
        )
        return config

    def __init__(self, config=None):
        super().__init__(config)
        self._traffic_rng = np.random.default_rng(self.config["traffic_seed"])
        self.episode_traffic_seed = None

    def seed_traffic(self, seed):
        self._traffic_rng = np.random.default_rng(seed)

    def reset(self, *args, **kwargs):
        self.lazy_init()  # the traffic manager exists only after the engine is created
        self.episode_traffic_seed = int(self._traffic_rng.integers(0, 2**31 - 1))
        # PGTrafficManager.seed() is a no-op when random_traffic=True, so seed its RNG through the base class.
        Randomizable.seed(self.engine.traffic_manager, self.episode_traffic_seed)
        obs, info = super().reset(*args, **kwargs)
        info.update(self._traffic_info())
        return obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        info.update(self._traffic_info())
        return obs, reward, terminated, truncated, info

    def traffic_vehicles(self):
        """All non-ego vehicles in the scene, including ones not yet triggered."""
        return [v for v in self.engine.traffic_manager.vehicles if v is not self.agent]

    def _traffic_info(self):
        others = self.traffic_vehicles()
        ego = np.asarray(self.agent.position)
        dists = [np.linalg.norm(np.asarray(v.position) - ego) for v in others]
        return dict(
            traffic_seed=self.episode_traffic_seed,
            num_traffic_vehicles=len(others),
            min_dist_to_traffic=float(min(dists)) if dists else float("inf"),
        )
