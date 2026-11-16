"""Static-goal intersection task on MetaDrive's built-in 4-way intersection (map="X").

PROVISIONAL setup: no teammate environment file was found, so this uses MetaDrive's own intersection block.

Task: ego starts on the left (west) approach, drives straight through the intersection and must reach a fixed
destination region at the end of the right (east) exit road. No traffic.

What is reused from MetaDrive unchanged:
  * observation  (LidarStateObservation: ego state + navigation checkpoints + lidar, 259 dims)
  * action       (continuous [steering, throttle/brake], applied by EnvInputPolicy, no other controller)
  * navigation   (NodeNetworkNavigation: shortest route from spawn lane to the `destination` road node)
  * reward       (MetaDriveEnv.reward_function: progress along the route + speed, +10 success, -5 crash/off-road)
  * success      (MetaDriveEnv._is_arrive_destination on navigation.final_lane)

What is added here (small): the destination is pinned to TARGET_NODE instead of chosen at random, spawn lane and
position on the left approach vary per episode, and `info` gets `goal_reached` and `dist_to_goal`. Navigation,
reward and success all derive from the same `navigation.final_lane`, which is set from TARGET_NODE.
"""
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.constants import DEFAULT_AGENT

SPAWN_ROAD = (">", ">>")  # first road of the left (west) approach, heading east
TARGET_NODE = "1X1_1_"  # end node of the east exit road 1X1_0_ -> 1X1_1_ (straight through)
SPAWN_LANES = (0, 1, 2)  # all three eastbound lanes of the approach
SPAWN_LONGITUDE = (0.0, 20.0)  # metres along the approach, sampled uniformly
GOAL_HALF_LENGTH = 5.0  # same +-5 m window that MetaDriveEnv._is_arrive_destination uses


class StaticGoalIntersectionEnv(MetaDriveEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(
            dict(
                map="X",
                num_scenarios=1,
                start_seed=0,
                traffic_density=0.0,
                random_spawn_lane_index=False,
                horizon=300,  # 300 steps x 0.1 s = 30 s; the route is ~120 m
                log_level=50,
                spawn_seed=None,  # None: random spawns; int: reproducible spawn sequence
                agent_configs={
                    DEFAULT_AGENT: dict(
                        use_special_color=True, spawn_lane_index=SPAWN_ROAD + (0, ), destination=TARGET_NODE
                    )
                },
            ),
            allow_add_new_key=True,
        )
        return config

    def __init__(self, config=None):
        super().__init__(config)
        self._spawn_rng = np.random.default_rng(self.config["spawn_seed"])

    def seed_spawns(self, seed):
        self._spawn_rng = np.random.default_rng(seed)

    def reset(self, *args, **kwargs):
        agent_cfg = self.config["agent_configs"][DEFAULT_AGENT]
        agent_cfg["spawn_lane_index"] = SPAWN_ROAD + (int(self._spawn_rng.choice(SPAWN_LANES)), )
        agent_cfg["spawn_longitude"] = float(self._spawn_rng.uniform(*SPAWN_LONGITUDE))
        obs, info = super().reset(*args, **kwargs)
        nav = self.agent.navigation
        assert nav.checkpoints[-1] == TARGET_NODE, nav.checkpoints
        info.update(self._goal_info())
        return obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        info.update(self._goal_info())
        return obs, reward, terminated, truncated, info

    def goal_region(self):
        """Corners of the acceptance region, built from the exact bounds in MetaDriveEnv._is_arrive_destination."""
        nav = self.agent.navigation
        lane, w, n = nav.final_lane, nav.get_current_lane_width(), nav.get_current_lane_num()
        longs = (lane.length - GOAL_HALF_LENGTH, lane.length + GOAL_HALF_LENGTH)
        lats = ((0.5 - n) * w, w / 2)
        return np.array([lane.position(longs[i], lats[j]) for i, j in ((0, 0), (1, 0), (1, 1), (0, 1))])

    def goal_center(self):
        return self.goal_region().mean(axis=0)

    def _goal_info(self):
        v = self.agent
        return dict(
            goal_reached=bool(self._is_arrive_destination(v)),
            dist_to_goal=float(np.linalg.norm(np.asarray(v.position) - self.goal_center())),
        )
