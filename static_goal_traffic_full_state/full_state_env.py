"""Static-goal traffic task with a full-state observation read from MetaDrive's simulator state.

Everything except the observation is inherited unchanged from static_goal_traffic/traffic_env.py
(map "X", randomized west-side ego start, fixed east destination, reward, 300-step horizon, traffic density 0.2,
trigger mode, random_traffic=True, per-episode traffic seeds, 6 IDM vehicles, EnvInputPolicy for the ego).

Observation (137 values, all in [-1, 1]; schema in OBSERVATION_SCHEMA.md, printed by `python full_state_env.py`):
  [0:19]    MetaDrive's own StateObservation, unchanged: ego state (9) + route checkpoints (10). This is the
            LiDAR-state observation of the previous experiment without its 240 LiDAR beams.
  [19:35]   ground-truth ego features (16)
  [35:137]  ground-truth features of 6 traffic vehicles, 17 per slot

Why the 240 LiDAR beams are dropped: in this map they only return other vehicles (MetaDrive's LiDAR does not hit
lane lines or road edges; road edges are already in the side-detector values of StateObservation), so every vehicle
they can see is described exactly by the traffic slots.

Slot order: at every reset the traffic vehicles are sorted once by where MetaDrive spawned them,
(spawn road start node, spawn road end node, lane index, spawn longitude), read from each vehicle's
config["spawn_lane_index"] and config["spawn_longitude"]. Every vehicle keeps its slot for the whole episode; when
PGTrafficManager removes it (it left the road at the end of an exit), the slot stays and its mask becomes 0.
Spawn positions are unique (10 m apart on a lane), so the order has no ties and does not depend on dict order.
"""
import os
import sys

import numpy as np
import gymnasium as gym

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.state_obs import LidarStateObservation, StateObservation
from metadrive.policy.idm_policy import IDMPolicy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "static_goal_traffic"))
sys.path.insert(0, os.path.join(ROOT, "static_goal"))
from traffic_env import StaticGoalTrafficEnv  # noqa: E402

NUM_SLOTS = 6
# Map "X" road nodes (from its road graph): the 4 incoming roads end at an entry node, junction lanes run from an
# entry node to an exit node, outgoing roads start at an exit node.
ENTRY_NODES = (">>>", "-1X0_0_", "-1X1_0_", "-1X2_0_")
EXIT_NODES = ("->>>", "1X0_0_", "1X1_0_", "1X2_0_")
# Route destinations of traffic vehicles, by direction of the exit they are heading to.
EXIT_DIRECTION = {"1X1_1_": 0, "1X2_1_": 1, "1X0_1_": 2, "->>": 3, "->": 3}  # east, north, south, west

POS_SCALE, VEL_SCALE, REL_VEL_SCALE, DIST_SCALE = 150.0, 30.0, 40.0, 150.0
MAP_CENTER, MAP_HALF = np.array([65.0, 10.0]), np.array([80.0, 70.0])

EGO_FEATURES = [
    ("x", "world x, (x - 65) / 80"),
    ("y", "world y, (y - 10) / 70"),
    ("vx", "world velocity x / 30 m/s"),
    ("vy", "world velocity y / 30 m/s"),
    ("speed", "speed / 30 m/s"),
    ("sin_heading", "sin(heading)"),
    ("cos_heading", "cos(heading)"),
    ("steering", "current steering command, in [-1, 1]"),
    ("lane_long", "longitudinal position on current lane / lane length"),
    ("lane_lat", "lateral offset from current lane centre / lane width"),
    ("on_approach", "1 if on a road leading into the junction"),
    ("in_junction", "1 if on a junction lane"),
    ("on_exit", "1 if on a road leaving the junction"),
    ("dest_forward", "destination centre, ego-frame forward offset / 150 m"),
    ("dest_left", "destination centre, ego-frame left offset / 150 m"),
    ("route_completion", "MetaDrive navigation.route_completion, clipped to [0, 1]"),
]
SLOT_FEATURES = [
    ("present", "1 if the vehicle exists in the simulator, else 0 (all other features 0)"),
    ("rel_forward", "relative position, ego-frame forward / 150 m"),
    ("rel_left", "relative position, ego-frame left / 150 m"),
    ("rel_vforward", "relative velocity (theirs - ego), ego-frame forward / 40 m/s"),
    ("rel_vleft", "relative velocity (theirs - ego), ego-frame left / 40 m/s"),
    ("speed", "their speed / 30 m/s"),
    ("sin_dheading", "sin(their heading - ego heading)"),
    ("cos_dheading", "cos(their heading - ego heading)"),
    ("distance", "centre distance to ego / 150 m"),
    ("on_approach", "1 if on a road leading into the junction"),
    ("in_junction", "1 if on a junction lane"),
    ("on_exit", "1 if on a road leaving the junction"),
    ("same_road_as_ego", "1 if on the same road (same start and end node) as the ego"),
    ("route_east", "1 if its IDM route ends at the east exit"),
    ("route_north", "1 if its IDM route ends at the north exit"),
    ("route_south", "1 if its IDM route ends at the south exit"),
    ("route_west", "1 if its IDM route ends at the west exit"),
]


def lane_relation(lane_index):
    start, end = lane_index[0], lane_index[1]
    return [float(end in ENTRY_NODES), float(start in ENTRY_NODES and end in EXIT_NODES), float(start in EXIT_NODES)]


def ego_frame(vec, heading):
    c, s = np.cos(heading), np.sin(heading)
    return np.array([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1]])


class FullStateObservation(BaseObservation):
    def __init__(self, config):
        super().__init__(config)
        self.state_obs = StateObservation(config)
        self.slot_names = [None] * NUM_SLOTS
        self.raw = None  # last observation before clipping (validation checks how often clipping is needed)

    @property
    def observation_space(self):
        n = self.state_obs.observation_space.shape[0] + len(EGO_FEATURES) + NUM_SLOTS * len(SLOT_FEATURES)
        return gym.spaces.Box(-1.0, 1.0, shape=(n, ), dtype=np.float32)

    def reset(self, env, vehicle=None):
        """Called by MetaDrive once per episode, after the traffic manager has spawned this episode's vehicles."""
        others = [o for o in self.engine.get_objects().values() if isinstance(o, BaseVehicle) and o is not vehicle]
        if len(others) > NUM_SLOTS:
            raise RuntimeError(f"{len(others)} traffic vehicles, observation has {NUM_SLOTS} slots")

        def spawn_key(v):
            lane = v.config["spawn_lane_index"]
            return (str(lane[0]), str(lane[1]), int(lane[2]), float(v.config["spawn_longitude"]))

        names = [v.name for v in sorted(others, key=spawn_key)]
        self.slot_names = names + [None] * (NUM_SLOTS - len(names))

    def slot_vehicles(self):
        objects = self.engine.get_objects()
        return [objects.get(n) if n is not None else None for n in self.slot_names]

    def observe(self, vehicle):
        heading = vehicle.heading_theta
        pos, vel = np.asarray(vehicle.position, dtype=float), np.asarray(vehicle.velocity, dtype=float)
        lane = vehicle.lane
        long, lat = lane.local_coordinates(pos)
        nav = vehicle.navigation
        dest = self._destination_center(nav)
        ego = [
            *((pos - MAP_CENTER) / MAP_HALF),
            *(vel / VEL_SCALE),
            vehicle.speed / VEL_SCALE,
            np.sin(heading),
            np.cos(heading),
            vehicle.steering,
            long / max(lane.length, 1e-6),
            lat / lane.width,
            *lane_relation(lane.index),
            *(ego_frame(dest - pos, heading) / DIST_SCALE),
            nav.route_completion,
        ]
        slots = []
        for v in self.slot_vehicles():
            if v is None:
                slots += [0.0] * len(SLOT_FEATURES)
                continue
            p, u = np.asarray(v.position, dtype=float), np.asarray(v.velocity, dtype=float)
            dh = v.heading_theta - heading
            route = [0.0] * 4
            if v.navigation is not None and v.navigation.checkpoints:
                d = EXIT_DIRECTION.get(v.navigation.checkpoints[-1])
                if d is not None:
                    route[d] = 1.0
            slots += [
                1.0,
                *(ego_frame(p - pos, heading) / POS_SCALE),
                *(ego_frame(u - vel, heading) / REL_VEL_SCALE),
                v.speed / VEL_SCALE,
                np.sin(dh),
                np.cos(dh),
                np.linalg.norm(p - pos) / DIST_SCALE,
                *lane_relation(v.lane_index),
                float(v.lane_index[:2] == lane.index[:2]),
                *route,
            ]
        self.raw = np.concatenate([self.state_obs.observe(vehicle), np.asarray(ego + slots, dtype=float)])
        return np.clip(self.raw, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _destination_center(nav):
        lane = nav.final_lane
        return np.asarray(lane.position(lane.length, 0.0), dtype=float)

    def schema(self):
        rows = [(f"state_obs[{i}]", "MetaDrive StateObservation (see OBSERVATION_SCHEMA.md)")
                for i in range(self.state_obs.observation_space.shape[0])]
        rows += [(f"ego.{n}", d) for n, d in EGO_FEATURES]
        rows += [(f"slot{k}.{n}", d) for k in range(NUM_SLOTS) for n, d in SLOT_FEATURES]
        return rows


class StaticGoalTrafficFullStateEnv(StaticGoalTrafficEnv):
    """Traffic env with FullStateObservation. With extra_lidar_obs=True the previous experiment's 259-dim
    LiDAR-state observation is also returned in info["lidar_obs"], so LiDAR-trained checkpoints can be evaluated on
    exactly the same simulated episodes in the same process."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(dict(agent_observation=FullStateObservation, extra_lidar_obs=False), allow_add_new_key=True)
        return config

    def __init__(self, config=None):
        super().__init__(config)
        self._lidar_obs = LidarStateObservation(self.config) if self.config["extra_lidar_obs"] else None
        self.want_lidar = True  # evaluation turns this off for groups that do not use the LiDAR observation

    @property
    def full_state_obs(self):
        return next(iter(self.observations.values()))

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._add_lidar(info)
        return obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)
        self._add_lidar(info)
        return obs, reward, terminated, truncated, info

    def _add_lidar(self, info):
        if self._lidar_obs is not None and self.want_lidar:
            info["lidar_obs"] = self._lidar_obs.observe(self.agent)

    def traffic_policies(self):
        """Policy class of every traffic vehicle (validation: must be IDMPolicy)."""
        return {v.name: type(self.engine.get_policy(v.name)).__name__ for v in self.traffic_vehicles()}


def is_idm(policy_name):
    return policy_name == IDMPolicy.__name__


if __name__ == "__main__":
    env = StaticGoalTrafficFullStateEnv(dict(spawn_seed=0, traffic_seed=0))
    try:
        obs, info = env.reset()
        print("observation space:", env.observation_space)
        for i, (name, desc) in enumerate(env.full_state_obs.schema()):
            print(f"{i:3d}  {name:24s} {obs[i]: .3f}  {desc}")
    finally:
        env.close()
