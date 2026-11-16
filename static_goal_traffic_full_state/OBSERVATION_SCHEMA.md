# Full-state observation schema

Dimension: **137**. Gym space: `Box(-1.0, 1.0, (137,), float32)`. Every value is clipped to [-1, 1]; the
"clipped" column is the share of validation steps on which the raw value fell outside that range.

Built by `FullStateObservation` in `full_state_env.py` from MetaDrive 0.4.3 simulator state, every step:
vehicle `position`, `velocity`, `speed`, `heading_theta`, `steering`, `navigation` (current lane, route
checkpoints, final lane, `route_completion`) for the ego, and the same objects for each traffic vehicle,
looked up by name in `engine.get_objects()`.

**Ego frame:** forward = along the ego's heading, left = 90 degrees counter-clockwise from it.

**Traffic slot order:** at each reset the traffic vehicles are sorted by their spawn location
`(spawn road start node, spawn road end node, lane index, spawn longitude)` from each vehicle's
`config["spawn_lane_index"]` and `config["spawn_longitude"]`. A vehicle keeps its slot for the whole
episode. When MetaDrive's traffic manager removes a vehicle (it drove off the end of an exit road), its slot
stays and every value in it, including `present`, becomes 0. Spawn points are 10 m apart, so there are no ties.
With density 0.2 there are always exactly 6 traffic vehicles at reset.

**Not included:** reward, `goal_reached`, episode outcome, future states, or anything computed from them.
The traffic `route_*` flags are the IDM driver's current route plan (its navigation destination), which is
part of the simulator state at that moment.

**Removed relative to the previous LiDAR experiment:** the 240 LiDAR beams. In this map they detect only
vehicles (validation check: with traffic off, every beam reads 1.0 = nothing detected for a whole episode),
and every vehicle is described exactly by the traffic slots.

| Index | Feature | Meaning | Clipped |
|---|---|---|---|
| 0 | `state[0]` | distance to left road edge / total road width (MetaDrive dist_to_left_side) | 0.0% |
| 1 | `state[1]` | distance to right road edge / total road width (MetaDrive dist_to_right_side) | 0.0% |
| 2 | `state[2]` | heading difference to current reference lane (MetaDrive heading_diff) | 0.0% |
| 3 | `state[3]` | (speed km/h + 1) / (max speed km/h + 1) | 0.0% |
| 4 | `state[4]` | (steering / MAX_STEERING + 1) / 2 | 0.0% |
| 5 | `state[5]` | last throttle/brake action, (a + 1) / 2 | 0.0% |
| 6 | `state[6]` | last steering action, (a + 1) / 2 | 0.0% |
| 7 | `state[7]` | yaw rate, clipped to [0, 1] | 0.0% |
| 8 | `state[8]` | lateral position on current lane, (2 * lat / max lane width + 1) / 2 | 0.0% |
| 9 | `state[9]` | checkpoint 1: forward offset (ego frame), normalised | 0.0% |
| 10 | `state[10]` | checkpoint 1: side offset (ego frame), normalised | 0.0% |
| 11 | `state[11]` | checkpoint 1: bend radius of the lane ending at the checkpoint | 0.0% |
| 12 | `state[12]` | checkpoint 1: bend direction (clockwise 1) | 0.0% |
| 13 | `state[13]` | checkpoint 1: bend angle | 0.0% |
| 14 | `state[14]` | checkpoint 2: forward offset (ego frame), normalised | 0.0% |
| 15 | `state[15]` | checkpoint 2: side offset (ego frame), normalised | 0.0% |
| 16 | `state[16]` | checkpoint 2: bend radius of the lane ending at the checkpoint | 0.0% |
| 17 | `state[17]` | checkpoint 2: bend direction (clockwise 1) | 0.0% |
| 18 | `state[18]` | checkpoint 2: bend angle | 0.0% |
| 19 | `ego.x` | world x, (x - 65) / 80 | 0.0% |
| 20 | `ego.y` | world y, (y - 10) / 70 | 0.0% |
| 21 | `ego.vx` | world velocity x / 30 m/s | 0.0% |
| 22 | `ego.vy` | world velocity y / 30 m/s | 0.0% |
| 23 | `ego.speed` | speed / 30 m/s | 0.0% |
| 24 | `ego.sin_heading` | sin(heading) | 0.0% |
| 25 | `ego.cos_heading` | cos(heading) | 0.0% |
| 26 | `ego.steering` | current steering command, in [-1, 1] | 0.0% |
| 27 | `ego.lane_long` | longitudinal position on current lane / lane length | 0.0% |
| 28 | `ego.lane_lat` | lateral offset from current lane centre / lane width | 0.0% |
| 29 | `ego.on_approach` | 1 if on a road leading into the junction | 0.0% |
| 30 | `ego.in_junction` | 1 if on a junction lane | 0.0% |
| 31 | `ego.on_exit` | 1 if on a road leaving the junction | 0.0% |
| 32 | `ego.dest_forward` | destination centre, ego-frame forward offset / 150 m | 0.0% |
| 33 | `ego.dest_left` | destination centre, ego-frame left offset / 150 m | 0.0% |
| 34 | `ego.route_completion` | MetaDrive navigation.route_completion, clipped to [0, 1] | 0.0% |
| 35 | `slot0.present` | 1 if the vehicle exists in the simulator, else 0 (all other features 0) | 0.0% |
| 36 | `slot0.rel_forward` | relative position, ego-frame forward / 150 m | 0.0% |
| 37 | `slot0.rel_left` | relative position, ego-frame left / 150 m | 0.0% |
| 38 | `slot0.rel_vforward` | relative velocity (theirs - ego), ego-frame forward / 40 m/s | 0.0% |
| 39 | `slot0.rel_vleft` | relative velocity (theirs - ego), ego-frame left / 40 m/s | 0.0% |
| 40 | `slot0.speed` | their speed / 30 m/s | 0.0% |
| 41 | `slot0.sin_dheading` | sin(their heading - ego heading) | 0.0% |
| 42 | `slot0.cos_dheading` | cos(their heading - ego heading) | 0.0% |
| 43 | `slot0.distance` | centre distance to ego / 150 m | 0.0% |
| 44 | `slot0.on_approach` | 1 if on a road leading into the junction | 0.0% |
| 45 | `slot0.in_junction` | 1 if on a junction lane | 0.0% |
| 46 | `slot0.on_exit` | 1 if on a road leaving the junction | 0.0% |
| 47 | `slot0.same_road_as_ego` | 1 if on the same road (same start and end node) as the ego | 0.0% |
| 48 | `slot0.route_east` | 1 if its IDM route ends at the east exit | 0.0% |
| 49 | `slot0.route_north` | 1 if its IDM route ends at the north exit | 0.0% |
| 50 | `slot0.route_south` | 1 if its IDM route ends at the south exit | 0.0% |
| 51 | `slot0.route_west` | 1 if its IDM route ends at the west exit | 0.0% |
| 52 | `slot1.present` | as slot0 | 0.0% |
| 53 | `slot1.rel_forward` | as slot0 | 0.0% |
| 54 | `slot1.rel_left` | as slot0 | 0.0% |
| 55 | `slot1.rel_vforward` | as slot0 | 0.0% |
| 56 | `slot1.rel_vleft` | as slot0 | 0.0% |
| 57 | `slot1.speed` | as slot0 | 0.0% |
| 58 | `slot1.sin_dheading` | as slot0 | 0.0% |
| 59 | `slot1.cos_dheading` | as slot0 | 0.0% |
| 60 | `slot1.distance` | as slot0 | 0.0% |
| 61 | `slot1.on_approach` | as slot0 | 0.0% |
| 62 | `slot1.in_junction` | as slot0 | 0.0% |
| 63 | `slot1.on_exit` | as slot0 | 0.0% |
| 64 | `slot1.same_road_as_ego` | as slot0 | 0.0% |
| 65 | `slot1.route_east` | as slot0 | 0.0% |
| 66 | `slot1.route_north` | as slot0 | 0.0% |
| 67 | `slot1.route_south` | as slot0 | 0.0% |
| 68 | `slot1.route_west` | as slot0 | 0.0% |
| 69 | `slot2.present` | as slot0 | 0.0% |
| 70 | `slot2.rel_forward` | as slot0 | 0.0% |
| 71 | `slot2.rel_left` | as slot0 | 0.0% |
| 72 | `slot2.rel_vforward` | as slot0 | 0.0% |
| 73 | `slot2.rel_vleft` | as slot0 | 0.0% |
| 74 | `slot2.speed` | as slot0 | 0.0% |
| 75 | `slot2.sin_dheading` | as slot0 | 0.0% |
| 76 | `slot2.cos_dheading` | as slot0 | 0.0% |
| 77 | `slot2.distance` | as slot0 | 0.0% |
| 78 | `slot2.on_approach` | as slot0 | 0.0% |
| 79 | `slot2.in_junction` | as slot0 | 0.0% |
| 80 | `slot2.on_exit` | as slot0 | 0.0% |
| 81 | `slot2.same_road_as_ego` | as slot0 | 0.0% |
| 82 | `slot2.route_east` | as slot0 | 0.0% |
| 83 | `slot2.route_north` | as slot0 | 0.0% |
| 84 | `slot2.route_south` | as slot0 | 0.0% |
| 85 | `slot2.route_west` | as slot0 | 0.0% |
| 86 | `slot3.present` | as slot0 | 0.0% |
| 87 | `slot3.rel_forward` | as slot0 | 0.0% |
| 88 | `slot3.rel_left` | as slot0 | 0.0% |
| 89 | `slot3.rel_vforward` | as slot0 | 0.0% |
| 90 | `slot3.rel_vleft` | as slot0 | 0.0% |
| 91 | `slot3.speed` | as slot0 | 0.0% |
| 92 | `slot3.sin_dheading` | as slot0 | 0.0% |
| 93 | `slot3.cos_dheading` | as slot0 | 0.0% |
| 94 | `slot3.distance` | as slot0 | 0.0% |
| 95 | `slot3.on_approach` | as slot0 | 0.0% |
| 96 | `slot3.in_junction` | as slot0 | 0.0% |
| 97 | `slot3.on_exit` | as slot0 | 0.0% |
| 98 | `slot3.same_road_as_ego` | as slot0 | 0.0% |
| 99 | `slot3.route_east` | as slot0 | 0.0% |
| 100 | `slot3.route_north` | as slot0 | 0.0% |
| 101 | `slot3.route_south` | as slot0 | 0.0% |
| 102 | `slot3.route_west` | as slot0 | 0.0% |
| 103 | `slot4.present` | as slot0 | 0.0% |
| 104 | `slot4.rel_forward` | as slot0 | 0.0% |
| 105 | `slot4.rel_left` | as slot0 | 0.0% |
| 106 | `slot4.rel_vforward` | as slot0 | 0.0% |
| 107 | `slot4.rel_vleft` | as slot0 | 0.0% |
| 108 | `slot4.speed` | as slot0 | 0.0% |
| 109 | `slot4.sin_dheading` | as slot0 | 0.0% |
| 110 | `slot4.cos_dheading` | as slot0 | 0.0% |
| 111 | `slot4.distance` | as slot0 | 0.0% |
| 112 | `slot4.on_approach` | as slot0 | 0.0% |
| 113 | `slot4.in_junction` | as slot0 | 0.0% |
| 114 | `slot4.on_exit` | as slot0 | 0.0% |
| 115 | `slot4.same_road_as_ego` | as slot0 | 0.0% |
| 116 | `slot4.route_east` | as slot0 | 0.0% |
| 117 | `slot4.route_north` | as slot0 | 0.0% |
| 118 | `slot4.route_south` | as slot0 | 0.0% |
| 119 | `slot4.route_west` | as slot0 | 0.0% |
| 120 | `slot5.present` | as slot0 | 0.0% |
| 121 | `slot5.rel_forward` | as slot0 | 0.0% |
| 122 | `slot5.rel_left` | as slot0 | 0.0% |
| 123 | `slot5.rel_vforward` | as slot0 | 0.0% |
| 124 | `slot5.rel_vleft` | as slot0 | 0.0% |
| 125 | `slot5.speed` | as slot0 | 0.0% |
| 126 | `slot5.sin_dheading` | as slot0 | 0.0% |
| 127 | `slot5.cos_dheading` | as slot0 | 0.0% |
| 128 | `slot5.distance` | as slot0 | 0.0% |
| 129 | `slot5.on_approach` | as slot0 | 0.0% |
| 130 | `slot5.in_junction` | as slot0 | 0.0% |
| 131 | `slot5.on_exit` | as slot0 | 0.0% |
| 132 | `slot5.same_road_as_ego` | as slot0 | 0.0% |
| 133 | `slot5.route_east` | as slot0 | 0.0% |
| 134 | `slot5.route_north` | as slot0 | 0.0% |
| 135 | `slot5.route_south` | as slot0 | 0.0% |
| 136 | `slot5.route_west` | as slot0 | 0.0% |

Groups: indices 0-18 MetaDrive `StateObservation` (unchanged from the previous experiment: ego state
9 + two route checkpoints 10), 19-34 ground-truth ego features, 35-136 six traffic slots of 17 values
(slot k starts at index 35 + 17 k).
