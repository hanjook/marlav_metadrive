# Static-goal baseline with background traffic

**Question:** can the policy reach the same static destination when the world contains moving vehicles?

**Short answer: partly.** The traffic-trained deterministic policy reaches the destination in 63 of 100 episodes, against
100 of 100 on the same starts with traffic switched off. It is clearly better than the no-traffic policy (46/100,
paired p = 0.01). It is not clearly better than a constant-throttle script (54/100, p = 0.29). Collisions with
vehicles are the main failure.

## What changed from the original baseline

| | Original (`static_goal/`) | Traffic (`static_goal_traffic/`) |
|---|---|---|
| Map, start, destination | `map="X"`; random lane (3) and 0–20 m start on the west approach; fixed destination `1X1_1_` (east exit) | **same** |
| Success region | x 117.5–127.5 m, y −1.75–8.75 m | **same** (checked on every inspected reset) |
| Observation, action, reward | MetaDrive defaults (259-dim lidar+state; [steer, throttle]; progress + speed, +10 / −5) | **same** |
| PPO | official example settings, torch, seed 0, 4 workers, batch 4000 | **same** (`ppo_config()` imported unchanged) |
| Traffic | none | 6 IDM vehicles per episode (MetaDrive `PGTrafficManager` + `IDMPolicy`) |
| Ego controller | `EnvInputPolicy` | `EnvInputPolicy`, checked on every evaluated step |

The original code, checkpoint, results and meeting media are unchanged. Everything new lives in `static_goal_traffic/`.

## Environment configuration

`StaticGoalTrafficEnv` (in `traffic_env.py`) subclasses `StaticGoalIntersectionEnv` and adds:

```python
traffic_density=0.2       # 0.1 -> 3 vehicles, 0.15 -> 4, 0.2 -> 6
traffic_mode="trigger"    # MetaDrive default
random_traffic=True       # do not reseed traffic from the single map seed every reset
traffic_seed=None         # int -> reproducible per-episode traffic seeds
```

Everything else is inherited: `map="X"`, `num_scenarios=1`, `horizon=300` (30 s), `destination="1X1_1_"`, and the
`spawn_seed` start sequence.

Before each reset, the env draws a per-episode traffic seed and seeds the traffic manager with it. The seed goes
through `Randomizable.seed`, because `PGTrafficManager.seed()` does nothing when `random_traffic=True`. Each episode
records `traffic_seed` and `num_traffic_vehicles`.

**Why trigger mode.**
- In trigger mode, vehicles are placed inside the junction, on the exit roads and on the three other incoming arms.
  They start driving when the ego reaches x = 10 m, and each picks a random exit.
- `respawn` mode was rejected. It also spawns cars on the ego's own approach road; one probe episode ended in a
  collision at step 1. It also gave the same 12 cars at every density.

**Inspection before training** (`inspect_traffic.py` → `media/inspect_resets.png`, `results/inspect_traffic.json`).
These are 6 seeded resets with a constant-throttle ego:
- every reset had 6 vehicles;
- 5–6 of them moved, and 5–6 drove through the junction;
- the ego and traffic were in the junction together for 12–32 steps;
- the closest approach was 3.5–10.5 m;
- 2 of 6 episodes ended in a vehicle collision;
- the success region was unchanged, and the ego was always `EnvInputPolicy`.

## Training

```bash
cd metadrive; PY=~/.venvs/metadrive-static-goal/bin/python
$PY static_goal_traffic/train_traffic.py --timesteps 20000 --exp-name smoke_traffic --checkpoint-freq 2   # smoke test
$PY static_goal_traffic/train_traffic.py --timesteps 500000                                               # main run
$PY static_goal_traffic/train_traffic.py --timesteps 1000000 --restore <run>/checkpoint_000125            # extended
```

- **Smoke test:** 20k steps in 9.7 s. Reward rose from 3.0 to 7.1, and checkpoints saved.
- **Main run:** 500k steps in 289 s of training time (304 s wall). Training success averaged 0.72, 0.79 and 0.81 over
  the last three 100k windows, so it was still improving and was extended.
- **Extension:** 500k → 1M took another 290 s of training time (305 s wall).
- **Totals:** 1,000,000 steps and 250 iterations, in 579 s of training time (about 10 min wall), at about 1,700
  steps/s, half the no-traffic speed.
- **Checkpoints:** every 10 iterations in `runs/static_goal_traffic_ppo/`. Copies are `checkpoint_final` (1M) and
  `checkpoint_500k`.
- **After 500k:** training success dropped (window averages 0.68, 0.33, 0.57, 0.43, 0.58) while policy entropy rose
  from 1.9 to 2.7. This is the same pattern as the no-traffic run (`media/training_curve.png`).

## Evaluation

```bash
$PY static_goal_traffic/evaluate_traffic.py --episodes 50 --seed 0      # and --seed 1
$PY static_goal_traffic/evaluate_traffic.py --episodes 50 --seed 0 --traffic-checkpoint static_goal_traffic/checkpoint_500k --groups traffic_det,traffic_stoch --tag _ckpt500k
```

- **Episodes:** 100 per group, made of 50 with spawn and traffic seed 0 plus 50 with seed 1. Every group sees the
  same 100 situations.
- **Traffic:** density 0.2 in trigger mode, with 6 vehicles in every episode. The count is fixed by density × road
  length; only positions, vehicle types and routes vary.
- **Columns:** "Vehicle collision" is MetaDrive's `crash_vehicle`. "Off-road" is `out_of_road`, `crash_sidewalk` or
  `crash_building`. No other crash types occurred.
- **Time to destination** is averaged over successful episodes only.

| Group | Success (seed 0 + seed 1) | Vehicle collision | Off-road | Timeout | Mean reward | Time to destination (s) |
|---|---|---|---|---|---|---|
| No-traffic policy (original checkpoint), deterministic | **46** (25 + 21) | 38 | 16 | 0 | 89.7 | 9.6 |
| Traffic-trained 1M (final), deterministic | **63** (34 + 29) | 28 | 9 | 0 | 97.6 | 9.5 |
| Traffic-trained 1M (final), stochastic | **41** (19 + 22) | 17 | 40 | 2 | 87.1 | 16.2 |
| Traffic-trained 500k, deterministic | **44** (23 + 21) | 46 | 10 | 0 | 85.8 | 9.3 |
| Traffic-trained 500k, stochastic | **68** (32 + 36) | 9 | 23 | 0 | 106.4 | 14.7 |
| Constant-throttle script (steer 0, throttle 0.5) | **54** (23 + 31) | 46 | 0 | 0 | 88.6 | 12.5 |
| *Reference: traffic-trained 1M, deterministic, traffic off* | *100* | *0* | *0* | *0* | *120.8* | *9.2* |

Paired exact McNemar tests on the same 100 episodes:

| Comparison | p |
|---|---|
| Final deterministic vs no-traffic policy | 0.01 |
| Final deterministic vs script | 0.29 |
| Final deterministic vs final stochastic | 0.003 |
| 500k stochastic vs script | 0.06 |
| 500k stochastic vs final deterministic | 0.57 |

**Deterministic vs stochastic.**
- **The ranking flips between checkpoints.** At 1M, the mean action is best (63) and sampling mostly drives off the
  road (40 off-road). At 500k, sampling is best (68) and the mean action collides often (46).
- **Sampled policies are slower** (14.7–16.2 s against about 9.4 s) and collide less. The extra braking and noise
  seem to help with traffic, at the cost of more off-road episodes.

**Is traffic really affecting the ego?** Yes. The final policy was run on the same starts with traffic switched off,
and the paired runs show:
- every trajectory diverged by more than 1 m (median maximum divergence 3.7 m for seed 0, 5.0 m for seed 1);
- the ego braked by more than 2 m/s in 20/100 episodes with traffic, against 0/100 without;
- the outcome changed in 37/100 episodes;
- in the empty world the same policy succeeds 100/100.

**Recordings** (`record_traffic.py` → `media/`). They replay exact evaluation episodes, and the script checks that
each outcome matches the evaluation file.
- `success.gif`: the traffic-trained policy succeeds (episode 0, seed 0).
- `failure.gif`: its most common failure, a vehicle collision in the junction (episode 1).
- `side_by_side.gif` / `.mp4`: same start and traffic. The no-traffic policy goes off-road; the traffic-trained policy
  succeeds. This episode was picked for contrast, so it illustrates rather than represents.
- `side_by_side_failure_episode.gif`: the reverse case, where the original policy succeeds and the traffic-trained
  policy collides.
- Charts: `media/evaluation_results.png` (seed 0) and `media/training_curve.png`.

## Main finding

The policy can usually reach the fixed destination with moving traffic, but not reliably. Its best success rate is
63–68%, against 100% without traffic. Training with traffic clearly helps compared with reusing the no-traffic
policy (63 vs 46).

The policy has not learned to yield. Deterministic runs arrive as fast as in an empty world (9.5 s vs 9.2 s). It
changes its path and sometimes brakes, but too little and too rarely, so it barely beats a constant-throttle script.

## Limitations

- **The observation cannot show other cars' speed.** It is a single-frame lidar distance scan; the default lidar has
  `num_others=0` and there is no frame stack.
- **The reward does not encourage waiting.** It is unchanged: a collision costs −5 against +10 for arrival, plus
  progress and speed terms. A fast policy that sometimes crashes scores about as well as a careful one.
- **The traffic is simple.** Trigger mode always gives 6 cars, which stay parked until the ego reaches x = 10 m and
  are never refilled. There is no traffic on the ego's own approach, and no lights or right-of-way rules. IDM cars
  react only to vehicles ahead in their own lane.
- **Results are sensitive to the checkpoint.** It was one training seed, training success is unstable after about
  500k steps (entropy rises), and the deterministic/stochastic ranking flips between 500k and 1M. The difference
  against the script is not significant.
- **Carried over from the original baseline:** a provisional map (MetaDrive's own intersection), a straight route,
  arrival at speed rather than stopping, and timeouts treated as terminal.

## Recommended next experiment

Give the policy the information it needs to yield, and change nothing else:

1. Set MetaDrive's documented lidar option `vehicle_config["lidar"]["num_others"] = 4`. This adds the relative
   position and velocity of the 4 nearest vehicles to the observation.
2. Retrain at density 0.2 with 3 PPO seeds.
3. Select checkpoints by deterministic evaluation on the same 100 seeded episodes.

If the success rate stays near the script's, the next lever is the reward, for example a larger
`crash_vehicle_penalty`. Capping or decaying the action std would address the entropy rise seen in both runs.
