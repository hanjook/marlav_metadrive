# Full-state traffic observation: results

**Question:** does a full-state traffic observation improve the ego policy's ability to reach the fixed destination
safely, compared with the existing LiDAR-based traffic policy?

**Short answer: not measurably.** On the held-out test set (100 episodes, the same for every policy):

| Comparison | Full-state | LiDAR | Paired p |
|---|---|---|---|
| Success | 81 | 77 | 0.58 |
| Vehicle collisions | 13 | 15 | 0.81 |

The validation-selected checkpoint of each seed scored 82, 75 and 81. Both learned policies clearly beat the
constant-throttle script (54) and the no-traffic policy (46).

The full-state policy does behave differently around traffic: it brakes and slows much more. But it does this mainly
by inflating its action noise, it slows inside the junction rather than before it, and it cannot drive at all when
no traffic is present.

## What the full state contains (137 values, all in [-1, 1])

The full schema, with exact indices and normalisation, is in `OBSERVATION_SCHEMA.md`. All values are read every step
from MetaDrive 0.4.3's simulator objects.

| Indices | Group | Contents |
|---|---|---|
| 0–18 | MetaDrive `StateObservation` (unchanged) | road-edge distances, heading error to lane, speed, steering, last action, yaw rate, lateral lane position, next 2 route checkpoints |
| 19–34 | Ground-truth ego (16) | world position, world velocity, speed, sin/cos heading, steering, lane-local longitudinal/lateral position, on approach / in junction / on exit, destination in the ego frame, route completion |
| 35–136 | 6 traffic slots × 17 | presence mask; position and velocity relative to the ego in the ego frame; speed; sin/cos heading difference; distance; on approach / in junction / on exit; same road as ego; IDM route destination (east / north / south / west) |

- **What was removed:** the 240 LiDAR beams. In this map they detect only vehicles; with traffic off, every beam reads
  "no hit" for whole episodes. Every vehicle they could see is described exactly by the traffic slots.
- **What is never included:** reward, `goal_reached`, outcome, or future states.
- **Slot ordering:**
  - At each reset, the vehicles are sorted by spawn location: (spawn road start node, end node, lane index, spawn
    longitude), from `vehicle.config`.
  - A vehicle keeps its slot for the whole episode. When MetaDrive removes it at the end of an exit road, the slot
    stays and becomes all zeros.
  - Spawn points are 10 m apart, so there are no ties and no dependence on dict order.

**Validation before training** (`validate_observation.py` → `results/validation.json`). All checks passed:
- shape matches the declared space;
- no NaN or inf;
- the same seeds give a bit-identical initial observation, and the next 30 steps agree to 1e-8 (Bullet float noise);
- a different traffic seed changes only the traffic features;
- moving or re-speeding one vehicle changes only its slot;
- slots stay stable, and removed slots never return;
- no value ever needed clipping;
- the ego is driven by `EnvInputPolicy`, and all traffic by `IDMPolicy`.

## What stayed the same

All of these match `static_goal_traffic/`:
- MetaDrive 0.4.3, Ray/RLlib 2.2.0, torch PPO, Python 3.10.5, CPU, and the same venv;
- `map="X"`, the randomized west start, and the fixed east destination with the same success region;
- the continuous [steering, throttle/brake] action, the reward and penalties, and the 300-step horizon;
- traffic density 0.2, trigger mode, `random_traffic=True`, 6 IDM vehicles, and the traffic-seed mechanism;
- PPO hyperparameters: `ppo_config()` imported unchanged (4 workers, batch 4000, seed per run). Only the input
  dimension changed (137 instead of 259).

The env reproduces the earlier results exactly. Evaluated here, the no-traffic, LiDAR 1M and script policies score
46/38/16, 63/28/9 and 54/46/0 (success / vehicle collision / off-road), identical to `TRAFFIC_RESULTS.md`
(`results/equivalence_check/`).

## Training

```bash
cd metadrive; PY=~/.venvs/metadrive-static-goal/bin/python
$PY static_goal_traffic_full_state/validate_observation.py
$PY static_goal_traffic_full_state/train_full_state.py --timesteps 20000 --seeds 0 --exp-name smoke_full_state --milestones 8000,16000
$PY static_goal_traffic_full_state/smoke_reload.py <smoke checkpoint>
$PY static_goal_traffic_full_state/train_full_state.py --timesteps 1000000 --seeds 0,1,2
```

- **Smoke test:** 20k steps in 11.6 s. Reward went 3.1 → 7.3, checkpoints saved and reloaded, the reloaded policy drove
  with `EnvInputPolicy`, and 4 IDM cars crossed the junction.
- **Main run:** three seeds (0, 1, 2), trained in parallel to 1,000,000 steps each. Each took 792 s of training time;
  the whole run took 820 s wall clock.
- **Checkpoints:** saved at 252k, 500k, 752k and 1M per seed. The 250k milestone falls at 252k because the batch is
  4,000 steps. The milestone callback is `milestone_callbacks.py`.
- **Curves:** `media/training_curves_by_seed.png`. Training success (sampled actions) stayed around 50–80% for all
  seeds after 300k, without the LiDAR run's deep collapses.

## Evaluation protocol

- **Validation:** seeds 100 and 101, 100 episodes. Used only to choose the checkpoint and action mode.
  - Every saved checkpoint was evaluated with both deterministic and sampled actions: 3 seeds × 4 checkpoints × 2 modes.
  - The LiDAR run went through the same selection, using its saved checkpoints nearest each milestone (240k, 500k,
    760k, 1M).
  - Rule: most successes, then fewest vehicle collisions, then highest reward.
- **Test:** seeds 0 and 1, 100 episodes, run once on the selected policies. These are the same 100 test episodes as
  the LiDAR experiment.
- **Pairing:** the same ego starts and traffic for every policy. Sampling noise is seeded per episode
  (`seed × 1000 + episode`), so traffic-on and traffic-off runs of a stochastic policy draw identical noise.
- **"Meaningful braking"** means the speed dropped more than 2 m/s below its running peak. The last 2 steps of
  collision or off-road episodes are excluded, because the impact itself otherwise counts as braking.
- **"Brake commanded"** means at least 3 consecutive steps with a throttle/brake action below 0.

### Validation (selection data, `media/validation_grid.png`)

Successes out of 100:

| Seed | Mode | 250k | 500k | 750k | 1M |
|---|---|---|---|---|---|
| Full state 0 | deterministic | 8 | 3 | 5 | 7 |
| Full state 0 | sampled | 78 | 68 | 80 | 70 |
| Full state 1 | deterministic | 26 | 11 | 8 | 6 |
| Full state 1 | sampled | 80 | 76 | 76 | 67 |
| Full state 2 | deterministic | 49 | 44 | 5 | 4 |
| Full state 2 | sampled | 49 | 66 | 80 | **86** |
| LiDAR (240k / 500k / 760k / 1M) | deterministic | 55 | 46 | 52 | 52 |
| LiDAR | sampled | **73** | 70 | 60 | 43 |

- **Selected:** full state seed 2 at 1M, sampled; LiDAR at 240k, sampled.
- **Per-seed picks:** seed 0 at 750k, seed 1 at 250k, seed 2 at 1M (all sampled).

### Held-out test (`media/test_outcomes.png`, `results/summary.json`, `results/*.csv`)

| Policy | Success | Vehicle collision | Off-road | Timeout | Mean reward | Time to destination | Min distance to traffic | Braked (episodes) | Brake commanded |
|---|---|---|---|---|---|---|---|---|---|
| No-traffic policy (deterministic) | 46 | 38 | 16 | 0 | 89.6 | 9.6 s | 5.3 m | 6 | 0 |
| Constant-throttle script | 54 | 46 | 0 | 0 | 88.6 | 12.5 s | 5.4 m | 0 | 0 |
| LiDAR, 240k, sampled (selected) | 77 | 15 | 8 | 0 | 107.8 | 14.1 s | 6.0 m | 29 | 17 |
| LiDAR, 240k, deterministic | 62 | 26 | 12 | 0 | 97.8 | – | – | 0 | 4 |
| **Full state, seed 2, 1M, sampled (selected)** | **81** | **13** | 5 | 1 | 108.3 | 17.2 s | 7.3 m | 96 | 83 |
| Full state, seed 2, 1M, deterministic | 6 | 47 | 47 | 0 | 63.7 | – | – | 0 | 0 |
| Full state, seed 0, 750k, sampled | 82 | 13 | 3 | 2 | 109.6 | 18.2 s | 7.1 m | 99 | 83 |
| Full state, seed 1, 250k, sampled | 75 | 21 | 4 | 0 | 104.7 | 18.3 s | 6.8 m | 97 | 52 |
| Full state (selected), **traffic off** | 0 | 0 | 10 | 90 | 69.6 | – | – | 100 | 99 |
| LiDAR (selected), traffic off | 94 | 0 | 6 | 0 | 119.7 | 13.8 s | – | 1 | 0 |

"Min distance to traffic" is the mean, over episodes, of the closest centre-to-centre distance to any traffic vehicle.

Paired exact McNemar tests on the test set:

| Comparison | Success (only A / only B, p) | Vehicle collision (only A / only B, p) |
|---|---|---|
| Full state vs LiDAR | 17 / 13, p = 0.58 | 8 / 10, p = 0.81 |
| Full state vs script | 37 / 10, p < 0.001 | 7 / 40, p < 0.001 |
| Full state vs no-traffic policy | 44 / 9, p < 0.001 | 10 / 35, p < 0.001 |
| LiDAR vs script | 33 / 10, p < 0.001 | 8 / 39, p < 0.001 |

## Interaction analysis (test set)

**Traffic on vs traffic off, same starts and same sampling noise** (`results/summary.json` → `interaction_test_set`):

| | Full state (selected) | LiDAR (selected) |
|---|---|---|
| Success, traffic on / off | 81 / **0** | 77 / 94 |
| Vehicle collisions, on / off | 13 / 0 | 15 / 0 |
| Median max trajectory difference | 40.8 m | 6.7 m |
| Mean absolute throttle / steering difference | 0.37 / 0.35 | 0.05 / 0.14 |
| Brake commanded, on / off (episodes) | 83 / 99 | 17 / 0 |
| Slowed > 2 m/s before the junction, on / off | 28 / 0 | 0 / 0 |
| Junction entry speed, on minus off | −4.2 m/s | +0.1 m/s |

**Where each policy drives below 3 m/s** (after the standing start; `where_slow.py` → `results/where_slow.json`):

| | Approach | Junction | Exit road |
|---|---|---|---|
| Full state, traffic on | 266 / 5,048 steps | **3,453 / 7,697 steps (86 episodes)** | 4 / 2,186 |
| LiDAR, traffic on | 484 / 6,305 | 232 / 3,924 (17 episodes) | 0 / 2,132 |
| Full state, traffic off | 103 / 4,500 | 20,742 / 23,282 (98 episodes) | 0 / 0 |

**How the full-state policy slows down** (`action_distribution.py` → `results/action_distribution.json`):
- **The mean action is always "go".** Its Gaussian mean throttle averages 1.2–1.8, above the action bound.
- **It slows by inflating its noise.** The throttle std exceeds 1 on 21–35% of steps, rising to 43–65% of steps when
  the car is below 3 m/s, so clipped samples become a coin flip between full throttle and full brake. The LiDAR
  policy does this on 5% of steps.
- **Consequences:** this is why the full-state policies fail almost completely with deterministic (mean) actions
  (3–49 validation successes; 6/100 on test), and why the selected policy commands braking in 83/100 episodes, against
  17 for LiDAR.

## Findings

1. **Did full state improve success?** No significant improvement. On test, the selected full-state policy scored 81
   against the LiDAR policy's 77 (p = 0.58). The per-seed selected checkpoints scored 82, 75 and 81. The validation
   gap (86 vs 73) shrank on test, as expected when choosing the best of 12 sampled policies.
2. **Did the collision rate decrease?** No significant decrease: 13 vs 15 vehicle collisions (p = 0.81). The
   full-state policy keeps slightly more distance (mean closest approach 7.3 m vs 6.0 m) and takes longer (17.2 s vs
   14.1 s).
3. **Did it learn to slow down or yield?** It slows around traffic, but not in a reliable, yielding way.
   - **Before the junction:** it slowed by more than 2 m/s in 28/100 episodes with traffic, against 0 without; the
     LiDAR policy, 0 either way.
   - **Inside the junction:** most of its slow driving happens here (86 episodes), not at the entry.
   - **The mechanism:** it slows by injecting action noise, not by commanding a lower mean.
   - **Without traffic:** it stops in the junction and times out in 90/100 episodes. It never saw an empty road in
     training, so it depends on traffic features to drive at all. The traffic-off comparison therefore measures
     out-of-distribution behaviour as much as the effect of traffic.
4. **Sensitivity.**
   - **Action mode dominates.** The sampled policies range from 49 to 86 validation successes, the deterministic ones
     from 3 to 49.
   - **The best checkpoint differs by seed** (750k, 250k, 1M), so the final checkpoint is not reliably best.
   - **Selection is noisy.** An earlier validation pass with a different noise seeding picked seed 0 at 750k
     (82, 10 collisions) instead of seed 2 at 1M. The top policies differ by only a few episodes in 100.
   - **LiDAR:** the earlier "final" 1M deterministic policy (63 on test) was not its best; 240k sampled scores 77.

## Correction to the previous experiment

`static_goal_traffic/TRAFFIC_RESULTS.md` says the LiDAR policy "braked by more than 2 m/s in 20/100 episodes with
traffic, against 0 without". That metric counted the speed drop at the moment of impact: all 20 of those episodes
were vehicle collisions. In fact, that policy (1M, deterministic) never commands a brake, and with the impact step
excluded it never slows by more than 1.8 m/s.

The folder was left unmodified as instructed. This experiment uses the corrected metric.

## Recordings and outputs (`media/`)

All recordings replay exact test episodes, and each replayed outcome is checked against the evaluation file.

| File | Content |
|---|---|
| `success.gif` | Full-state policy succeeds (seed 0, episode 7) |
| `failure.gif` | Full-state policy, vehicle collision in the junction (seed 0, episode 2); its most common failure |
| `lidar_vs_full_state.gif` / `.mp4` | Same episode: LiDAR collides (left), full state succeeds (right). Picked for contrast, so it is illustrative, not typical |
| `traffic_on_vs_off.gif` / `.mp4` | Same start: full state succeeds with traffic (left), and stalls in the junction and times out without traffic (right) |
| `training_curves_by_seed.png`, `validation_grid.png`, `test_outcomes.png` | Charts |

- **Data:** `results/summary.json`, `results/summary_by_group.csv` (every group, both sets), `results/test_episodes.csv`
  (per episode), `results/selection.json`, `results/validation/` and `results/test/` (per-group JSON and trajectories).
- **Checkpoints:** `runs/full_state_ppo/PPO_*` (seed in `params.json`), at iterations 63, 125, 188 and 250.

```bash
$PY static_goal_traffic_full_state/evaluate_full_state.py run --set validation --groups "$($PY static_goal_traffic_full_state/evaluate_full_state.py grid)"
$PY static_goal_traffic_full_state/evaluate_full_state.py select
$PY static_goal_traffic_full_state/evaluate_full_state.py test
$PY static_goal_traffic_full_state/evaluate_full_state.py report
$PY static_goal_traffic_full_state/record_full_state.py
$PY static_goal_traffic_full_state/make_full_state_charts.py
```

## Limitations

- **The degenerate noise mechanism.** PPO's Gaussian std is unbounded and actions are clipped, so the policy can
  control speed through its noise. This affects both observation types, but the full-state policy relies on it much
  more. Comparisons therefore depend on the action mode, and the mean action is not a usable controller.
- **No empty-road training.** Every training episode starts with 6 vehicles, so the policy is out of distribution
  without traffic.
- **Simple traffic.** Trigger mode, always exactly 6 cars, no traffic lights or right-of-way rules, and IDM cars that
  react only to vehicles ahead in their lane. The reward is unchanged, so nothing rewards waiting or penalises
  creeping in the junction.
- **Limited evidence.** Three training seeds and 100 test episodes cannot resolve a 4-point success difference; the
  LiDAR baseline has one training seed.
- **Inherited from earlier experiments:** a provisional built-in map, a straight route, arrival at speed rather than
  stopping, and timeouts treated as terminal.

## Recommended next experiment

Keep the full-state observation, and remove the two confounds that most affect it:

1. **Bound the action distribution:** squash with tanh, cap the log-std, or add a small entropy penalty, so that
   speed must be controlled through the mean action.
2. **Randomize the traffic count per episode, including 0 vehicles,** so the policy also learns to drive on an empty
   road.

Then compare full state and LiDAR again with 3 seeds each and the same validation/test protocol. Only after that,
consider reward changes such as a larger `crash_vehicle_penalty` or a time penalty inside the junction.
