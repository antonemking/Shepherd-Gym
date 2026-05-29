# Course grounding spec — the real Shepherd-v0 MDP

> This is the STEP 1 grounding report. Everything in the three lessons is built on
> the values below, read directly from [`shepherd_gym/env.py`](../shepherd_gym/env.py)
> (the `ShepherdConfig` dataclass and `ShepherdEnv` class). Dimensions were verified by
> instantiating the env: **obs_dim = 38, state_dim (privileged) = 45, action_dim = 2.**
> Where a lesson uses a simplification (the L2 gridworld), it says so explicitly.

## The agent and the world
- The agent is **the dog**. It herds `n_sheep` boid-sheep into a circular **fold/pen**.
- Default config (`ShepherdConfig`, also the `t1_paddock` baseline `BASE`):
  `n_sheep=8`, `world=20.0` m square, `pen_center=(16,16)`, `pen_radius=3.0`,
  `dt=0.1`, `max_steps=600`, `k_nearest=5`.
- Speeds: `dog_speed=2.4`, `sheep_max_speed=1.9` (m/step-unit).

## Action space — continuous, 2-D
`action_dim = 2`, `a ∈ [-1, 1]²` (clipped in `step`). It's a desired velocity
direction: `dog_vel = a * dog_speed`; `dog_pos += dog_vel * dt` (clipped to field).
The dog steers — it does not teleport, and it never pushes sheep directly.

## Observation space — 38-D actor observation (`_obs()`), realistically sensable only
| idx | content | normaliser |
|---|---|---|
| 0:2 | `dog_pos` | `/world` |
| 2:4 | `dog_vel` | `/dog_speed` |
| 4:6 | `centroid − dog_pos` (free-sheep centroid) | `/world` |
| 6 | `flock_spread` (mean dist to centroid) | `/world` |
| 7 | `penned.mean()` (fraction folded) | — |
| 8 | **`mean_ear`** — noisy aggregate ear angle | `/ear_norm` |
| 9:11 | `pen − dog_pos` | `/world` |
| 11:13 | `pen − centroid` | `/world` |
| 13:38 | **k_nearest=5** nearest free sheep, each: `(pos−dog)/world` [2], `vel/sheep_max_speed` [2], `ear/ear_norm` [1] |

The **ear angle** (`_ear_angles()`) is the dog's only stress proxy:
`base = ear_neutral_deg + (ear_aroused_deg − ear_neutral_deg)·arousal + N(0, ear_noise_deg)`,
with `ear_neutral_deg=25` (calm, ears up), `ear_aroused_deg=−10` (aroused, ears back),
`ear_noise_deg=4.0`, `ear_norm=35`. **It is a noisy readout of the hidden arousal, not arousal itself.**

## Privileged critic state — 45-D (`privileged_state()`)
The 38-D obs **plus 7 sim-only truths**: `[mean_arousal, max_arousal]` over free sheep,
and the **true arousal of each of the k_nearest=5 sheep**. Used only at train time to
reduce value-estimate variance (asymmetric actor-critic).

## Reward (`step`, computed on the TRUE latent arousal)
```
reward =  r_pen·newly            # +1.0 per sheep that JUST entered the fold
        + r_progress·progress    # 0.0 — OFF (superseded by fetch)
        + r_fetch·fetch          # +0.15 × Σ per-sheep decrease in distance-to-pen (potential shaping)
        − r_spread·spread        # −0.02 × mean flock spread
        − r_arousal·mean_ar·dt   # WELFARE: −0.5 × mean(true arousal of free sheep) × dt(0.1)
        − r_time                 # −0.01 per step
   (+ r_success = +10.0 once penned.all())
```
**How "calm" is encoded:** `r_arousal = 0.5` (tunable; the `RESULTS.md` Pareto study sweeps
0 / 0.25 / 0.5). It is a **per-step integrated dose** (`−0.5·mean_arousal·dt`), charged on the
**hidden true arousal** — the dog is graded on welfare it can't directly see. The competition
`composite` score (`scoring.py`) additionally makes welfare first-class:
`SCORE = 100·(0.50·penned + 0.35·welfare + 0.15·speed)`, `welfare = 1 − clip(mean_arousal)`,
**speed gated on full success** so "stall to stay calm" loses.

## Arousal dynamics (`_update_arousal`) — flight-zone grounded
Per sheep: `intrusion = clip((flight_zone − dist)/flight_zone, 0,1)**intrusion_exp`
(`flight_zone=7.0`, `intrusion_exp=1.8`); `closing = max(0, dir·dog_vel)/dog_speed`
(`w_closing=1.0`); `pred_pressure = (w_pred_arousal=1.3 + w_closing·closing)·intrusion`;
`isolation = clip((nearest_nbr − social_dist)/social_dist, 0,1)` (`w_isolation=0.9`,
`social_dist=3.0`). Update: `arousal += (drive·arousal_rise − arousal_decay)·dt`,
**asymmetric** (`arousal_rise=1.6` fast up, `arousal_decay=0.22` slow down); clipped [0,1].

## Transition dynamics — boids + flee (`_step_flock`)
Force = `w_sep·1.8`(separation) + `w_ali·0.5`(alignment) + `w_coh·0.6`(cohesion)
+ `w_flee·2.6·flee` + `w_bounds·2.0`(walls). **Flee** only inside `flight_zone`:
`flee = unit(sheep−dog)·(1 − dist/flight_zone)`. Speed-capped; penned sheep are contained.

## Policy representation & running episodes
- **A policy is a callable `policy(env) -> action ∈ [-1,1]²`** (`baselines.py`, `adapter.py`).
- Scripted baselines: `RandomPolicy`, `GreedyDriver` (push-from-behind; scatters loose flocks),
  `FlankShepherd` (Strömbom collect-then-drive expert).
- Learned: `scripts/train.py` — asymmetric PPO; `Actor` = 2×128 Tanh MLP → Gaussian
  (`log_std`); `Critic` reads the 45-D privileged state. **γ=0.99, GAE λ=0.95**, clip 0.2.
- Run/render: `render.record_episode(env, policy, seed=…)` returns PIL frames (sheep tinted
  calm-white→stress-red); `env.render_state()` exposes `dog/sheep/arousal/penned/pen`.
- Real headline (`results/RESULTS.md`): flank 75% pen / 0.459 arousal; PPO `r=0.25`
  63% / **0.188** (≈59% less stress).

## Flags / gaps (called out honestly)
1. **No gridworld / discrete / tabular mode exists** (grepped `grid|discret|value_iter|tabular`).
   The L2 value-iteration example is a **derived teaching gridworld**
   ([`assets/l2_value_iteration.py`](assets/l2_value_iteration.py)) whose reward terms map onto
   the real env (`r_success=+10`, `r_time=−0.01`, a near-wall stress cost scaled by `r_arousal`).
   It is clearly labelled as an abstraction, not something shipped in the repo.
2. Tiers `t5_predators` / `t6_obstacles` are stubs (`available=False`) — not referenced as live.
3. Ear-angle endpoints are literature-anchored, **not data-fit** (the code comments say so).

## Assets in this course (all runnable against the real env)
| file | lesson | output |
|---|---|---|
| [`assets/l1_annotated_frame.py`](assets/l1_annotated_frame.py) | L1 | annotated real-env state frame (S, A, flight zone, hidden arousal) |
| [`assets/l2_value_iteration.py`](assets/l2_value_iteration.py) | L2 | value heatmap + greedy policy over the pasture; hand-checkable value table |
| [`assets/l3_explore_vs_exploit.py`](assets/l3_explore_vs_exploit.py) | L3 | greedy-vs-flank comparison + ε exploration trace |

Run any asset with `python course/assets/<file>.py` (deps: numpy, pillow; matplotlib for L2/L3).
