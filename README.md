# shepherd-gym 🐕🐑

**A low-stress herding RL environment.** A dog (the agent) musters a flock of
boid-sheep into a fold. Each sheep carries an **arousal** signal in `[0,1]` that
rises under dog pressure, crowding, and high-speed flight — a deliberate echo of
the **ear-angle stress signal** measured in [SamSeesSheep](../../lorewood-advisors/sheep-seg).
The reward trades penning the flock *fast* against keeping it *calm*, so a trained
policy can be studied along a **speed-vs-stress frontier**.

> Sibling experiment to the SamSeesSheep welfare-CV series: *from measuring sheep
> stress in real video → simulating it → optimising a herding policy against it.*

Pure-numpy core, Gymnasium-compatible, PufferLib-ready. The renderer is built for
the same comparison-visual grammar as the SamSeesSheep loops (ghost trails,
model-vs-model overlays, swarm-of-attempts, reward/stress curves).

## Run it (no training stack needed)

Only needs `numpy`, `pillow`, `opencv-python-headless` (already in your env):

```bash
python scripts/demo.py 40        # benchmark scripted baselines + emit visuals
```

Outputs to `out/`:
- `benchmark.md` — metrics table (success%, steps, penned/N, return, **arousal/welfare**)
- `<policy>.gif` / `.mp4` — single episodes with ghost trails, sheep tinted by stress
- `compare_random_greedy_flank.mp4` — side-by-side on a shared seed (the v0.2-vs-v0.4 grammar)
- `swarm_<policy>.png` — 40 runs ghosted on one frame (green=penned, red=failed)

## The environment — `Shepherd-v0`

- **Agent:** the dog. **Action:** continuous desired velocity `(vx, vy) ∈ [-1,1]²`.
- **Flock:** Reynolds boids (separation/alignment/cohesion) + flee-from-dog + soft bounds.
- **Arousal:** per-sheep stress; ↑ with dog proximity × dog speed and own flight speed,
  ↓ when calm; penned sheep calm fastest. Rendered as sheep colour (white→red).
- **Observation:** dog state, flock centroid/spread/mean-arousal/penned-fraction,
  fold-relative vectors, and the K nearest free sheep (rel pos/vel + arousal).
- **Reward:** `+penned`, `+progress-to-fold`, `−spread`, **`−mean_arousal`** (welfare),
  `−time`, big bonus on full pen. Weights live in `ShepherdConfig` — turn `r_arousal`
  up/down to sweep the gentle-vs-fast trade-off.

## Roadmap

- **P1 (done):** env + scripted baselines (random / greedy / Strömbom-style flank) +
  content renderer + benchmark harness. *Runs on numpy alone.*
- **P2:** `uv sync --extra train`; PPO (PufferLib, or CleanRL single-file fallback);
  training-curve + **checkpoint-ghosting** visuals (step-1k vs 10k vs 100k racing the
  same seed); trained agent added to the benchmark.
- **P3:** predators (wolves) — the dog must herd *and* screen them off; and the
  **speed-vs-stress Pareto study** (sweep `r_arousal`).
- **P4:** small-LLM-as-shepherd — can a tiny LM act as the policy from text-framed obs?
- **P5 (stretch):** calibrate the flock model against real YOLO-pose trajectories from
  SamSeesSheep barn video (behavioural digital twin); optional 3D render.

## Layout

```
shepherd_gym/
  env.py         # Shepherd-v0 (numpy, gym-compatible)
  baselines.py   # random / greedy / flank scripted policies
  render.py      # ghost trails, swarm overlay, compare panel; gif/webp/mp4
  benchmark.py   # eval over held-out seeds -> table + artifacts
scripts/demo.py  # one-command benchmark + visuals
```
