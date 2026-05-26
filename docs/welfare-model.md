# The welfare model & the asymmetric setup

How shepherd-gym connects to SamSeesSheep, and exactly how the agent "knows" stress
and is rewarded for reducing it. Written to be the methods backbone of the write-up.

## Constructs (be precise)

- We model **acute handling arousal/fear**, the affective axis that **ear posture
  indexes** (Reefmann et al. 2009; Boissy et al. 2011) — *not* nociceptive pain. The
  Sheep Pain Facial Expression Scale (McLennan & Mahmoud 2019) is a related but
  distinct *pain* construct; we deliberately don't claim it here.
- Per sheep, `arousal ∈ [0,1]` is a **latent** state. SamSeesSheep *estimates* this
  latent from video (ear angle); shepherd-gym *optimises against* it.

## Latent arousal dynamics — flight-zone grounded

Each step, per sheep (see `_update_arousal` in `env.py`):

```
intrusion   = clip((flight_zone − dist_to_dog) / flight_zone, 0, 1) ** intrusion_exp
closing     = max(0, dog_velocity · unit(dog→sheep)) / dog_speed     # is the dog approaching?
pred        = (w_pred + w_closing · closing) · intrusion
isolation   = clip((nearest_flockmate_dist − social_dist) / social_dist, 0, 1)
arousal += (arousal_rise · (pred + w_isolation · isolation) − arousal_decay) · dt
```

Each term is grounded in a documented handling-stress mechanism:

| term | behaviour | grounding |
|---|---|---|
| `flight_zone = 7 m`, threshold + nonlinear `intrusion` | calm outside the flight zone; stress rises steeply as the handler penetrates it | livestock **flight zone / flight-initiation-distance** (Grandin handling work; FID studies) |
| `closing` (approach speed) | a *fast, direct* approach is far more arousing than a slow/tangential one | handling guidance: move slowly, approach at an angle |
| `isolation` from flockmates | a sheep split from the flock is acutely stressed | sheep **gregariousness** — separation is a primary acute stressor |
| asymmetric `rise=1.6 ≫ decay=0.22` | stress spikes fast, settles slowly | stress onset vs recovery asymmetry |

**Honesty bar:** these parameters are *informed by* the flight-zone literature, **not
yet fit to data**. The next rigor step is calibration against real ear-angle time
series from SamSeesSheep clips (paired dog-position / flock-speed / spacing ↔ measured
ear angle). Until then, results are a *methods demonstration*, not a claim about real
Katahdins. (The *locomotion* model — flee + flocking — has empirical roots in Strömbom
et al. 2014, who fit shepherding heuristics to GPS tracks of a real dog and flock.)

## Asymmetric actor-critic — how the policy "knows", and why

This mirrors real deployment, where you only ever have the CV's noisy estimate:

- **Actor observation (38-d, realistically sensable):** dog pose, flock centroid/
  spread/penned-fraction, fold-relative vectors, and a **noisy ear-angle observable**
  per nearby sheep + an aggregate — `ear = lerp(neutral 25°, aroused −8°, arousal) +
  N(0, σ=4°)`. The σ matches SamSeesSheep's v0.4 held-out jitter. The policy must
  **infer** stress from posture + behaviour, exactly as your pipeline does — it never
  sees the true latent.
- **Critic (privileged) state (45-d):** the actor obs **plus the true latent arousal**
  (mean, peak, and per-near-sheep). Free in simulation; used only at training time to
  cut value-estimate variance. Exposed via `info["privileged_state"]` / `privileged_state()`.
- **Reward** is computed on the **true latent** arousal: `− r_arousal · mean_arousal · dt`,
  i.e. the **integrated stress dose** over the episode, traded against penning speed
  (`r_arousal` is the speed-vs-stress dial for the Pareto study).

Why this is the right shape: at *train* time the critic exploits ground truth it can
never have at *deploy* time; the actor is forced to operate on the same noisy ear-angle
signal SamSeesSheep produces. So a policy trained here is, in principle, deployable on
real CV output — and the sim↔real gap is exactly the arousal-model calibration above.

## Open calibration work (the path to a real claim)

1. Extract from handling video (via SamSeesSheep): time series of dog/handler position
   relative to flock, flock speed, inter-sheep spacing, and ear-angle-derived arousal.
2. Fit `flight_zone`, `intrusion_exp`, `w_closing`, `w_isolation`, rise/decay so
   simulated arousal trajectories match measured ear-angle responses under matched stimuli.
3. Re-run the benchmark; report the trained policy's stress reduction vs heuristics
   *in calibrated units*.

## References (verify against your own refs before publishing)

- McLennan & Mahmoud 2019 — Sheep Pain Facial Expression Scale (pain construct).
- Reefmann, Wechsler & Gygax 2009 — ear postures as indicators of emotional valence/arousal in sheep.
- Boissy et al. 2011 — ear postures and emotional states.
- Strömbom et al. 2014, *J. R. Soc. Interface* — shepherding heuristics fit to GPS data (locomotion model).
- Grandin — livestock flight zone / point of balance / low-stress handling.
