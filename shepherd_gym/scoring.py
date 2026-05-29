"""The composite competition score — one number per tier, in [0, 100].

Per episode (T = tier max_steps):
    P_i = penned fraction at episode end                  (penning)
    W_i = 1 - clip(mean_arousal_i, 0, 1)                  (welfare; gentler = higher)
    S_i = (1 - steps_i / T) if the whole flock was penned else 0   (speed, success-gated)

Tier score (mean over the tier's seed block):
    SCORE = 100 * (w_pen * mean(P) + w_welf * mean(W) + w_speed * mean(S))

Why speed is gated on full success: it makes stalling-to-dodge-stress a losing strategy.
A policy that times out to keep arousal low gets S=0 and a mediocre P, so it can't beat a
real herder. Never approaching the flock gives high W but P≈0 (caps ~35). Scattering the
flock spikes arousal so W drops. The gentle-but-effective policy wins — exactly the
speed-vs-stress thesis in results/RESULTS.md.

Deterministic: no global RNG is touched here, so a deterministic policy gets the same score
every time on a tier (each episode is `env.reset(seed=tier.seed0 + k)`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .benchmark import rollout_episode
from .env import ShepherdEnv
from .tiers import Tier


@dataclass(frozen=True)
class ScoreWeights:
    """Single source of truth for the composite. Pinned by tests/test_scoring.py."""
    pen: float = 0.50
    welf: float = 0.35
    speed: float = 0.15


WEIGHTS = ScoreWeights()


def composite(P: float, W: float, S: float, w: ScoreWeights = WEIGHTS) -> float:
    return 100.0 * (w.pen * P + w.welf * W + w.speed * S)


def score_episodes(eps: list[dict], max_steps: int, w: ScoreWeights = WEIGHTS) -> dict:
    """Aggregate per-episode rollouts (from `rollout_episode`) into a tier result."""
    n = len(eps)
    if n == 0:
        raise ValueError("no episodes to score")
    P = float(np.mean([e["penned"] for e in eps]))
    W = float(np.mean([1.0 - min(max(e["arousal"], 0.0), 1.0) for e in eps]))
    S = float(np.mean([(1.0 - e["steps"] / max_steps) if e["won"] else 0.0 for e in eps]))
    return {
        "score": composite(P, W, S, w),
        "P": P, "W": W, "S": S,
        "success": float(np.mean([e["won"] for e in eps])),
        "arousal": float(np.mean([e["arousal"] for e in eps])),  # raw, for context
        "n": n,
    }


def score_tier(policy, tier: Tier, w: ScoreWeights = WEIGHTS, n_seeds: int | None = None) -> dict:
    """Score `policy` (a callable env->action) on `tier`. Returns the score breakdown.

    `n_seeds` overrides the tier's block length (used by fast tests); the seed block
    still starts at `tier.seed0`.
    """
    if not tier.available:
        raise ValueError(
            f"tier '{tier.key}' is locked: {tier.note} "
            "It needs env mechanics that don't exist yet — not scorable."
        )
    n = n_seeds if n_seeds is not None else tier.n_seeds
    env = ShepherdEnv(tier.cfg)
    eps = [rollout_episode(env, policy, tier.seed0 + k) for k in range(n)]
    out = score_episodes(eps, tier.cfg.max_steps, w)
    out["tier"] = tier.key
    return out
