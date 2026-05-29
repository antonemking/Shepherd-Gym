"""The competition difficulty ladder.

Each tier is a `ShepherdConfig` variant (only difficulty knobs overridden) plus a
held-out seed block and the composite score you must beat to unlock the NEXT tier.
Higher tiers tighten the flight zone / sheep speed / flee weight so *welfare* gets
harder to preserve — not just penning — keeping the speed-vs-stress thesis load-bearing.

Tiers marked `available=False` need env mechanics that don't exist yet (predators,
obstacles — roadmap P3); the CLI refuses to score them rather than pretending.

Seed blocks are disjoint per tier (`seed0 .. seed0+n_seeds-1`). `ShepherdEnv.reset(seed)`
reseeds deterministically, so a deterministic policy gets a reproducible tier score.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .env import ShepherdConfig

BASE = ShepherdConfig()  # 8 sheep · world 20 · flight_zone 7 — the RESULTS.md config


@dataclass(frozen=True)
class Tier:
    key: str
    name: str
    cfg: ShepherdConfig
    seed0: int            # start of this tier's held-out seed block
    n_seeds: int          # episodes scored
    unlock_score: float   # composite (0–100) you must beat to unlock the NEXT tier
    available: bool = True  # False => needs new env mechanics (stubbed, not scorable)
    note: str = ""


LADDER = [
    Tier("t0_pasture", "Pasture (tutorial)",
         replace(BASE, n_sheep=6, world=18.0, flight_zone=8.0, sheep_max_speed=1.6),
         seed0=100_000, n_seeds=30, unlock_score=45.0,
         note="Small, calm flock with a roomy flight zone — learn the controls."),
    Tier("t1_paddock", "Paddock (baseline)",
         BASE,
         seed0=200_000, n_seeds=50, unlock_score=55.0,
         note="The canonical config the baselines and RESULTS.md are measured on."),
    Tier("t2_range", "Open Range",
         replace(BASE, n_sheep=14, world=28.0, max_steps=800),
         seed0=300_000, n_seeds=50, unlock_score=55.0,
         note="Bigger field, bigger flock — gathering distance grows."),
    Tier("t3_skittish", "Skittish Flock",
         replace(BASE, n_sheep=12, flight_zone=9.5, sheep_max_speed=2.3,
                 w_flee=3.2, arousal_rise=2.0),
         seed0=400_000, n_seeds=50, unlock_score=50.0,
         note="Jumpy sheep: wide flight zone, fast flee, fast-rising arousal."),
    Tier("t4_big_muster", "Big Muster",
         replace(BASE, n_sheep=24, world=34.0, max_steps=1000, k_nearest=7),
         seed0=500_000, n_seeds=50, unlock_score=50.0,
         note="Two dozen sheep on a large field — the endurance tier."),
    # ---- LOCKED: require NEW env mechanics that do not exist yet ----
    Tier("t5_predators", "Wolves (LOCKED)", BASE,
         seed0=600_000, n_seeds=50, unlock_score=100.0, available=False,
         note="Needs predator entities + screening reward in env.py (roadmap P3)."),
    Tier("t6_obstacles", "Obstacles (LOCKED)", BASE,
         seed0=700_000, n_seeds=50, unlock_score=100.0, available=False,
         note="Needs static obstacles + collision in env.py. Stub only."),
]

LADDER_BY_KEY = {t.key: t for t in LADDER}
AVAILABLE = [t for t in LADDER if t.available]


def get_tier(key: str) -> Tier:
    if key not in LADDER_BY_KEY:
        raise KeyError(f"unknown tier '{key}'. Known: {', '.join(LADDER_BY_KEY)}")
    return LADDER_BY_KEY[key]


def next_tier(key: str) -> Tier | None:
    """The tier immediately after `key` in the ladder, or None if it's the last."""
    for i, t in enumerate(LADDER):
        if t.key == key:
            return LADDER[i + 1] if i + 1 < len(LADDER) else None
    raise KeyError(f"unknown tier '{key}'")
