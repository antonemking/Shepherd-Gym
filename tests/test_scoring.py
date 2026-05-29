"""Lock the composite score: the exact formula, the anti-gaming property, determinism."""
import numpy as np
import pytest

from shepherd_gym.baselines import FlankShepherd
from shepherd_gym.scoring import WEIGHTS, composite, score_episodes, score_tier
from shepherd_gym.tiers import get_tier


def test_weights_sum_to_one():
    assert WEIGHTS.pen + WEIGHTS.welf + WEIGHTS.speed == pytest.approx(1.0)


def test_composite_formula_pinned():
    # P=1, W=1, S=1 -> perfect 100
    assert composite(1.0, 1.0, 1.0) == pytest.approx(100.0)
    # known mix
    assert composite(0.5, 0.5, 0.0) == pytest.approx(100 * (0.5 * 0.5 + 0.35 * 0.5))


def test_score_episodes_known_values():
    # one full-pen fast episode, one failure
    eps = [
        {"won": True, "steps": 100, "arousal": 0.2, "penned": 1.0},
        {"won": False, "steps": 600, "arousal": 0.8, "penned": 0.5},
    ]
    T = 600
    P = (1.0 + 0.5) / 2
    W = ((1 - 0.2) + (1 - 0.8)) / 2
    S = ((1 - 100 / T) + 0.0) / 2  # second is failure -> 0 speed credit
    res = score_episodes(eps, T)
    assert res["P"] == pytest.approx(P)
    assert res["W"] == pytest.approx(W)
    assert res["S"] == pytest.approx(S)
    assert res["score"] == pytest.approx(composite(P, W, S))


def test_speed_credit_gated_on_success():
    # identical arousal/penned, but a win (slow) must still beat a near-timeout loss
    won = score_episodes([{"won": True, "steps": 590, "arousal": 0.3, "penned": 1.0}], 600)
    lost = score_episodes([{"won": False, "steps": 590, "arousal": 0.3, "penned": 1.0}], 600)
    assert won["S"] > 0.0
    assert lost["S"] == 0.0
    assert won["score"] > lost["score"]


def test_stall_policy_loses_to_flank():
    """A policy that sits still to keep arousal low must score below the flank expert —
    the core anti-gaming guarantee."""
    tier = get_tier("t1_paddock")
    stall = lambda env: np.zeros(2)
    n = 8  # small for speed
    stall_res = score_tier(stall, tier, n_seeds=n)
    flank_res = score_tier(FlankShepherd(), tier, n_seeds=n)
    assert flank_res["score"] > stall_res["score"]


def test_score_tier_deterministic():
    tier = get_tier("t1_paddock")
    a = score_tier(FlankShepherd(), tier, n_seeds=6)
    b = score_tier(FlankShepherd(), tier, n_seeds=6)
    assert a == b


def test_locked_tier_refuses():
    with pytest.raises(ValueError):
        score_tier(FlankShepherd(), get_tier("t5_predators"), n_seeds=2)
