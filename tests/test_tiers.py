"""Lock the tier ladder: configs build, seed blocks are disjoint, obs_dim is stable."""
from shepherd_gym.env import ShepherdEnv
from shepherd_gym.tiers import AVAILABLE, LADDER, get_tier, next_tier


def test_available_tiers_build_envs():
    for t in AVAILABLE:
        env = ShepherdEnv(t.cfg)
        assert env.obs_dim > 0 and env.state_dim > 0


def test_locked_tiers_marked():
    locked = [t for t in LADDER if not t.available]
    assert {t.key for t in locked} == {"t5_predators", "t6_obstacles"}
    for t in locked:
        assert t.note  # must explain why it's locked


def test_seed_blocks_disjoint():
    blocks = [range(t.seed0, t.seed0 + t.n_seeds) for t in LADDER]
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            assert set(blocks[i]).isdisjoint(blocks[j])


def test_thresholds_in_range():
    for t in LADDER:
        assert 0.0 <= t.unlock_score <= 100.0
        assert t.n_seeds > 0


def test_obs_dim_per_tier_stable():
    # building the same tier twice gives the same obs width (catches accidental k drift)
    for t in AVAILABLE:
        assert ShepherdEnv(t.cfg).obs_dim == ShepherdEnv(t.cfg).obs_dim


def test_next_tier_links():
    assert next_tier(LADDER[0].key).key == LADDER[1].key
    assert next_tier(LADDER[-1].key) is None


def test_get_tier_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        get_tier("nope")
