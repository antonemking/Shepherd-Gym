"""Lock the submission adapter: the three load paths, and that scripted entries are
torch-free."""
import sys

import numpy as np
import pytest

from shepherd_gym.adapter import Submission, load_policy
from shepherd_gym.env import ShepherdEnv
from shepherd_gym.tiers import get_tier

TIER = get_tier("t1_paddock")


def _probe():
    return ShepherdEnv(TIER.cfg)


def test_registry_load_is_torch_free():
    sys.modules.pop("torch", None)
    pol = load_policy(Submission(kind="registry", ref="flank"), _probe())
    env = _probe()
    env.reset(seed=0)
    a = np.asarray(pol(env), dtype=float)
    assert a.shape == (2,)
    assert "torch" not in sys.modules


def test_unknown_registry_raises():
    with pytest.raises(KeyError):
        load_policy(Submission(kind="registry", ref="nope"), _probe())


# a module-level callable entry, referenced by dotted path
def my_policy(env):
    return np.zeros(2)


def test_callable_load():
    pol = load_policy(Submission(kind="callable", ref=f"{__name__}:my_policy"), _probe())
    env = _probe()
    env.reset(seed=0)
    assert np.asarray(pol(env)).shape == (2,)


def test_callable_class_is_instantiated():
    pol = load_policy(
        Submission(kind="callable", ref="shepherd_gym.baselines:FlankShepherd"), _probe())
    env = _probe()
    env.reset(seed=0)
    assert np.asarray(pol(env)).shape == (2,)


def test_bad_callable_ref_raises():
    with pytest.raises(ValueError):
        load_policy(Submission(kind="callable", ref="no_colon_here"), _probe())


def test_checkpoint_obs_dim_guard():
    torch = pytest.importorskip("torch")
    import os
    import tempfile

    scripts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from train import Actor

    # a checkpoint with the WRONG obs_dim must be rejected with a clear error
    bad = Actor(obs_dim=_probe().obs_dim + 5, act_dim=2)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(bad.state_dict(), f.name)
        path = f.name
    with pytest.raises(ValueError, match="obs_dim"):
        load_policy(Submission(kind="checkpoint", ref=path), _probe())

    # a correctly-sized checkpoint loads and produces a 2-vector
    good = Actor(obs_dim=_probe().obs_dim, act_dim=2)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(good.state_dict(), f.name)
        gpath = f.name
    pol = load_policy(Submission(kind="checkpoint", ref=gpath), _probe())
    env = _probe()
    env.reset(seed=0)
    assert np.asarray(pol(env)).shape == (2,)
