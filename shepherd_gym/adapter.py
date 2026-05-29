"""Normalize any submission into the one policy contract: `policy(env) -> action ∈ [-1,1]²`.

Three accepted submission kinds let scripted and learned entries compete on one board:
  * "registry"   — a name in shepherd_gym.baselines.REGISTRY ("random"/"greedy"/"flank").
  * "callable"   — a dotted path "module:attr"; if it's a class it's instantiated.
  * "checkpoint" — a PyTorch actor state_dict (.pt) as emitted by scripts/train.py.

torch is imported LAZILY, only in the checkpoint branch — entering with a pure callable
needs nothing beyond the base deps. A checkpoint is tier-specific: its input width must
match the tier's obs_dim (k_nearest / n_sheep change it), so we guard and fail clearly.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass
class Submission:
    kind: str          # "registry" | "callable" | "checkpoint"
    ref: str           # registry name, "module:attr", or path/to/actor.pt
    author: str = ""
    title: str = ""


def load_policy(sub: Submission, probe_env):
    """Return a callable `policy(env) -> action`. `probe_env` is a fresh env built from
    the target tier's cfg (only the checkpoint branch needs it, for the obs_dim guard)."""
    if sub.kind == "registry":
        from .baselines import REGISTRY
        if sub.ref not in REGISTRY:
            raise KeyError(f"unknown baseline '{sub.ref}'. Known: {', '.join(REGISTRY)}")
        return REGISTRY[sub.ref]
    if sub.kind == "callable":
        if ":" not in sub.ref:
            raise ValueError(f"callable ref must be 'module:attr', got '{sub.ref}'")
        mod_name, attr = sub.ref.split(":", 1)
        obj = getattr(importlib.import_module(mod_name), attr)
        return obj() if isinstance(obj, type) else obj
    if sub.kind == "checkpoint":
        return _load_checkpoint_policy(sub.ref, probe_env)
    raise ValueError(f"unknown submission kind '{sub.kind}'")


def _load_checkpoint_policy(path: str, probe_env):
    import os
    import sys

    import torch                                # LAZY — only needed for learned entries

    # reuse the exact Actor net definition from scripts/train.py (same trick as pareto.py)
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from train import Actor

    # weights_only=True refuses to unpickle arbitrary objects — a submitted .pt is
    # untrusted input, and torch<2.6 still defaults this to False. We only need tensors.
    sd = torch.load(path, map_location="cpu", weights_only=True)
    w0 = sd["net.0.weight"]
    if w0.shape[1] != probe_env.obs_dim:
        raise ValueError(
            f"checkpoint obs_dim {w0.shape[1]} != tier obs_dim {probe_env.obs_dim}. "
            "This checkpoint was trained for a different tier — learned entries are "
            "submitted per tier (scripted entries are tier-agnostic)."
        )
    net = Actor(probe_env.obs_dim, probe_env.action_dim)
    net.load_state_dict(sd)
    net.eval()

    def policy(env):
        with torch.no_grad():
            mean = net.net(torch.tensor(env._obs(), dtype=torch.float32))
        return mean.numpy()

    return policy
