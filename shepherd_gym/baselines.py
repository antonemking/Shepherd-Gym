"""Scripted policies for benchmarking and for 'before training' footage.

Each policy is a callable: policy(env) -> action in [-1, 1]^2 (the dog's desired
velocity). These are the bars a trained agent has to beat.
"""
from __future__ import annotations
import numpy as np


def _toward(frm, to):
    d = to - frm
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else np.zeros(2)


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env):
        return self.rng.uniform(-1.0, 1.0, size=2)


class GreedyDriver:
    """Naive: always sit behind the flock centroid and shove it at the fold.
    Works on a tight flock, scatters a loose one — the classic novice mistake."""
    name = "greedy"
    offset = 3.0

    def __call__(self, env):
        free = env.sheep_pos[~env.penned]
        if len(free) == 0:
            return np.zeros(2)
        c = free.mean(axis=0)
        behind = _toward(env.pen, c)               # pen -> flock direction
        drive_point = c + behind * self.offset
        return _toward(env.dog_pos, drive_point)


class FlankShepherd:
    """Strömbom-style heuristic: COLLECT stragglers when the flock is loose,
    otherwise DRIVE the whole flock toward the fold from directly behind."""
    name = "flank"
    drive_offset = 3.0
    collect_offset = 2.0

    def __call__(self, env):
        free_idx = np.where(~env.penned)[0]
        if len(free_idx) == 0:
            return np.zeros(2)
        free = env.sheep_pos[free_idx]
        c = free.mean(axis=0)
        dists = np.linalg.norm(free - c, axis=1)
        # cohesion radius grows with flock size (Strömbom ~ r0 * n^(2/3))
        cohere_r = 1.0 * max(len(free), 1) ** 0.5
        furthest = dists.max()
        if furthest > cohere_r:
            # COLLECT: get behind the worst straggler relative to the centroid
            s = free[int(np.argmax(dists))]
            point = s + _toward(c, s) * self.collect_offset
        else:
            # DRIVE: get behind the centroid on the line away from the fold
            point = c + _toward(env.pen, c) * self.drive_offset
        return _toward(env.dog_pos, point)


REGISTRY = {p.name: p for p in [RandomPolicy(), GreedyDriver(), FlankShepherd()]}
