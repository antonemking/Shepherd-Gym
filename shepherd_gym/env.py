"""Shepherd-v0 — a low-stress herding RL environment.

The agent is a dog. It herds a flock of boid-sheep into a fold. Each sheep
carries an **arousal** signal in [0,1] that rises under dog pressure, crowding,
and high-speed flight and decays when calm — a deliberate echo of the ear-angle
stress signal measured in the SamSeesSheep welfare work. The reward trades off
penning the flock *quickly* against keeping the flock *calm*, so a trained policy
can be studied along a speed-vs-stress frontier.

Pure-numpy, Gymnasium-compatible API (no hard gymnasium dependency, so it runs on
numpy alone). Designed to vectorize cleanly for PufferLib-style training later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ShepherdConfig:
    n_sheep: int = 8
    world: float = 20.0            # square field side (metres-ish)
    pen_center: tuple = (16.0, 16.0)
    pen_radius: float = 3.0
    dt: float = 0.1
    max_steps: int = 600
    k_nearest: int = 5             # sheep exposed individually in the observation

    dog_speed: float = 2.4
    sheep_max_speed: float = 1.9
    flee_radius: float = 5.0       # dog presence radius
    sep_radius: float = 1.6
    neighbor_radius: float = 4.0

    # boid weights
    w_sep: float = 1.8
    w_ali: float = 0.5
    w_coh: float = 0.6
    w_flee: float = 2.6
    w_bounds: float = 2.0

    # arousal / stress dynamics
    arousal_gain: float = 1.4      # how fast pressure raises arousal
    arousal_decay: float = 0.5     # how fast it relaxes when calm
    speed_arousal: float = 0.25    # own flight speed adds arousal

    # reward weights
    r_pen: float = 1.0             # per sheep newly entering the fold
    r_progress: float = 0.6        # flock moving toward the fold
    r_spread: float = 0.02         # keep the flock together
    r_arousal: float = 0.5         # WELFARE: penalise mean flock stress (per step)
    r_time: float = 0.01           # gentle urgency
    r_success: float = 10.0        # all penned

    seed: int | None = None


class ShepherdEnv:
    metadata = {"name": "Shepherd-v0"}

    def __init__(self, cfg: ShepherdConfig | None = None):
        self.cfg = cfg or ShepherdConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        c = self.cfg
        self.pen = np.array(c.pen_center, dtype=np.float64)
        # action: dog desired velocity (vx, vy) in [-1, 1]
        self.action_dim = 2
        self.obs_dim = self._obs().shape[0] if False else self._compute_obs_dim()
        self.t = 0

    # ---- spaces (lightweight; real gymnasium spaces added in the wrapper) ----
    def _compute_obs_dim(self) -> int:
        # dog(4) + centroid_rel(2) + spread(1) + mean_arousal(1) + frac_penned(1)
        # + pen_rel_dog(2) + pen_rel_centroid(2) + k*(rel_pos2+rel_vel2+arousal1)
        return 4 + 2 + 1 + 1 + 1 + 2 + 2 + self.cfg.k_nearest * 5

    # ---------------------------------------------------------------- reset
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c = self.cfg
        # scatter sheep away from the fold; dog starts near field centre
        pts = []
        while len(pts) < c.n_sheep:
            p = self._rng.uniform(2.0, c.world - 2.0, size=2)
            if np.linalg.norm(p - self.pen) > c.pen_radius + 3.0:
                pts.append(p)
        self.sheep_pos = np.array(pts, dtype=np.float64)
        self.sheep_vel = np.zeros((c.n_sheep, 2))
        self.arousal = np.full(c.n_sheep, 0.1)
        self.penned = np.zeros(c.n_sheep, dtype=bool)
        self.dog_pos = np.array([c.world * 0.5, c.world * 0.4])
        self.dog_vel = np.zeros(2)
        self.t = 0
        self._prev_flock_pen_dist = self._flock_pen_dist()
        return self._obs(), {}

    # ----------------------------------------------------------------- step
    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        self.dog_vel = a * c.dog_speed
        self.dog_pos = np.clip(self.dog_pos + self.dog_vel * c.dt, 0.0, c.world)

        self._step_flock()
        newly = self._update_penning()
        self._update_arousal()

        # ---- reward ----
        d = self._flock_pen_dist()
        progress = self._prev_flock_pen_dist - d
        self._prev_flock_pen_dist = d
        spread = self._flock_spread()
        mean_ar = float(self.arousal[~self.penned].mean()) if (~self.penned).any() else 0.0

        reward = (
            c.r_pen * newly
            + c.r_progress * progress
            - c.r_spread * spread
            - c.r_arousal * mean_ar * c.dt
            - c.r_time
        )
        terminated = bool(self.penned.all())
        if terminated:
            reward += c.r_success
        self.t += 1
        truncated = self.t >= c.max_steps

        info = {
            "penned": int(self.penned.sum()),
            "frac_penned": float(self.penned.mean()),
            "mean_arousal": mean_ar,
            "flock_pen_dist": d,
        }
        return self._obs(), float(reward), terminated, truncated, info

    # --------------------------------------------------------- flock physics
    def _step_flock(self):
        c = self.cfg
        pos, vel = self.sheep_pos, self.sheep_vel
        n = c.n_sheep
        diff = pos[:, None, :] - pos[None, :, :]      # (n,n,2) i - j
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)

        # separation: push from neighbours within sep_radius
        sep_mask = dist < c.sep_radius
        with np.errstate(invalid="ignore"):
            unit = diff / dist[:, :, None]
        sep = np.nansum(np.where(sep_mask[:, :, None], unit, 0.0), axis=1)

        # alignment + cohesion within neighbor_radius
        nbr = dist < c.neighbor_radius
        cnt = nbr.sum(axis=1)[:, None]
        safe = np.maximum(cnt, 1)
        ali = (nbr[:, :, None] * vel[None, :, :]).sum(axis=1) / safe
        centroid = (nbr[:, :, None] * pos[None, :, :]).sum(axis=1) / safe
        coh = np.where(cnt > 0, centroid - pos, 0.0)

        # flee from the dog
        to_self = pos - self.dog_pos[None, :]
        dd = np.linalg.norm(to_self, axis=1)
        flee = np.zeros_like(pos)
        m = (dd < c.flee_radius) & (dd > 1e-6)
        strength = np.where(m, 1.0 - dd / c.flee_radius, 0.0)
        flee[m] = (to_self[m] / dd[m, None]) * strength[m, None]

        # soft bounds
        bnd = np.zeros_like(pos)
        bnd[:, 0] += (pos[:, 0] < 1.5).astype(float) - (pos[:, 0] > c.world - 1.5).astype(float)
        bnd[:, 1] += (pos[:, 1] < 1.5).astype(float) - (pos[:, 1] > c.world - 1.5).astype(float)

        force = (
            c.w_sep * _norm_rows(sep)
            + c.w_ali * _norm_rows(ali)
            + c.w_coh * _norm_rows(coh)
            + c.w_flee * flee
            + c.w_bounds * bnd
        )
        # penned sheep ignore flock/flee and just settle near the fold centre,
        # so the fold actually *holds* them while the dog fetches the rest.
        if self.penned.any():
            pen_off = self.pen[None, :] - pos
            pen_d = np.linalg.norm(pen_off, axis=1, keepdims=True)
            contain = np.where(pen_d > 0.7, _norm_rows(pen_off), 0.0)
            force[self.penned] = 1.5 * contain[self.penned]

        vel = vel + force * c.dt
        sp = np.linalg.norm(vel, axis=1)
        cap = np.where(self.penned, 0.5, c.sheep_max_speed)
        too_fast = sp > cap
        vel[too_fast] = vel[too_fast] / sp[too_fast, None] * cap[too_fast, None]
        # grazing drag when calm
        vel *= np.where(self.arousal[:, None] < 0.2, 0.9, 1.0)
        self.sheep_vel = vel
        self.sheep_pos = np.clip(pos + vel * c.dt, 0.0, c.world)

    def _update_penning(self) -> int:
        inside = np.linalg.norm(self.sheep_pos - self.pen, axis=1) < self.cfg.pen_radius
        newly = int((inside & ~self.penned).sum())
        self.penned = inside
        return newly

    def _update_arousal(self):
        c = self.cfg
        dd = np.linalg.norm(self.sheep_pos - self.dog_pos[None, :], axis=1)
        dog_press = np.clip(1.0 - dd / c.flee_radius, 0.0, 1.0)
        dog_press *= 0.5 + 0.5 * (np.linalg.norm(self.dog_vel) / max(c.dog_speed, 1e-6))
        own_speed = np.linalg.norm(self.sheep_vel, axis=1) / max(c.sheep_max_speed, 1e-6)
        rise = (dog_press + c.speed_arousal * own_speed) * c.arousal_gain
        self.arousal += (rise - c.arousal_decay) * c.dt
        # penned sheep calm faster (they're safe in the fold)
        self.arousal[self.penned] -= 0.6 * c.dt
        np.clip(self.arousal, 0.0, 1.0, out=self.arousal)

    # ------------------------------------------------------------- helpers
    def _flock_centroid(self):
        free = self.sheep_pos[~self.penned]
        return free.mean(axis=0) if len(free) else self.pen.copy()

    def _flock_pen_dist(self):
        return float(np.linalg.norm(self._flock_centroid() - self.pen))

    def _flock_spread(self):
        free = self.sheep_pos[~self.penned]
        if len(free) < 2:
            return 0.0
        return float(np.linalg.norm(free - free.mean(axis=0), axis=1).mean())

    def _obs(self):
        c = self.cfg
        W = c.world
        centroid = self._flock_centroid()
        parts = [
            self.dog_pos / W, self.dog_vel / c.dog_speed,
            (centroid - self.dog_pos) / W,
            np.array([self._flock_spread() / W]),
            np.array([float(self.arousal[~self.penned].mean()) if (~self.penned).any() else 0.0]),
            np.array([self.penned.mean()]),
            (self.pen - self.dog_pos) / W,
            (self.pen - centroid) / W,
        ]
        # k nearest free sheep to the dog
        free_idx = np.where(~self.penned)[0]
        if len(free_idx) == 0:
            free_idx = np.arange(c.n_sheep)
        rel = self.sheep_pos[free_idx] - self.dog_pos[None, :]
        order = np.argsort(np.linalg.norm(rel, axis=1))[: c.k_nearest]
        for j in range(c.k_nearest):
            if j < len(order):
                i = free_idx[order[j]]
                parts += [(self.sheep_pos[i] - self.dog_pos) / W,
                          self.sheep_vel[i] / c.sheep_max_speed,
                          np.array([self.arousal[i]])]
            else:
                parts += [np.zeros(2), np.zeros(2), np.zeros(1)]
        return np.concatenate(parts).astype(np.float32)

    # snapshot for the renderer
    def render_state(self) -> dict:
        return {
            "dog": self.dog_pos.copy(),
            "sheep": self.sheep_pos.copy(),
            "arousal": self.arousal.copy(),
            "penned": self.penned.copy(),
            "pen": self.pen.copy(),
            "pen_r": self.cfg.pen_radius,
            "world": self.cfg.world,
            "t": self.t,
        }


def _norm_rows(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, n, out=np.zeros_like(a), where=n > 1e-9)
