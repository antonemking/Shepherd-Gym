"""Shepherd-v0 — a low-stress herding RL environment.

The agent is a dog. It herds a flock of boid-sheep into a fold. Each sheep carries
a latent **arousal** ∈ [0,1] — acute handling stress. Its dynamics are grounded in
the livestock flight-zone / flight-initiation-distance literature (Grandin's
handling work; sheep flight-zone studies) and in sheep gregariousness (separation
from flockmates is a primary acute stressor): arousal rises sharply once the dog
enters the flight zone, more so when the dog is *closing fast*, and when a sheep is
*isolated* from its flockmates; it recovers slowly when pressure lifts.

ASYMMETRIC information (actor-critic):
  * The POLICY only observes what is realistically sensable — positions, velocities,
    flock geometry, and a *noisy ear-angle* observable rendered from arousal (the
    sim analogue of what the SamSeesSheep CV pipeline estimates from video).
  * The CRITIC gets a PRIVILEGED state that also includes the true latent arousal
    (free in sim; used only at training time to reduce value-estimate variance).
  * The REWARD is computed on the true latent arousal — the actual welfare cost.
This mirrors real deployment: at run time you act on noisy CV estimates; the true
stress is only knowable in simulation.

Pure-numpy, Gymnasium-compatible API (no hard gymnasium dependency). Vectorizes
cleanly for PufferLib-style training later.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    sep_radius: float = 1.6
    neighbor_radius: float = 4.0

    # boid weights
    w_sep: float = 1.8
    w_ali: float = 0.5
    w_coh: float = 0.6
    w_flee: float = 2.6
    w_bounds: float = 2.0

    # ---- flight-zone arousal model (grounded in flight-zone / FID / handling lit) ----
    flight_zone: float = 7.0       # FID radius (m): inside this the dog is a threat (Grandin)
    intrusion_exp: float = 1.8     # arousal steepens nonlinearly as the dog gets closer
    w_pred_arousal: float = 1.3    # base weight of in-flight-zone predator pressure
    w_closing: float = 1.0         # extra arousal when the dog is APPROACHING (closing speed)
    w_isolation: float = 0.9       # arousal from separation from flockmates (gregariousness)
    social_dist: float = 3.0       # comfortable nearest-neighbour distance (m)
    arousal_rise: float = 1.6      # onset gain — fast
    arousal_decay: float = 0.22    # recovery — slow (asymmetric: stress settles slower than it spikes)

    # ---- ear-angle observable (mirrors the SamSeesSheep CV output) ----
    # Anchored to SamSeesSheep's SPFES thresholds + v0.4 benchmark. Endpoints are NOT yet
    # data-fit — that needs the validated pipeline on curated motionless clips (the available
    # reviewed labels are too few/uncontrolled; see scripts/calibrate.py, docs/data-roadmap.md).
    ear_neutral_deg: float = 25.0  # calm: ears up/forward (within the NEUTRAL→UP band, UP>30°)
    ear_aroused_deg: float = -10.0 # aroused: ears back/down (SamSeesSheep clinical DOWN threshold)
    ear_noise_deg: float = 4.0     # measurement σ — SamSeesSheep v0.4 held-out benchmark (docs/v0.4-benchmark.md)
    ear_norm: float = 35.0         # normaliser for the observation

    # ---- reward weights (computed on TRUE latent arousal) ----
    r_pen: float = 1.0
    r_progress: float = 0.0        # centroid progress (off — superseded by per-sheep fetch)
    r_fetch: float = 0.15          # per-sheep potential: reward EVERY sheep's progress to the fold
    r_spread: float = 0.02
    r_arousal: float = 0.5         # WELFARE: penalise mean flock arousal (per step → integrated dose)
    r_time: float = 0.01
    r_success: float = 10.0

    seed: int | None = None


class ShepherdEnv:
    metadata = {"name": "Shepherd-v0"}

    def __init__(self, cfg: ShepherdConfig | None = None):
        self.cfg = cfg or ShepherdConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.pen = np.array(self.cfg.pen_center, dtype=np.float64)
        self.action_dim = 2
        self._near_idx = np.arange(min(self.cfg.k_nearest, self.cfg.n_sheep))
        o, _ = self.reset(seed=self.cfg.seed)
        self.obs_dim = o.shape[0]                       # actor observation
        self.state_dim = self.privileged_state().shape[0]  # critic (privileged) state

    # ---------------------------------------------------------------- reset
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c = self.cfg
        pts = []
        while len(pts) < c.n_sheep:
            p = self._rng.uniform(2.0, c.world - 2.0, size=2)
            if np.linalg.norm(p - self.pen) > c.pen_radius + 3.0:
                pts.append(p)
        self.sheep_pos = np.array(pts, dtype=np.float64)
        self.sheep_vel = np.zeros((c.n_sheep, 2))
        self.arousal = np.full(c.n_sheep, 0.1)
        self.penned = np.zeros(c.n_sheep, dtype=bool)
        self._nn = np.full(c.n_sheep, c.social_dist)
        self.dog_pos = np.array([c.world * 0.5, c.world * 0.4])
        self.dog_vel = np.zeros(2)
        self.t = 0
        self._prev_flock_pen_dist = self._flock_pen_dist()
        self._prev_sheep_dist = self._sheep_pen_dist()
        obs = self._obs()
        self._last_obs = obs
        return obs, {}

    # ----------------------------------------------------------------- step
    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        self.dog_vel = a * c.dog_speed
        self.dog_pos = np.clip(self.dog_pos + self.dog_vel * c.dt, 0.0, c.world)

        self._step_flock()
        newly = self._update_penning()
        self._update_arousal()

        d = self._flock_pen_dist()
        progress = self._prev_flock_pen_dist - d
        self._prev_flock_pen_dist = d
        sd = self._sheep_pen_dist()
        fetch = float((self._prev_sheep_dist - sd).sum())   # per-sheep potential shaping
        self._prev_sheep_dist = sd
        spread = self._flock_spread()
        mean_ar = float(self.arousal[~self.penned].mean()) if (~self.penned).any() else 0.0

        reward = (
            c.r_pen * newly
            + c.r_progress * progress
            + c.r_fetch * fetch
            - c.r_spread * spread
            - c.r_arousal * mean_ar * c.dt      # welfare cost, on the TRUE latent
            - c.r_time
        )
        terminated = bool(self.penned.all())
        if terminated:
            reward += c.r_success
        self.t += 1
        truncated = self.t >= c.max_steps

        obs = self._obs()
        self._last_obs = obs
        info = {
            "penned": int(self.penned.sum()),
            "frac_penned": float(self.penned.mean()),
            "mean_arousal": mean_ar,                 # TRUE latent welfare cost
            "peak_arousal": float(self.arousal.max()),
            "flock_pen_dist": d,
            "privileged_state": self.privileged_state(),
        }
        return obs, float(reward), terminated, truncated, info

    # --------------------------------------------------------- flock physics
    def _step_flock(self):
        c = self.cfg
        pos, vel = self.sheep_pos, self.sheep_vel
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        self._nn = dist.min(axis=1) if c.n_sheep > 1 else np.array([c.social_dist])

        sep_mask = dist < c.sep_radius
        with np.errstate(invalid="ignore"):
            unit = diff / dist[:, :, None]
        sep = np.nansum(np.where(sep_mask[:, :, None], unit, 0.0), axis=1)

        nbr = dist < c.neighbor_radius
        cnt = nbr.sum(axis=1)[:, None]
        safe = np.maximum(cnt, 1)
        ali = (nbr[:, :, None] * vel[None, :, :]).sum(axis=1) / safe
        centroid = (nbr[:, :, None] * pos[None, :, :]).sum(axis=1) / safe
        coh = np.where(cnt > 0, centroid - pos, 0.0)

        to_self = pos - self.dog_pos[None, :]
        dd = np.linalg.norm(to_self, axis=1)
        flee = np.zeros_like(pos)
        m = (dd < c.flight_zone) & (dd > 1e-6)
        strength = np.where(m, 1.0 - dd / c.flight_zone, 0.0)
        flee[m] = (to_self[m] / dd[m, None]) * strength[m, None]

        bnd = np.zeros_like(pos)
        bnd[:, 0] += (pos[:, 0] < 1.5).astype(float) - (pos[:, 0] > c.world - 1.5).astype(float)
        bnd[:, 1] += (pos[:, 1] < 1.5).astype(float) - (pos[:, 1] > c.world - 1.5).astype(float)

        force = (
            c.w_sep * _norm_rows(sep) + c.w_ali * _norm_rows(ali)
            + c.w_coh * _norm_rows(coh) + c.w_flee * flee + c.w_bounds * bnd
        )
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
        vel *= np.where(self.arousal[:, None] < 0.2, 0.9, 1.0)
        self.sheep_vel = vel
        self.sheep_pos = np.clip(pos + vel * c.dt, 0.0, c.world)

    def _update_penning(self) -> int:
        inside = np.linalg.norm(self.sheep_pos - self.pen, axis=1) < self.cfg.pen_radius
        newly = int((inside & ~self.penned).sum())
        self.penned = inside
        return newly

    def _update_arousal(self):
        """Flight-zone stress model. Pressure rises inside the flight zone (FID),
        scaled by how fast the dog is closing in, plus an isolation term for sheep
        separated from flockmates; recovery is slow (asymmetric)."""
        c = self.cfg
        to_self = self.sheep_pos - self.dog_pos[None, :]
        dd = np.linalg.norm(to_self, axis=1)
        dir_ds = to_self / np.maximum(dd[:, None], 1e-6)              # unit dog→sheep
        closing = np.maximum(0.0, dir_ds @ self.dog_vel) / max(c.dog_speed, 1e-6)  # dog approaching?
        intrusion = np.clip((c.flight_zone - dd) / c.flight_zone, 0.0, 1.0) ** c.intrusion_exp
        pred_pressure = (c.w_pred_arousal + c.w_closing * closing) * intrusion
        isolation = np.clip((self._nn - c.social_dist) / c.social_dist, 0.0, 1.0)
        drive = pred_pressure + c.w_isolation * isolation
        self.arousal += (drive * c.arousal_rise - c.arousal_decay) * c.dt
        self.arousal[self.penned] -= 0.6 * c.dt   # safe in the fold → calm faster
        np.clip(self.arousal, 0.0, 1.0, out=self.arousal)

    # ------------------------------------------------------------- helpers
    def _flock_centroid(self):
        free = self.sheep_pos[~self.penned]
        return free.mean(axis=0) if len(free) else self.pen.copy()

    def _flock_pen_dist(self):
        return float(np.linalg.norm(self._flock_centroid() - self.pen))

    def _sheep_pen_dist(self):
        return np.linalg.norm(self.sheep_pos - self.pen, axis=1)

    def _flock_spread(self):
        free = self.sheep_pos[~self.penned]
        if len(free) < 2:
            return 0.0
        return float(np.linalg.norm(free - free.mean(axis=0), axis=1).mean())

    def _ear_angles(self):
        """Per-sheep NOISY ear angle (deg) — the observable the CV would estimate.
        Monotonic in arousal (calm → neutral/forward, aroused → flat/back) + meas. noise."""
        c = self.cfg
        base = c.ear_neutral_deg + (c.ear_aroused_deg - c.ear_neutral_deg) * self.arousal
        return base + self._rng.normal(0.0, c.ear_noise_deg, size=self.cfg.n_sheep)

    def _obs(self):
        """ACTOR observation — only realistically sensable quantities (no true arousal)."""
        c = self.cfg
        W = c.world
        centroid = self._flock_centroid()
        ear = self._ear_angles()
        free = ~self.penned
        mean_ear = float(ear[free].mean()) if free.any() else c.ear_neutral_deg
        parts = [
            self.dog_pos / W, self.dog_vel / c.dog_speed,
            (centroid - self.dog_pos) / W,
            np.array([self._flock_spread() / W]),
            np.array([self.penned.mean()]),
            np.array([mean_ear / c.ear_norm]),               # noisy aggregate ear angle (CV-style)
            (self.pen - self.dog_pos) / W,
            (self.pen - centroid) / W,
        ]
        free_idx = np.where(free)[0]
        if len(free_idx) == 0:
            free_idx = np.arange(c.n_sheep)
        rel = self.sheep_pos[free_idx] - self.dog_pos[None, :]
        order = np.argsort(np.linalg.norm(rel, axis=1))[: c.k_nearest]
        self._near_idx = free_idx[order]                     # cache for the privileged state
        for j in range(c.k_nearest):
            if j < len(order):
                i = free_idx[order[j]]
                parts += [(self.sheep_pos[i] - self.dog_pos) / W,
                          self.sheep_vel[i] / c.sheep_max_speed,
                          np.array([ear[i] / c.ear_norm])]   # noisy per-sheep ear angle
            else:
                parts += [np.zeros(2), np.zeros(2), np.zeros(1)]
        return np.concatenate(parts).astype(np.float32)

    def privileged_state(self):
        """CRITIC state — the actor obs PLUS the TRUE latent arousal (sim-only)."""
        obs = self._last_obs if hasattr(self, "_last_obs") else self._obs()
        free = ~self.penned
        mean_ar = float(self.arousal[free].mean()) if free.any() else 0.0
        near = np.zeros(self.cfg.k_nearest, dtype=np.float32)
        for j, i in enumerate(self._near_idx[: self.cfg.k_nearest]):
            near[j] = self.arousal[i]
        return np.concatenate([obs, np.array([mean_ar, float(self.arousal.max())], dtype=np.float32), near]).astype(np.float32)

    def render_state(self) -> dict:
        return {
            "dog": self.dog_pos.copy(), "sheep": self.sheep_pos.copy(),
            "arousal": self.arousal.copy(), "penned": self.penned.copy(),
            "pen": self.pen.copy(), "pen_r": self.cfg.pen_radius,
            "world": self.cfg.world, "t": self.t,
        }


def _norm_rows(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, n, out=np.zeros_like(a), where=n > 1e-9)
