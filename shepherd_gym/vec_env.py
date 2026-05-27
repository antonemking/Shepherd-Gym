"""Vectorised Shepherd-v0 — B environments stepped in one batched numpy pass.

A faithful batched port of ShepherdEnv (same flight-zone arousal model, same
asymmetric obs/critic layout, same reward). Auto-resets per-env on done. This is
the throughput engine for training; the single-env ShepherdEnv stays the reference
for eval / rendering / baselines. `tests`-style consistency check lives in
scripts (compare_single_vs_vec).

obs[B, obs_dim] is the actor view; info["privileged_state"][B, state_dim] adds the
true latent arousal for the critic; reward[B] is computed on the true latent.
"""
from __future__ import annotations
import numpy as np
from .env import ShepherdConfig


def _u(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-9)


class VecShepherdEnv:
    def __init__(self, cfg: ShepherdConfig | None = None, num_envs: int = 32, seed: int | None = None):
        self.cfg = cfg or ShepherdConfig()
        self.B = num_envs
        self.N = self.cfg.n_sheep
        self.k = self.cfg.k_nearest
        self.pen = np.array(self.cfg.pen_center, dtype=np.float64)
        self.action_dim = 2
        self._rng = np.random.default_rng(seed)
        self.reset()
        self.obs_dim = self._last_obs.shape[1]
        self.state_dim = self._last_state.shape[1]

    # ---------------------------------------------------------------- reset
    def reset(self):
        B, N = self.B, self.N
        self.pos = np.zeros((B, N, 2)); self.vel = np.zeros((B, N, 2))
        self.arousal = np.full((B, N), 0.1); self.penned = np.zeros((B, N), bool)
        self.nn = np.full((B, N), self.cfg.social_dist)
        self.dog_pos = np.zeros((B, 2)); self.dog_vel = np.zeros((B, 2))
        self.t = np.zeros(B, np.int64); self.prev_d = np.zeros(B)
        self.prev_sheep_d = np.zeros((B, N))
        self._init_envs(np.arange(B))
        obs = self._obs(); self._last_obs = obs
        self._last_state = self._privileged(obs)
        return obs

    def _init_envs(self, idx):
        c = self.cfg
        m = len(idx)
        if m == 0:
            return
        pts = self._rng.uniform(2.0, c.world - 2.0, size=(m, self.N, 2))
        for _ in range(25):
            bad = np.linalg.norm(pts - self.pen, axis=2) < c.pen_radius + 3.0
            if not bad.any():
                break
            pts[bad] = self._rng.uniform(2.0, c.world - 2.0, size=(int(bad.sum()), 2))
        self.pos[idx] = pts
        self.vel[idx] = 0.0
        self.arousal[idx] = 0.1
        self.penned[idx] = False
        self.nn[idx] = c.social_dist
        self.dog_pos[idx] = np.array([c.world * 0.5, c.world * 0.4])
        self.dog_vel[idx] = 0.0
        self.t[idx] = 0
        self.prev_d[idx] = self._flock_pen_dist()[idx]
        self.prev_sheep_d[idx] = np.linalg.norm(self.pos[idx] - self.pen, axis=2)

    def privileged_state(self):
        return self._last_state

    # ----------------------------------------------------------------- step
    def step(self, actions):
        c = self.cfg
        a = np.clip(np.asarray(actions, np.float64), -1.0, 1.0) * c.dog_speed
        self.dog_vel = a
        self.dog_pos = np.clip(self.dog_pos + a * c.dt, 0.0, c.world)

        self._step_flock()
        newly = self._update_penning()
        self._update_arousal()

        free = ~self.penned
        cntf = free.sum(1)
        safe = np.maximum(cntf, 1)[:, None]
        centroid = (self.pos * free[..., None]).sum(1) / safe
        nofree = cntf == 0
        centroid[nofree] = self.pen
        d = np.linalg.norm(centroid - self.pen, axis=1)
        progress = self.prev_d - d
        self.prev_d = d
        sd = np.linalg.norm(self.pos - self.pen[None, None, :], axis=2)   # (B,N)
        fetch = (self.prev_sheep_d - sd).sum(1)                            # per-sheep potential
        self.prev_sheep_d = sd
        dev = np.linalg.norm(self.pos - centroid[:, None, :], axis=2)
        spread = (dev * free).sum(1) / np.maximum(cntf, 1)
        mean_ar = (self.arousal * free).sum(1) / np.maximum(cntf, 1)
        mean_ar[nofree] = 0.0

        reward = (c.r_pen * newly + c.r_progress * progress + c.r_fetch * fetch
                  - c.r_spread * spread - c.r_arousal * mean_ar * c.dt - c.r_time)
        term = self.penned.all(1)
        reward = reward + c.r_success * term
        self.t += 1
        trunc = self.t >= c.max_steps
        done = term | trunc

        if done.any():
            self._init_envs(np.where(done)[0])

        obs = self._obs(); self._last_obs = obs
        state = self._privileged(obs); self._last_state = state
        info = {"privileged_state": state, "mean_arousal": mean_ar,
                "term": term, "frac_penned": self.penned.mean(1)}
        return obs, reward.astype(np.float32), done, info

    # --------------------------------------------------------- batched physics
    def _step_flock(self):
        c = self.cfg
        pos, vel = self.pos, self.vel
        N = self.N
        diff = pos[:, :, None, :] - pos[:, None, :, :]            # (B,N,N,2)
        dist = np.linalg.norm(diff, axis=3)
        dist[:, np.arange(N), np.arange(N)] = np.inf
        self.nn = dist.min(axis=2)

        unit = diff / np.maximum(dist, 1e-9)[..., None]
        sep = np.where((dist < c.sep_radius)[..., None], unit, 0.0).sum(2)

        nbr = dist < c.neighbor_radius
        cnt = nbr.sum(2)
        safe = np.maximum(cnt, 1)[..., None]
        ali = (nbr[..., None] * vel[:, None, :, :]).sum(2) / safe
        cen = (nbr[..., None] * pos[:, None, :, :]).sum(2) / safe
        coh = np.where((cnt > 0)[..., None], cen - pos, 0.0)

        to_self = pos - self.dog_pos[:, None, :]
        dd = np.linalg.norm(to_self, axis=2)
        m = (dd < c.flight_zone) & (dd > 1e-6)
        strength = np.where(m, 1.0 - dd / c.flight_zone, 0.0)
        flee = np.where(m[..., None], to_self / np.maximum(dd, 1e-9)[..., None] * strength[..., None], 0.0)

        bnd = np.zeros_like(pos)
        bnd[..., 0] = (pos[..., 0] < 1.5).astype(float) - (pos[..., 0] > c.world - 1.5).astype(float)
        bnd[..., 1] = (pos[..., 1] < 1.5).astype(float) - (pos[..., 1] > c.world - 1.5).astype(float)

        force = (c.w_sep * _u(sep) + c.w_ali * _u(ali) + c.w_coh * _u(coh)
                 + c.w_flee * flee + c.w_bounds * bnd)

        if self.penned.any():
            pen_off = self.pen[None, None, :] - pos
            pen_d = np.linalg.norm(pen_off, axis=2, keepdims=True)
            contain = np.where(pen_d > 0.7, _u(pen_off), 0.0)
            force = np.where(self.penned[..., None], 1.5 * contain, force)

        vel = vel + force * c.dt
        sp = np.linalg.norm(vel, axis=2)
        cap = np.where(self.penned, 0.5, c.sheep_max_speed)
        tf = sp > cap
        vel = np.where(tf[..., None], vel / np.maximum(sp, 1e-9)[..., None] * cap[..., None], vel)
        vel = vel * np.where(self.arousal[..., None] < 0.2, 0.9, 1.0)
        self.vel = vel
        self.pos = np.clip(pos + vel * c.dt, 0.0, c.world)

    def _update_penning(self):
        inside = np.linalg.norm(self.pos - self.pen[None, None, :], axis=2) < self.cfg.pen_radius
        newly = (inside & ~self.penned).sum(1).astype(np.float64)
        self.penned = inside
        return newly

    def _update_arousal(self):
        c = self.cfg
        to_self = self.pos - self.dog_pos[:, None, :]
        dd = np.linalg.norm(to_self, axis=2)
        dir_ds = to_self / np.maximum(dd, 1e-6)[..., None]
        closing = np.maximum(0.0, (dir_ds * self.dog_vel[:, None, :]).sum(2)) / max(c.dog_speed, 1e-6)
        intrusion = np.clip((c.flight_zone - dd) / c.flight_zone, 0.0, 1.0) ** c.intrusion_exp
        pred = (c.w_pred_arousal + c.w_closing * closing) * intrusion
        iso = np.clip((self.nn - c.social_dist) / c.social_dist, 0.0, 1.0)
        drive = pred + c.w_isolation * iso
        self.arousal += (drive * c.arousal_rise - c.arousal_decay) * c.dt
        self.arousal[self.penned] -= 0.6 * c.dt
        np.clip(self.arousal, 0.0, 1.0, out=self.arousal)

    # ------------------------------------------------------------- obs/state
    def _flock_pen_dist(self):
        free = ~self.penned
        cntf = free.sum(1)
        cen = (self.pos * free[..., None]).sum(1) / np.maximum(cntf, 1)[:, None]
        cen[cntf == 0] = self.pen
        return np.linalg.norm(cen - self.pen, axis=1)

    def _obs(self):
        c = self.cfg; W = c.world; B = self.B
        ear = c.ear_neutral_deg + (c.ear_aroused_deg - c.ear_neutral_deg) * self.arousal
        ear = ear + self._rng.normal(0.0, c.ear_noise_deg, size=(B, self.N))
        free = ~self.penned
        cntf = free.sum(1); safe = np.maximum(cntf, 1)
        centroid = (self.pos * free[..., None]).sum(1) / safe[:, None]
        centroid[cntf == 0] = self.pen
        spread = (np.linalg.norm(self.pos - centroid[:, None, :], axis=2) * free).sum(1) / safe
        mean_ear = (ear * free).sum(1) / safe
        mean_ear[cntf == 0] = c.ear_neutral_deg

        rel = self.pos - self.dog_pos[:, None, :]
        reldist = np.linalg.norm(rel, axis=2)
        masked = np.where(free, reldist, np.inf)
        order = np.argsort(masked, axis=1)[:, : self.k]
        self._near_idx = order
        bidx = np.arange(B)[:, None]
        valid = np.take_along_axis(free, order, axis=1)
        self._near_valid = valid
        sel_rel = rel[bidx, order] * valid[..., None]
        sel_vel = self.vel[bidx, order] * valid[..., None]
        sel_ear = np.where(valid, ear[bidx, order], 0.0)

        parts = [self.dog_pos / W, self.dog_vel / c.dog_speed, (centroid - self.dog_pos) / W,
                 (spread / W)[:, None], self.penned.mean(1)[:, None], (mean_ear / c.ear_norm)[:, None],
                 (self.pen[None, :] - self.dog_pos) / W, (self.pen[None, :] - centroid) / W]
        for j in range(self.k):
            parts += [sel_rel[:, j] / W, sel_vel[:, j] / c.sheep_max_speed, (sel_ear[:, j] / c.ear_norm)[:, None]]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _privileged(self, obs):
        free = ~self.penned
        cntf = free.sum(1)
        mean_ar = (self.arousal * free).sum(1) / np.maximum(cntf, 1)
        mean_ar[cntf == 0] = 0.0
        maxa = self.arousal.max(1)
        bidx = np.arange(self.B)[:, None]
        near_ar = np.where(self._near_valid, self.arousal[bidx, self._near_idx], 0.0)
        return np.concatenate([obs, mean_ar[:, None], maxa[:, None], near_ar], axis=1).astype(np.float32)
