"""Content-grade renderer for Shepherd-v0.

Built for the SamSeesSheep visual grammar: ghost trails, swarm-of-attempts
overlays, and side-by-side model comparison, exported as gif / animated webp
(the LinkedIn loop format) / mp4. Pure PIL + opencv — no torch, no display.

Sheep are tinted by arousal (calm woolly white -> stressed red), so the welfare
signal is legible at a glance, the same way the ear-angle readout is.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2  # for mp4
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

GRASS = (60, 110, 60)
PEN_FILL = (150, 120, 80)
PEN_RING = (90, 70, 45)
DOG = (30, 30, 50)
CALM = (236, 236, 228)
STRESS = (224, 70, 52)
PENNED_RING = (120, 200, 130)


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Renderer:
    def __init__(self, world: float, size: int = 540, margin: int = 20, trail_len: int = 28):
        self.world = world
        self.size = size
        self.margin = margin
        self.scale = (size - 2 * margin) / world
        self.trail_len = trail_len

    def _px(self, p):
        return (self.margin + p[0] * self.scale, self.size - (self.margin + p[1] * self.scale))

    def _blank(self):
        img = Image.new("RGB", (self.size, self.size), GRASS)
        return img

    def _draw_pen(self, d, pen, pen_r):
        c = self._px(pen)
        r = pen_r * self.scale
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=PEN_FILL, outline=PEN_RING, width=4)

    def _dot(self, d, p, r, fill, outline=None, w=1):
        c = self._px(p)
        box = [c[0] - r, c[1] - r, c[0] + r, c[1] + r]
        d.ellipse(box, fill=fill, outline=outline, width=w)

    def frame(self, state, dog_trail=None, sheep_trail=None, label=None):
        img = self._blank()
        d = ImageDraw.Draw(img, "RGBA")
        self._draw_pen(d, state["pen"], state["pen_r"])

        # ghost trails (fading)
        if sheep_trail:
            for age, snap in enumerate(sheep_trail):
                a = int(40 * (age + 1) / len(sheep_trail))
                for p in snap:
                    self._dot(d, p, 2.0, (255, 255, 255, a))
        if dog_trail and len(dog_trail) > 1:
            n = len(dog_trail)
            for i in range(1, n):
                a = int(200 * i / n)
                p0, p1 = self._px(dog_trail[i - 1]), self._px(dog_trail[i])
                d.line([p0, p1], fill=(255, 220, 120, a), width=3)

        # sheep, tinted by arousal
        for i, p in enumerate(state["sheep"]):
            col = _lerp(CALM, STRESS, float(state["arousal"][i]))
            if state["penned"][i]:
                self._dot(d, p, 5.0, col, outline=PENNED_RING, w=2)
            else:
                self._dot(d, p, 5.0, col)

        # dog
        self._dot(d, state["dog"], 7.0, DOG, outline=(255, 255, 255), w=2)

        if label:
            d.rectangle([0, 0, self.size, 26], fill=(0, 0, 0, 120))
            d.text((8, 6), label, fill=(255, 255, 255))
        return img


def record_episode(env, policy, renderer: Renderer | None = None, seed=0, max_steps=None, label=None):
    """Roll out one episode; return (frames, info). Frames carry ghost trails."""
    obs, _ = env.reset(seed=seed)
    r = renderer or Renderer(env.cfg.world)
    dog_trail = deque(maxlen=r.trail_len)
    sheep_hist = deque(maxlen=8)
    frames = []
    ret, steps, arousal_sum, peak_penned = 0.0, 0, 0.0, 0
    T = max_steps or env.cfg.max_steps
    for _ in range(T):
        st = env.render_state()
        dog_trail.append(st["dog"].copy())
        sheep_hist.append(st["sheep"].copy())
        frames.append(r.frame(st, dog_trail=list(dog_trail), sheep_trail=list(sheep_hist), label=label))
        a = policy(env)
        obs, rew, term, trunc, info = env.step(a)
        ret += rew
        steps += 1
        arousal_sum += info["mean_arousal"]
        peak_penned = max(peak_penned, info["penned"])
        if term or trunc:
            break
    info_out = {
        "return": ret,
        "steps": steps,
        "success": bool(env.penned.all()),
        "penned": int(env.penned.sum()),
        "n_sheep": env.cfg.n_sheep,
        "mean_arousal": arousal_sum / max(steps, 1),
    }
    return frames, info_out


def save_gif(frames, path, fps=20):
    if not frames:
        return
    dur = int(1000 / fps)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0, disposal=2)


def save_webp(frames, path, fps=20):
    if not frames:
        return
    dur = int(1000 / fps)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0, format="WEBP")


def save_mp4(frames, path, fps=20):
    if not frames or not _HAS_CV2:
        return False
    w, h = frames[0].size
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR))
    vw.release()
    return True


def swarm_overlay(env_factory, policy_factory, n_runs=40, size=540, seed0=0):
    """The 'swarm of attempts' shot: every run's dog path ghosted on one frame,
    plus where each run left its flock. Instantly shows policy consistency."""
    env0 = env_factory()
    r = Renderer(env0.cfg.world, size=size)
    img = r._blank()
    d = ImageDraw.Draw(img, "RGBA")
    r._draw_pen(d, env0.pen, env0.cfg.pen_radius)
    successes = 0
    for k in range(n_runs):
        env = env_factory()
        pol = policy_factory()
        env.reset(seed=seed0 + k)
        trail = [env.dog_pos.copy()]
        for _ in range(env.cfg.max_steps):
            _, _, term, trunc, _ = env.step(pol(env))
            trail.append(env.dog_pos.copy())
            if term or trunc:
                break
        success = bool(env.penned.all())
        successes += success
        col = (120, 230, 140, 70) if success else (235, 120, 110, 55)
        pts = [r._px(p) for p in trail]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
        for p in env.sheep_pos:
            r._dot(d, p, 2.0, (255, 255, 255, 90))
    d.rectangle([0, 0, size, 26], fill=(0, 0, 0, 130))
    d.text((8, 6), f"{n_runs} runs — {successes} penned (green) / {n_runs - successes} failed (red)", fill=(255, 255, 255))
    return img, successes


def compare_panel(named_frames: dict, fps=20):
    """Tile several policies' episodes side by side on a shared timeline —
    the v0.2-vs-v0.4 overlay, but for herding policies."""
    labels = list(named_frames.keys())
    seqs = [named_frames[k] for k in labels]
    n = max(len(s) for s in seqs)
    out = []
    for i in range(n):
        tiles = []
        for k, s in zip(labels, seqs):
            f = s[min(i, len(s) - 1)].copy()
            tiles.append(f)
        w = sum(t.width for t in tiles)
        h = max(t.height for t in tiles)
        canvas = Image.new("RGB", (w, h), (20, 20, 20))
        x = 0
        for t in tiles:
            canvas.paste(t, (x, 0))
            x += t.width
        out.append(canvas)
    return out
