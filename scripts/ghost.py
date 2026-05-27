"""Checkpoint-ghosting footage — the 'watch it learn' shot.

Loads policy checkpoints saved across training and, on a SHARED seed, renders:
  * out/ghost_compare.mp4/.gif — early→late policies side by side on the same scenario
  * out/ghost_paths.png        — every checkpoint's dog-path overlaid on one frame,
                                  coloured early(red)→late(green)

    .venv/bin/python scripts/ghost.py [seed]
"""
import sys, re, pathlib
from collections import deque
import numpy as np
import torch
from PIL import ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from train import Actor                                  # reuse the trained net def
from shepherd_gym.env import ShepherdEnv, ShepherdConfig
from shepherd_gym import render as R

OUT = ROOT / "out"
CKPT_BASE = OUT / "checkpoints"


def _mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


@torch.no_grad()
def rollout(actor, seed, label, renderer):
    cfg = ShepherdConfig(); env = ShepherdEnv(cfg)
    obs, _ = env.reset(seed=seed)
    trail, hist, frames, traj = deque(maxlen=renderer.trail_len), deque(maxlen=8), [], []
    for _ in range(cfg.max_steps):
        st = env.render_state(); trail.append(st["dog"].copy()); hist.append(st["sheep"].copy())
        traj.append(st["dog"].copy())
        frames.append(renderer.frame(st, dog_trail=list(trail), sheep_trail=list(hist), label=label))
        mean = actor.net(torch.tensor(env._obs(), dtype=torch.float32)).numpy()
        obs, r, term, trunc, info = env.step(mean)
        if term or trunc:
            break
    return frames, traj, info


def main(seed=7, tag="ra0.5"):
    ckpt = CKPT_BASE / tag
    cks = sorted(ckpt.glob("actor_*.pt"), key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))
    if not cks:
        print("No checkpoints in", ckpt, "— run scripts/train.py first."); return
    # up to 4 checkpoints spread across training
    pick = cks if len(cks) <= 4 else [cks[i] for i in np.linspace(0, len(cks) - 1, 4).astype(int)]
    steps = [int(re.search(r"(\d+)", p.name).group(1)) for p in pick]
    print("Ghosting checkpoints at steps:", steps)

    probe = ShepherdEnv(ShepherdConfig())
    rnd = R.Renderer(probe.cfg.world)
    named, trajs = {}, []
    for p, s in zip(pick, steps):
        actor = Actor(probe.obs_dim, probe.action_dim); actor.load_state_dict(torch.load(p)); actor.eval()
        frames, traj, info = rollout(actor, seed, f"step {s//1000}k", rnd)
        named[f"step {s//1000}k"] = frames; trajs.append((s, traj))
        print(f"  step {s:>7}: penned {info['penned']}/{probe.cfg.n_sheep}, arousal {info['mean_arousal']:.3f}")

    panel = R.compare_panel(named)
    R.save_mp4(panel, str(OUT / f"ghost_compare_{tag}.mp4")); R.save_gif(panel, str(OUT / f"ghost_compare_{tag}.gif"))

    # path overlay
    img = rnd._blank(); d = ImageDraw.Draw(img, "RGBA")
    rnd._draw_pen(d, probe.pen, probe.cfg.pen_radius)
    n = len(trajs)
    for i, (s, traj) in enumerate(trajs):
        t = i / max(n - 1, 1)
        col = _mix((224, 70, 52), (120, 230, 140), t)        # red(early) → green(late)
        pts = [rnd._px(p) for p in traj]
        if len(pts) > 1:
            d.line(pts, fill=col + (230,), width=3)
        d.rectangle([10, 10 + i * 22, 30, 26 + i * 22], fill=col + (255,))
        d.text((36, 12 + i * 22), f"step {s//1000}k", fill=(20, 20, 20))
    img.save(str(OUT / f"ghost_paths_{tag}.png"))
    print(f"Wrote ghost_compare_{tag}.mp4/.gif and ghost_paths_{tag}.png to {OUT}/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="ra0.5")
    a = ap.parse_args()
    main(a.seed, a.tag)
