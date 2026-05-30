"""L1 asset — an ANNOTATED state frame of the real Shepherd-v0 env.

Renders one frame from the actual environment (via shepherd_gym.render) and
overlays the MDP labels a learner needs to *see* the (S, A, R) on the field:
the dog, its action arrow, the flight-zone circle around the dog, the fold,
and each sheep tinted by its TRUE latent arousal (the welfare signal the policy
never directly observes).

Everything drawn here is read from the live env — positions, the flight_zone
radius, pen geometry, and per-sheep arousal — nothing is faked.

Run:
    python course/assets/l1_annotated_frame.py
Output:
    course/assets/out/l1_annotated_frame.png

Deps: numpy, pillow (base shepherd-gym deps).
"""
from __future__ import annotations
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from shepherd_gym import ShepherdEnv, ShepherdConfig
from shepherd_gym.baselines import FlankShepherd
from shepherd_gym.render import Renderer, GRASS

OUT = pathlib.Path(__file__).resolve().parent / "out"

# colours that match the repo's render grammar
INK = (245, 245, 240)
ARROW = (255, 210, 90)
ZONE = (255, 120, 90)


def annotate(seed: int = 7, warmup: int = 40, scale: int = 2):
    """Roll the expert a few steps so the field looks like a real episode mid-muster,
    then render + annotate the resulting state."""
    cfg = ShepherdConfig(seed=seed)
    env = ShepherdEnv(cfg)
    env.reset(seed=seed)
    pol = FlankShepherd()
    action = np.zeros(2)
    for _ in range(warmup):
        action = pol(env)
        env.step(action)

    st = env.render_state()
    r = Renderer(cfg.world, size=540 * scale, margin=20 * scale)
    img = r.frame(st)  # repo renderer: sheep tinted calm->stress, dog, fold
    img = img.convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")

    def px(p):
        return r._px(p)

    # --- flight zone (the FID radius the sheep flee inside) ---
    dc = px(st["dog"])
    fz = cfg.flight_zone * r.scale
    d.ellipse([dc[0] - fz, dc[1] - fz, dc[0] + fz, dc[1] + fz],
              outline=ZONE + (220,), width=3 * scale)
    d.text((dc[0] + fz * 0.30, dc[1] - fz - 16 * scale),
           f"flight zone  r={cfg.flight_zone:g} m  (sheep flee inside)", fill=ZONE)

    # --- action arrow A = a * dog_speed (the only thing the agent controls) ---
    tip = st["dog"] + action * cfg.dog_speed * 1.4
    tp = px(tip)
    d.line([dc, tp], fill=ARROW + (255,), width=5 * scale)
    d.ellipse([tp[0] - 5 * scale, tp[1] - 5 * scale, tp[0] + 5 * scale, tp[1] + 5 * scale],
              fill=ARROW + (255,))
    d.text((tp[0] + 6 * scale, tp[1]),
           f"action a=({action[0]:+.2f},{action[1]:+.2f}) -> dog_vel", fill=ARROW)

    # --- label the dog, the fold, and the most-aroused sheep (the welfare cost) ---
    d.text((dc[0] + 10 * scale, dc[1] + 8 * scale), "DOG = agent", fill=INK)
    pc = px(st["pen"])
    d.text((pc[0] - 14 * scale, pc[1] - 6 * scale), "FOLD", fill=(20, 20, 20, 255))

    hot = int(np.argmax(st["arousal"]))
    hp = px(st["sheep"][hot])
    d.text((hp[0] + 7 * scale, hp[1] - 7 * scale),
           f"arousal={st['arousal'][hot]:.2f} (TRUE latent, hidden from policy)", fill=ZONE)

    # --- header: the MDP at a glance, all from the live env ---
    mean_ar = float(st["arousal"][~st["penned"]].mean()) if (~st["penned"]).any() else 0.0
    d.rectangle([0, 0, img.width, 30 * scale], fill=(0, 0, 0, 150))
    d.text((8 * scale, 8 * scale),
           f"Shepherd-v0  |  S: obs={env.obs_dim}-d (no true stress)  "
           f"A: 2-d in [-1,1]^2  |  penned {st['penned'].sum()}/{cfg.n_sheep}  "
           f"mean arousal {mean_ar:.2f}  t={st['t']}",
           fill=INK)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "l1_annotated_frame.png"
    img.save(path)
    print(f"wrote {path}  ({img.width}x{img.height})")
    print(f"  obs_dim={env.obs_dim} state_dim(privileged)={env.state_dim} "
          f"action_dim={env.action_dim}")
    return path


if __name__ == "__main__":
    annotate()
