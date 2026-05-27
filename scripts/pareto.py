"""Speed-vs-stress frontier — the thesis figure.

Reads the fine-tune sweep histories (train_history_ra*_bc.json), plots pen-success
vs mean flock arousal across the welfare weight r_arousal, and overlays the flank
heuristic and the BC warm-start as reference points (all under the same eval
protocol). Top-left = the goal: high success at low stress.

    .venv/bin/python scripts/pareto.py
"""
import sys, re, json, pathlib
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from shepherd_gym.env import ShepherdEnv, ShepherdConfig
from shepherd_gym.baselines import FlankShepherd
from train import Actor
OUT = ROOT / "out"


def eval_policy(pol, episodes=24, seed0=9000):
    cfg = ShepherdConfig(); succ, ar, pen = 0, [], []
    for k in range(episodes):
        env = ShepherdEnv(cfg); env.reset(seed=seed0 + k)
        a_sum, t = 0.0, 0
        for _ in range(cfg.max_steps):
            _, _, term, trunc, info = env.step(pol(env)); a_sum += info["mean_arousal"]; t += 1
            if term or trunc:
                break
        succ += bool(env.penned.all()); ar.append(a_sum / max(t, 1)); pen.append(float(env.penned.mean()))
    return succ / episodes, float(np.mean(ar)), float(np.mean(pen))


def actor_pol(path):
    probe = ShepherdEnv(ShepherdConfig())
    net = Actor(probe.obs_dim, probe.action_dim); net.load_state_dict(torch.load(path)); net.eval()
    def f(env):
        with torch.no_grad():
            return net.net(torch.tensor(env._obs(), dtype=torch.float32)).numpy()
    return f


def best_ckpt(tag, n_top=3, episodes=30):
    """Fine-tuning is unstable (peaks then drifts), so report the BEST checkpoint —
    the policy you'd actually deploy — re-evaluated on more seeds for robustness."""
    hist = json.load(open(OUT / f"train_history_{tag}.json"))
    top = sorted(hist, key=lambda e: (-e["success"], e["arousal"]))[:n_top]
    best = None
    for e in top:
        p = OUT / "checkpoints" / tag / f"actor_{e['step']}.pt"
        if not p.exists():
            continue
        s, a, _ = eval_policy(actor_pol(p), episodes=episodes)
        if best is None or (s, -a) > (best[0], -best[1]):
            best = (s, a, e["step"])
    return best


def main():
    pts = []   # (label, r_arousal, success, arousal)
    print("Re-evaluating best checkpoints per run (30 seeds)...")
    for h in sorted(OUT.glob("train_history_ra*_bc.json")):
        w = float(re.search(r"ra([\d.]+)_bc", h.name).group(1))
        b = best_ckpt(f"ra{w:g}_bc")
        if b:
            pts.append((f"PPO r={w:g}", w, b[0], b[1]))
            print(f"  r={w:g}: best ckpt @{b[2]} → success={b[0]*100:.0f}% arousal={b[1]:.3f}")
    pts.sort(key=lambda x: x[1])

    refs = []
    print("Evaluating references (flank, BC) under the eval protocol...")
    s, a, _ = eval_policy(FlankShepherd()); refs.append(("flank (expert)", s, a, "#5A462D"))
    if (OUT / "bc_actor.pt").exists():
        s, a, _ = eval_policy(actor_pol(OUT / "bc_actor.pt")); refs.append(("BC warm-start", s, a, "#6E5A8C"))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if pts:
        xs = [p[3] for p in pts]; ys = [p[2] for p in pts]
        ax.plot(xs, ys, "-o", color="#3C6E3C", lw=2, ms=9, label="PPO (BC warm-start), swept r_arousal", zorder=3)
        for lab, w, sc, ar in pts:
            ax.annotate(f"r={w:g}", (ar, sc), textcoords="offset points", xytext=(8, 6), fontsize=9, color="#1E1E32")
    for lab, sc, ar, col in refs:
        ax.scatter([ar], [sc], color=col, s=120, marker="D", zorder=4, label=lab)
    ax.set_xlabel("mean flock arousal  (welfare cost → worse)")
    ax.set_ylabel("pen success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Speed-vs-stress frontier — can RL herd as well, but gentler?")
    ax.annotate("← gentler · just as effective ↑", (0.02, 0.95), xycoords="axes fraction",
                fontsize=10, color="#3C6E3C")
    ax.grid(alpha=0.25); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "pareto.png", dpi=130)
    print("\nFrontier points:")
    for lab, w, sc, ar in pts:
        print(f"  {lab:12} success={sc*100:5.1f}%  arousal={ar:.3f}")
    for lab, sc, ar, _ in refs:
        print(f"  {lab:14} success={sc*100:5.1f}%  arousal={ar:.3f}")
    print(f"\nwrote {OUT}/pareto.png")


if __name__ == "__main__":
    main()
