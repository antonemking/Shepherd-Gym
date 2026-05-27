"""Write-up footage: flank (expert) vs BC warm-start vs the gentle RL policy,
side by side on a shared seed. Sheep are tinted by true arousal, so the welfare
difference is visible — the gentle policy's flock stays calmer (less red).

    .venv/bin/python scripts/showcase.py [--seed 7] [--tag ra0.25_bc]
"""
import sys, json, pathlib, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from shepherd_gym.env import ShepherdEnv, ShepherdConfig
from shepherd_gym.baselines import FlankShepherd
from shepherd_gym import render as R
from pareto import actor_pol
OUT = ROOT / "out"


def best_step(tag):
    hist = json.load(open(OUT / f"train_history_{tag}.json"))
    return sorted(hist, key=lambda e: (-e["success"], e["arousal"]))[0]["step"]


def main(seed=7, tag="ra0.25_bc"):
    cfg = ShepherdConfig(); rnd = R.Renderer(cfg.world)
    ppo_path = OUT / "checkpoints" / tag / f"actor_{best_step(tag)}.pt"
    policies = [("flank", "flank (expert)", FlankShepherd()),
                ("bc", "BC warm-start", actor_pol(OUT / "bc_actor.pt")),
                ("ppo", "PPO gentle (r=0.25)", actor_pol(ppo_path))]
    named = {}
    for key, label, pol in policies:
        env = ShepherdEnv(cfg)
        frames, info = R.record_episode(env, pol, renderer=rnd, seed=seed, label=label)
        named[label] = frames
        R.save_gif(frames, str(OUT / f"showcase_{key}.gif"))
        print(f"  {label:22} penned {info['penned']}/{info['n_sheep']}  arousal {info['mean_arousal']:.3f}")
    panel = R.compare_panel(named)
    R.save_mp4(panel, str(OUT / "showcase_compare.mp4"))
    R.save_gif(panel, str(OUT / "showcase_compare.gif"))
    print(f"wrote showcase_compare.gif/.mp4 + per-policy gifs to {OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="ra0.25_bc")
    a = ap.parse_args()
    main(a.seed, a.tag)
