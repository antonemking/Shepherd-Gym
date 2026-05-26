"""Benchmark harness: run policies over many held-out seeds, report a metrics
table, and emit the comparison artifacts (per-policy clips, side-by-side panel,
swarm overlays) that slot into the SamSeesSheep loop format.

Metrics, per policy:
  success%      — episodes where the whole flock was penned
  steps(succ)   — mean steps to pen (successful episodes only; lower = faster)
  penned/N      — mean fraction of the flock penned
  return        — mean episode reward
  arousal       — mean flock arousal over the episode  (WELFARE: lower = gentler)
"""
from __future__ import annotations
import os
import numpy as np

from .env import ShepherdEnv, ShepherdConfig
from .baselines import RandomPolicy, GreedyDriver, FlankShepherd
from . import render as R


def evaluate(make_env, make_policy, episodes=30, seed0=1000):
    succ, steps_succ, rets, ars, penned = 0, [], [], [], []
    for k in range(episodes):
        env = make_env()
        pol = make_policy()
        env.reset(seed=seed0 + k)
        ret, ar, t = 0.0, 0.0, 0
        for _ in range(env.cfg.max_steps):
            _, r, term, trunc, info = env.step(pol(env))
            ret += r
            ar += info["mean_arousal"]
            t += 1
            if term or trunc:
                break
        won = bool(env.penned.all())
        succ += won
        if won:
            steps_succ.append(t)
        rets.append(ret)
        ars.append(ar / max(t, 1))
        penned.append(float(env.penned.mean()))
    return {
        "success": succ / episodes,
        "steps_succ": float(np.mean(steps_succ)) if steps_succ else float("nan"),
        "penned": float(np.mean(penned)),
        "return": float(np.mean(rets)),
        "arousal": float(np.mean(ars)),
    }


def markdown_table(results: dict) -> str:
    rows = ["| policy | success% | steps(succ) | penned/N | return | arousal (welfare ↓) |",
            "|---|---|---|---|---|---|"]
    for name, m in results.items():
        steps = "—" if np.isnan(m["steps_succ"]) else f"{m['steps_succ']:.0f}"
        rows.append(
            f"| {name} | {m['success']*100:.0f}% | {steps} | "
            f"{m['penned']*100:.0f}% | {m['return']:.1f} | {m['arousal']:.3f} |"
        )
    return "\n".join(rows)


def main(out_dir="out", episodes=30, cfg: ShepherdConfig | None = None):
    os.makedirs(out_dir, exist_ok=True)
    cfg = cfg or ShepherdConfig()
    make_env = lambda: ShepherdEnv(cfg)
    policies = {
        "random": lambda: RandomPolicy(seed=np.random.randint(1 << 30)),
        "greedy": lambda: GreedyDriver(),
        "flank":  lambda: FlankShepherd(),
    }

    # --- metrics ---
    results = {name: evaluate(make_env, fac, episodes=episodes) for name, fac in policies.items()}
    table = markdown_table(results)
    print("\n" + table + "\n")

    # --- artifacts ---
    rnd = R.Renderer(cfg.world)
    show_seed = 7
    named_frames = {}
    for name, fac in policies.items():
        frames, info = R.record_episode(make_env(), fac(), renderer=rnd, seed=show_seed, label=f"{name}")
        named_frames[name] = frames
        R.save_gif(frames, os.path.join(out_dir, f"{name}.gif"))
        R.save_mp4(frames, os.path.join(out_dir, f"{name}.mp4"))
        img, succ = R.swarm_overlay(make_env, fac, n_runs=40)
        img.save(os.path.join(out_dir, f"swarm_{name}.png"))
        print(f"  {name:8} demo: penned {info['penned']}/{info['n_sheep']} in {info['steps']} steps, "
              f"arousal {info['mean_arousal']:.3f}; swarm 40-run success {succ}/40")

    panel = R.compare_panel(named_frames)
    R.save_mp4(panel, os.path.join(out_dir, "compare_random_greedy_flank.mp4"))
    R.save_gif(panel, os.path.join(out_dir, "compare_random_greedy_flank.gif"), fps=20)

    md = ["# Shepherd-v0 baseline benchmark", "",
          f"_{episodes} held-out seeds · {cfg.n_sheep} sheep · field {cfg.world:.0f}m · "
          f"arousal-penalty weight r_arousal={cfg.r_arousal}_", "",
          table, "",
          "Artifacts: `compare_random_greedy_flank.mp4` (side-by-side), "
          "`swarm_*.png` (40-run ghost overlay), `<policy>.gif/.mp4` (single episodes).", "",
          "Arousal is the welfare signal (mean flock stress over the episode, lower is gentler) — "
          "the simulated analogue of the ear-angle readout in SamSeesSheep. A trained policy is "
          "interesting precisely when it pens the flock *and* keeps arousal low.", ""]
    with open(os.path.join(out_dir, "benchmark.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nWrote {out_dir}/benchmark.md and artifacts.")
    return results


if __name__ == "__main__":
    main()
