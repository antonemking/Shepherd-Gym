"""shepherd-compete — the Kaggle-style competition CLI for Shepherd-v0.

Score a policy on a tier, submit it to the leaderboard, view the board / your unlocked
ladder, or run the adaptive "endless" mode. Scripted entries (baselines or your own
callable) and trained .pt checkpoints compete on the same composite score.

    # dry-run scores (no store write)
    python scripts/compete.py score  --tier t1_paddock --registry flank
    python scripts/compete.py score  --tier t1_paddock --callable mymod:MyPolicy
    python scripts/compete.py score  --tier t1_paddock --checkpoint out/.../actor_X.pt

    # submit to the board (+ optional result-page artifacts)
    python scripts/compete.py submit --tier t1_paddock --author tone --title "gentle ppo" \
                                     --checkpoint out/.../actor_X.pt --render

    python scripts/compete.py leaderboard [--tier t1_paddock]
    python scripts/compete.py ladder  --author tone
    python scripts/compete.py endless --registry flank --max-tiers 6

The core path needs only the base deps; torch is imported lazily, only for --checkpoint.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataclasses import replace

from shepherd_gym import leaderboard as LB
from shepherd_gym import scoring
from shepherd_gym.adapter import Submission, load_policy
from shepherd_gym.env import ShepherdEnv
from shepherd_gym.tiers import AVAILABLE, get_tier

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _submission_from_args(a) -> Submission:
    chosen = [(k, v) for k, v in (("registry", a.registry), ("callable", a.callable),
                                  ("checkpoint", a.checkpoint)) if v]
    if len(chosen) != 1:
        raise SystemExit("pick exactly one of --registry / --callable / --checkpoint")
    kind, ref = chosen[0]
    return Submission(kind=kind, ref=ref,
                      author=getattr(a, "author", "") or "", title=getattr(a, "title", "") or ref)


def _resolve(sub: Submission, tier):
    probe = ShepherdEnv(tier.cfg)
    return load_policy(sub, probe)


def _print_result(label, res):
    print(f"\n  {label}")
    print(f"    SCORE {res['score']:6.1f}   "
          f"pen={res['P']*100:4.0f}%  welfare={res['W']:.3f}  speed={res['S']:.3f}  "
          f"success={res['success']*100:4.0f}%  arousal={res['arousal']:.3f}  (n={res['n']})")


# ----------------------------------------------------------------- commands
def cmd_score(a):
    tier = get_tier(a.tier)
    sub = _submission_from_args(a)
    pol = _resolve(sub, tier)
    res = scoring.score_tier(pol, tier, n_seeds=a.n_seeds)
    _print_result(f"{sub.title} [{sub.kind}] on {tier.name}", res)
    return res


def cmd_submit(a):
    tier = get_tier(a.tier)
    sub = _submission_from_args(a)
    if not a.author:
        raise SystemExit("--author is required to submit")
    pol = _resolve(sub, tier)
    res = scoring.score_tier(pol, tier, n_seeds=a.n_seeds)
    _print_result(f"{sub.title} [{sub.kind}] on {tier.name}", res)

    entry = LB.make_entry(tier.key, sub.author, sub.title, sub.kind, sub.ref, res)
    ranked = LB.record(entry)
    board = LB.write_board()
    place = next((i for i, e in enumerate(ranked, 1) if e["author"] == sub.author), None)
    print(f"\n  recorded → leaderboard/{tier.key}.json  (rank {place}/{len(ranked)})")
    print(f"  board → {os.path.relpath(board, ROOT)}")

    nxt_unlocked = res["score"] >= tier.unlock_score
    print(f"  {'✓ cleared' if nxt_unlocked else '✗ below'} unlock threshold "
          f"{tier.unlock_score:.0f} for the next tier.")

    if a.render:
        _render_result_page(tier, sub, pol)
    return res


def cmd_leaderboard(a):
    if a.tier:
        print("\n" + LB.tier_table(a.tier) + "\n")
    else:
        print("\n" + LB.render_board() + "\n")
    path = LB.write_board()
    print(f"(wrote {os.path.relpath(path, ROOT)})")


def cmd_ladder(a):
    unlocked = set(LB.unlocked_tiers(a.author))
    print(f"\n  Ladder for '{a.author}':")
    for t in AVAILABLE:
        mark = "🔓" if t.key in unlocked else "🔒"
        best = max([e["score"] for e in LB.load_tier(t.key) if e["author"] == a.author],
                   default=None)
        bs = f"best {best:.1f}" if best is not None else "—"
        print(f"    {mark} {t.key:14} {t.name:22} unlock@{t.unlock_score:>4.0f}  ({bs})")
    print()


def cmd_endless(a):
    """Adaptive depth: keep hardening the last available tier until the score drops below
    the floor. Reports the deepest level cleared."""
    sub = _submission_from_args(a)
    base = AVAILABLE[-1]
    print(f"\n  Endless mode from '{base.name}' — {sub.title} [{sub.kind}]")
    depth = 0
    for level in range(1, a.max_tiers + 1):
        cfg = _harden(base.cfg, level)
        tier = replace(base, key=f"endless_L{level}", name=f"Endless L{level}",
                       cfg=cfg, seed0=900_000 + level * 1000, n_seeds=a.n_seeds or 20)
        pol = _resolve(sub, tier)
        res = scoring.score_tier(pol, tier)
        status = "cleared" if res["score"] >= a.floor else "FAILED"
        print(f"    L{level}: {cfg.n_sheep:>2} sheep · world {cfg.world:4.0f} → "
              f"score {res['score']:5.1f}  ({status})")
        if res["score"] < a.floor:
            break
        depth = level
    print(f"\n  Endless depth cleared: {depth}\n")
    return depth


def _harden(cfg, level: int):
    return replace(
        cfg,
        n_sheep=cfg.n_sheep + 4 * level,
        world=cfg.world * (1 + 0.12 * level),
        flight_zone=cfg.flight_zone * (1 + 0.05 * level),
        sheep_max_speed=cfg.sheep_max_speed * (1 + 0.06 * level),
        max_steps=int(cfg.max_steps * (1 + 0.10 * level)),
        seed=None,
    )


def _render_result_page(tier, sub, pol):
    """Per-submission artifacts in the SamSeesSheep visual grammar (reuses render.py)."""
    from shepherd_gym import render as R
    from shepherd_gym.baselines import FlankShepherd

    out = ROOT / "leaderboard" / "runs" / tier.key / sub.author
    out.mkdir(parents=True, exist_ok=True)
    rnd = R.Renderer(tier.cfg.world)
    seed = tier.seed0

    frames, info = R.record_episode(ShepherdEnv(tier.cfg), pol, renderer=rnd, seed=seed,
                                    label=f"{sub.author}: {sub.title}")
    R.save_gif(frames, str(out / "episode.gif"))
    R.save_mp4(frames, str(out / "episode.mp4"))

    img, succ = R.swarm_overlay(lambda: ShepherdEnv(tier.cfg), lambda: pol,
                                n_runs=40, seed0=seed)
    img.save(str(out / "swarm.png"))

    flank = FlankShepherd()
    fframes, _ = R.record_episode(ShepherdEnv(tier.cfg), flank, renderer=rnd, seed=seed,
                                  label="flank (expert)")
    panel = R.compare_panel({sub.title: frames, "flank (expert)": fframes})
    R.save_mp4(panel, str(out / "compare_vs_flank.mp4"))
    R.save_gif(panel, str(out / "compare_vs_flank.gif"))

    with open(out / "result.md", "w") as f:
        f.write(f"# {sub.author}: {sub.title} — {tier.name}\n\n")
        f.write(f"Penned {info['penned']}/{info['n_sheep']} in {info['steps']} steps, "
                f"mean arousal {info['mean_arousal']:.3f}; swarm 40-run success {succ}/40.\n\n")
        f.write("![episode](episode.gif)\n\n![swarm](swarm.png)\n\n"
                "![vs flank](compare_vs_flank.gif)\n")
    print(f"  result page → {os.path.relpath(out, ROOT)}/")


# --------------------------------------------------------------------- main
def _add_policy_args(p, with_author=False):
    p.add_argument("--tier", required=True)
    p.add_argument("--registry", default="")
    p.add_argument("--callable", dest="callable", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=None,
                   help="override the tier's seed-block length (faster, for testing)")
    if with_author:
        p.add_argument("--author", default="")
        p.add_argument("--title", default="")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shepherd-compete", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("score", help="dry-run score on a tier")
    _add_policy_args(ps, with_author=True)

    pu = sub.add_parser("submit", help="score + record to the leaderboard")
    _add_policy_args(pu, with_author=True)
    pu.add_argument("--render", action="store_true", help="also render result-page artifacts")

    pl = sub.add_parser("leaderboard", help="print/regenerate the board")
    pl.add_argument("--tier", default="")

    pd = sub.add_parser("ladder", help="show an author's unlocked tiers")
    pd.add_argument("--author", required=True)

    pe = sub.add_parser("endless", help="adaptive depth mode")
    pe.add_argument("--registry", default="")
    pe.add_argument("--callable", dest="callable", default="")
    pe.add_argument("--checkpoint", default="")
    pe.add_argument("--author", default="")
    pe.add_argument("--title", default="")
    pe.add_argument("--max-tiers", dest="max_tiers", type=int, default=6)
    pe.add_argument("--n-seeds", dest="n_seeds", type=int, default=None)
    pe.add_argument("--floor", type=float, default=40.0)

    a = ap.parse_args(argv)
    return {"score": cmd_score, "submit": cmd_submit, "leaderboard": cmd_leaderboard,
            "ladder": cmd_ladder, "endless": cmd_endless}[a.cmd](a)


if __name__ == "__main__":
    main()
