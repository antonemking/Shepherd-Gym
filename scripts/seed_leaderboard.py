"""Seed the committed reference leaderboard: the scripted baselines on every available
tier — the bars a submitted policy has to beat. Run once to (re)generate the board.

    python scripts/seed_leaderboard.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from shepherd_gym import leaderboard as LB
from shepherd_gym import scoring
from shepherd_gym.baselines import REGISTRY
from shepherd_gym.tiers import AVAILABLE

TITLES = {"random": "random walk", "greedy": "greedy driver", "flank": "flank expert (Strömbom)"}


def main():
    for tier in AVAILABLE:
        for name, pol in REGISTRY.items():
            res = scoring.score_tier(pol, tier)
            entry = LB.make_entry(tier.key, f"baseline:{name}", TITLES[name],
                                  "registry", name, res)
            LB.record(entry)
            print(f"  {tier.key:14} {name:7} score={res['score']:5.1f} "
                  f"(pen {res['P']*100:.0f}% welfare {res['W']:.2f} success {res['success']*100:.0f}%)")
    path = LB.write_board()
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
