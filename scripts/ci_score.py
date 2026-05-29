"""Score a single submission JSON and print ONE markdown table row (used by CI).

    python scripts/ci_score.py submissions/my_entry.json

Submission JSON: {"tier", "author", "title", "kind", "ref"}. Prints a row; on error,
prints a row with the error so the PR comment shows what went wrong rather than failing
the whole job.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from shepherd_gym import scoring
from shepherd_gym.adapter import Submission, load_policy
from shepherd_gym.env import ShepherdEnv
from shepherd_gym.tiers import get_tier


def main(path: str):
    name = pathlib.Path(path).stem
    try:
        spec = json.load(open(path))
        tier = get_tier(spec["tier"])
        sub = Submission(kind=spec["kind"], ref=spec["ref"],
                         author=spec.get("author", name), title=spec.get("title", name))
        pol = load_policy(sub, ShepherdEnv(tier.cfg))
        r = scoring.score_tier(pol, tier)
        cleared = "✓" if r["score"] >= tier.unlock_score else "✗"
        print(f"| {sub.title} ({sub.author}) | {tier.name} {cleared} | "
              f"**{r['score']:.1f}** | {r['P']*100:.0f}% | {r['W']:.3f} | {r['success']*100:.0f}% |")
    except Exception as e:  # surface the error in the PR comment, don't crash CI
        print(f"| {name} | — | ⚠️ {type(e).__name__} | {str(e)[:80]} | | |")


if __name__ == "__main__":
    main(sys.argv[1])
