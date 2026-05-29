"""🐕 Your first sheepdog — a starter policy for the Shepherd-Gym competition.

Copy this file, fill in the TODO, and you have a competition entry. No machine
learning required to start: a *policy* is just a function that looks at the field
and decides where the dog should move. That's it.

Run it:
    python scripts/compete.py score --tier t0_pasture --callable starter_kit.my_sheepdog:MySheepdog

Then try to beat the built-in baselines (random / greedy / flank) on the leaderboard.
The full walkthrough is in docs/tutorial.md.
"""
import numpy as np


class MySheepdog:
    """A policy is any object you can *call* with the environment, that returns an
    action: `policy(env) -> [dx, dy]`, each in [-1, 1] (the dog's desired move
    direction; the env scales it to the dog's top speed).

    What you can read off `env` (all in field-metre coordinates):
        env.dog_pos      -> np.array([x, y])         where the dog is
        env.sheep_pos    -> np.array of [x, y]       where every sheep is
        env.penned       -> bool array               which sheep are already in the fold
        env.pen          -> np.array([x, y])         the centre of the fold
        env.cfg.pen_radius, env.cfg.world            fold size, field size

    The GOAL: get every sheep into the fold (`env.pen`) — and keep them CALM while
    you do it. Sheep flee when the dog gets close, so charging straight in scatters
    them and spikes their stress. The score rewards penning AND gentleness AND speed.
    """

    name = "my_sheepdog"

    def __call__(self, env):
        # Which sheep still need herding?
        free = env.sheep_pos[~env.penned]
        if len(free) == 0:
            return np.zeros(2)            # everyone's in — relax

        # The flock's "middle".
        flock_centre = free.mean(axis=0)

        # --- The classic move: get BEHIND the flock (on the far side from the
        # fold), so when the sheep flee from you, they run toward the fold. ---
        pen_to_flock = _unit(flock_centre - env.pen)      # points fold -> flock
        stand_behind = flock_centre + pen_to_flock * 3.0  # 3 m behind the flock

        # Move the dog toward that spot.
        action = _unit(stand_behind - env.dog_pos)

        # TODO: this naive "push from behind" works on a tight flock but scatters a
        # loose one. Ideas to climb the leaderboard:
        #   1. If one sheep is far from the others, go COLLECT it first (get behind
        #      the straggler instead of the centre). See FlankShepherd in
        #      shepherd_gym/baselines.py for how the expert does this.
        #   2. Back off when the flock is already calm and moving the right way —
        #      pressure raises stress. Gentler handling scores higher (welfare!).
        #   3. Don't crowd the fold mouth; let sheep flow in.

        return action


def _unit(v):
    """A unit vector pointing the same way as v (or zero if v is ~zero)."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(2)


# So you can also run this file directly to sanity-check it before submitting:
#     python starter_kit/my_sheepdog.py
if __name__ == "__main__":
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from shepherd_gym import scoring
    from shepherd_gym.tiers import get_tier

    tier = get_tier("t0_pasture")
    result = scoring.score_tier(MySheepdog(), tier, n_seeds=20)
    print(f"\n  {MySheepdog.name} on {tier.name}:")
    print(f"    SCORE {result['score']:.1f}   "
          f"penned {result['P']*100:.0f}%  welfare {result['W']:.3f}  "
          f"success {result['success']*100:.0f}%\n")
    print("  Beat the baselines, then submit! See docs/tutorial.md.")
