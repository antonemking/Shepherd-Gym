# 🐕 Write your first sheepdog (a gentle intro to RL)

You don't need to know reinforcement learning to enter this competition. You need
to write one small function. By the end of this page you'll have a working sheepdog,
a score, and — if you want — a foothold on what RL actually *is*.

The job: **herd a flock of sheep into a fold, and keep them calm while you do it.**

---

## 1. The 20-line sheepdog

A **policy** is the brain of the dog. It's just a function: it looks at the field and
returns where the dog should move. Here's the whole thing:

```python
import numpy as np

class MySheepdog:
    name = "my_sheepdog"

    def __call__(self, env):
        free = env.sheep_pos[~env.penned]        # sheep not yet in the fold
        if len(free) == 0:
            return np.zeros(2)                   # all penned — stop

        flock_centre = free.mean(axis=0)
        # stand behind the flock, on the far side from the fold...
        behind = flock_centre + _unit(flock_centre - env.pen) * 3.0
        # ...and walk toward that spot. The sheep flee away from you — toward the fold.
        return _unit(behind - env.dog_pos)

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros(2)
```

That's a real, scorable entry. It's in [`starter_kit/my_sheepdog.py`](../starter_kit/my_sheepdog.py) —
copy it and go.

## 2. Run it

```bash
git clone https://github.com/antonemking/Shepherd-Gym
cd Shepherd-Gym
pip install -e .                                 # base deps only — no GPU, no torch

# score the starter kit
python scripts/compete.py score --tier t0_pasture \
    --callable starter_kit.my_sheepdog:MySheepdog
```

You'll see something like:

```
  starter_kit.my_sheepdog:MySheepdog [callable] on Pasture (tutorial)
    SCORE   58.7   pen=  90%  welfare=0.287  speed=0.246  success=  73%
```

Congratulations — you have a sheepdog. Now let's make it better.

## 3. What the dog can see

Inside `__call__`, `env` is the live field. Everything is in field-metre coordinates:

| You read | What it is |
|---|---|
| `env.dog_pos` | `[x, y]` — where your dog is |
| `env.sheep_pos` | array of `[x, y]` — every sheep |
| `env.penned` | bool array — which sheep are already in the fold |
| `env.pen` | `[x, y]` — centre of the fold |
| `env.cfg.pen_radius`, `env.cfg.world` | fold size, field size |

You return `[dx, dy]`, each in `[-1, 1]` — the direction to push the dog. The
environment turns that into motion. Sheep run *away* from the dog when it gets close
(their "flight zone"), which is the whole trick: you don't push sheep, you position
yourself so they flee where you want.

## 4. How you're scored

One number, 0–100, blends three things:

```
SCORE = 50·penning + 35·welfare + 15·speed
```

- **penning** — what fraction of the flock ended up in the fold.
- **welfare** — how *calm* you kept them (`1 − average stress`). This is the heart of
  the research: gentler handling scores higher.
- **speed** — how fast you finished — **but only if you penned the whole flock.**

That last rule matters: you can't win by dawdling around to keep the sheep calm. You
have to actually get the job done *and* be gentle. (That tension — fast vs. gentle —
is exactly what the research is about. See the [README](../README.md).)

## 5. Climb the leaderboard

The starter kit is a naive "push from behind." It works on a tight flock and scatters
a loose one. Three ideas, easiest first:

1. **Collect stragglers.** If one sheep is way off on its own, go get *behind that
   sheep* first, not the flock's centre. The built-in expert
   ([`FlankShepherd`](../shepherd_gym/baselines.py)) does exactly this — read it.
2. **Ease off when things are going well.** Pressure raises stress. If the flock is
   already calm and drifting fold-ward, back off — your welfare score thanks you.
3. **Don't crowd the gate.** Give the sheep room to flow into the fold.

When you beat the baselines, the next tier (`t1_paddock`, then `t2_range`, …) gets
harder: more sheep, bigger fields, jumpier flocks. Check what you've unlocked:

```bash
python scripts/compete.py ladder --author your-handle
```

## 6. ...and where the RL comes in

Everything above is a *hand-written* policy — you, the human, encoding the strategy.
**Reinforcement learning** flips that: instead of you writing the rules, an agent
*learns* them by trial and error, getting that same 0–100 score as its reward signal,
running thousands of episodes, and gradually discovering a strategy — sometimes a
gentler one than any heuristic. This repo ships a working RL trainer
([`scripts/train.py`](../scripts/train.py)); the research result is that RL found
herding **gentler than the textbook expert** ([`results/RESULTS.md`](../results/RESULTS.md)).

You can submit a trained model too (`--checkpoint actor.pt`) and it competes on the
same board as the hand-written policies. Writing a heuristic first is the best way to
understand what the RL agent is up against — and why "keep them calm" is hard.

## 7. Submit

```bash
python scripts/compete.py submit --tier t0_pasture --author your-handle \
    --title "my first sheepdog" \
    --callable starter_kit.my_sheepdog:MySheepdog --render
```

`--render` saves a replay GIF of your run (sheep tinted by stress — red = panicked).
Then open a PR adding a small JSON file under `submissions/` and CI scores you
automatically. Full submission rules: [`submissions/README.md`](../submissions/README.md).

Welcome to the trial. 🐑
