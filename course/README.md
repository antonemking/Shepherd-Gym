# Shepherd-Gym — Reinforcement Learning, taught through herding

A text-first, Interview-Cake-style intro to reinforcement learning, grounded in **this repo's
real environment** (`shepherd_gym/env.py`). You herd a flock into a fold while keeping it
calm — and learn RL from the inside as you go. The first three lessons are free and complete on
their own; you never have to pay to get genuine value out of them.

Each lesson follows the same shape: a farm hook, a question, intuition **before** any math, the
formalism in herding terms, a practice problem with progressive hints, a worked solution, a
"when this breaks" gotcha, and a visual built from the real env. Every lesson ends with the
interview question it prepares you for.

## The lessons

1. **[The farm as an MDP](lessons/L1_the_farm_as_an_mdp.md)** — states, actions, transitions,
   reward, discount; designing a reward and how it gets hacked ("panic-ram the pen").
2. **[Value, Bellman, and credit assignment](lessons/L2_value_functions_and_bellman.md)** — why
   the move that looks bad now is the good one; how reward flows backward from the goal.
3. **[Model-free learning & explore vs exploit](lessons/L3_model_free_and_exploration.md)** —
   learning by trying (the Q-learning update / TD error), the exploration gamble, and the hook
   into actually training a policy in the gym.

## Grounding

[`SPEC.md`](SPEC.md) is the exact MDP extracted from the code (state/action/reward/dynamics,
with real variable names and values) — read it if you want to verify any claim in the lessons
against `shepherd_gym/env.py`.

## Assets

All figures are generated from the **live environment** (deps: `numpy`, `pillow`; `matplotlib`
for L2/L3 — `pip install -e . && pip install matplotlib`):

```bash
python course/assets/l1_annotated_frame.py      # annotated real-env state frame
python course/assets/l2_value_iteration.py       # value heatmap + hand-checkable value table
python course/assets/l3_explore_vs_exploit.py    # greedy-vs-flank comparison + exploration trace
```

Outputs land in `course/assets/out/`.

> Note: the L2 value-iteration example runs on a small **derived teaching gridworld** (the real
> env is continuous and has no gridworld mode); its reward terms are mapped onto the real env and
> it's labelled as an abstraction throughout. See `SPEC.md` for the honest accounting.
