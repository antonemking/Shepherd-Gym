# Lesson 2 — Value, Bellman, and the move that pays off ten steps later

*Free lesson · ~25 min · one equation, fully earned*

---

## 1. The hook

I learned this herding my actual flock: the best move often *feels* like the wrong one. There's
a moment, gathering sheep off the top of a hill, where the dog has to swing **wide and away**
from the fold — losing ground, by every instinct — so that it ends up *behind* the stragglers
and can bring the whole mob down clean. A young dog won't do it. It takes the greedy line
straight at the sheep, splits them, and spends the next two minutes paying for the ten seconds
it saved. Knowing that the wide cast is worth it — even though it looks worse right now — is
exactly what a **value function** is.

## 2. The question

> The reward arrives one step at a time, but the *consequences* of a move can show up much
> later. How can a dog (or an agent) tell that a move which looks bad this instant is actually
> the good one — and how does it ever learn **which** earlier move deserves the credit when
> things finally go right?

## 3. Intuition first

Forget equations. Think about how you'd grade *positions* on the field, not moves.

Stand anywhere in the paddock and ask: **"From here, playing well, how does the rest of this
muster go?"** Some positions are obviously great — dog tucked behind a tight mob, fold dead
ahead, sheep drifting in. Some are obviously grim — flock split, half of them bolting for the
far fence. You can feel the *worth* of a position before a single further move is made.

That felt worth is the **value** of a state: not what you score right now, but the total you
expect to score *from here to the end*, if you keep herding well. The wide cast looks bad on
the **immediate** scorecard (you lost ground, `r_fetch` went negative for a step) but it moves
you to a **high-value position** (behind the mob, about to bring them all in). Greedy-for-now
and good-in-total are different things, and value is the word for the second one.

Now the second half of the question — **credit assignment**. Say the muster ends well: all
eight penned, `+10`. *Which* of the two hundred moves earned that? The last nudge through the
gate? Sure, a bit. But really it was that unglamorous wide cast a hundred steps back that set
the whole thing up. The reward landed at the end; the *cause* was much earlier. Learning has to
somehow pass credit **backwards** from the payoff to the move that actually deserves it. That
backward flow is the engine of everything in RL, and it has a beautifully simple form.

## 4. The formalism — the Bellman equation, in herding terms

Define the **value** of a state `s` under some policy (some way of herding) as the expected
**discounted** sum of all future reward starting from `s`:

```
V(s) = E[ r_t + γ·r_{t+1} + γ²·r_{t+2} + … ]      starting from state s
```

That `γ = 0.99` (the gym's actual training discount) is last lesson's discount, doing its job:
a reward `n` steps away is worth `γⁿ` of one now. At `γ = 0.99`, the `+10` for penning is still
worth `10 · 0.99¹⁰⁰ ≈ 3.7` from a hundred steps out — faded, but absolutely worth working
toward. Small γ would make the dog short-sighted (it'd never pay ten steps now for a reward
later); γ near 1 lets it plan the wide cast.

Writing out that infinite sum is hopeless. The trick — **Richard Bellman's** trick — is to
notice it's **recursive**. The value of *here* is just the reward you get stepping to the
**next** place, plus the (discounted) value of **that** place:

```
V(s) = r(s, a) + γ · V(s')        for the move a that takes you from s to s'
```

And if you're herding *optimally*, you pick the move that makes that as large as possible:

```
V*(s) = max over moves a of [ r(s, a) + γ · V*(s') ]      ← the Bellman optimality equation
```

Read it in plain English: **"the worth of this position is the best single move's immediate
reward, plus the worth of wherever that move lands you."** That's it. A position is valuable if
a good move from it leads to another valuable position — recursion all the way down to the
fold, where the `+10` lives.

This recursion is *also* the answer to credit assignment. Because value is defined in terms of
the **next** state's value, the `+10` at the fold seeps **one step backward** into the cells
beside the fold, then another step into the cells beside *those*, and so on — a tide of worth
flowing outward from the goal. Repeatedly applying the Bellman equation until the numbers stop
changing is called **value iteration**, and watching it run *is* watching credit propagate back
from the payoff to the moves that set it up.

## 5. Practice problem

I can't run honest value iteration on the full continuous env by hand — but I built a small
**teaching gridworld** that mirrors it (one cell per metre of the real 20 m field, the fold at
the discretised `pen_center = (16,16)`, the same `+10` for reaching it, the same `−0.01`
per-step time cost, and a stress cost near the fences). It's in
[`course/assets/l2_value_iteration.py`](../assets/l2_value_iteration.py), and it dumps a
hand-checkable table near the fold:

```
(12,16) V= 9.66 E   (13,16) V= 9.77 E   (14,16) V= 9.88 E   (15,16) V= 9.99 E   (16,16) V= 0.00 FOLD
(12,15) V= 9.56 N   (13,15) V= 9.66 N   (14,15) V= 9.77 N   (15,15) V= 9.88 N   (16,15) V= 9.99 N
```

> The fold cell `(16,16)` is terminal, so its value is **0** (there's no future left). The
> `+10` is paid on the *transition into* it. Given that, and `γ = 0.99`, `r_time = 0.01`, with
> **no** stress on these particular cells: **work out `V*(15,16)` and then `V*(14,16)` by hand**
> from the Bellman optimality equation, and check them against the table.

<details><summary>Hint 1</summary>

From `(15,16)` the best move is **E**, stepping straight into the fold. That transition is
*terminal* — there's no `γ·V(s')` bootstrap afterward, because the fold's value is 0 and the
episode ends. So `V*(15,16)` is just the reward of that one step.
</details>

<details><summary>Hint 2</summary>

The reward of a step is `−r_time − stress(landing cell) + (r_success if you land in the fold)`.
For the step `(15,16) → (16,16)`: stress there is 0, you land in the fold, so
`r = −0.01 − 0 + 10`.
</details>

<details><summary>Hint 3</summary>

For `V*(14,16)` the best move E lands you in `(15,16)` — **not** terminal — so now you *do*
bootstrap: `V*(14,16) = [−0.01 − 0] + 0.99 · V*(15,16)`. Plug in the number you just found.
</details>

## 6. Worked solution

**`V*(15,16)`** — best move is E, into the fold (terminal):
```
V*(15,16) = r(step into fold) = −r_time − stress + r_success
          = −0.01 − 0 + 10
          = 9.99   ✓  (matches the table)
```
No discount term, because the fold is terminal — nothing follows, so there's no `γ·V(s')`.

**`V*(14,16)`** — best move is E, into `(15,16)`, which is *not* terminal, so we bootstrap on
its value:
```
V*(14,16) = [−r_time − stress(15,16)] + γ · V*(15,16)
          = [−0.01 − 0] + 0.99 · 9.99
          = −0.01 + 9.8901
          = 9.88   ✓  (matches the table)
```

Step back once more and you'd get `−0.01 + 0.99 · 9.88 = 9.77`, then `9.66`, then `9.56` going
down a row — exactly the staircase in the table. **That staircase *is* credit assignment made
arithmetic:** the `+10` at the fold, discounted by `γ` and nicked by `r_time` at each hop,
ripples outward so that every cell knows how good it is *by virtue of how good its best
neighbour is.* No cell was told its value directly except the fold; every other number was
*earned backward* from the goal. That's the whole idea, and it's why the heatmap glows brightest
at the fold and fades smoothly outward.

## 7. The gotcha — when "greedy on value" still scatters the flock

Two traps, both of which bite real agents.

**Trap one: the value is only as honest as the model.** Value iteration above assumed I *knew*
the dynamics — that stepping E lands me E. In the **real** env I don't: I move the *dog*, and
the *sheep* react through the flee forces, which depend on the whole flock's geometry. The true
"state" isn't a grid cell, it's the dog **and** every sheep's position and hidden arousal — a
38-plus-dimensional continuous thing. You can't enumerate it, so you can't run table-based value
iteration on it; you have to **learn** an approximate value function from experience. That's
exactly the jump we make next lesson. The gridworld is a faithful picture of the *idea*, not a
solver for the real problem — and that gap is the whole reason model-free RL exists.

**Trap two: acting greedily on a value estimate can look like the eager-dog mistake.** Suppose
your learned value function is a bit *wrong* near the flock — it over-values "get close to the
mob" because close positions are usually good. An agent greedy on that estimate charges in,
triggers the flee forces, and **scatters the flock** — the same failure as the young dog,
except now it's caused by a *miscalibrated value estimate*, not impatience. The cure isn't to
abandon value; it's to keep improving the estimate from real outcomes (the scatter teaches it
those close states weren't so valuable after all) and to **explore** rather than blindly trust
early numbers. Which is precisely Lesson 3.

## 8. Visual to build — value heatmap over the pasture

**Asset:** [`course/assets/l2_value_iteration.py`](../assets/l2_value_iteration.py) →
`out/l2_value_heatmap.png` (+ `out/l2_value_table.txt`, the hand-checkable block above)

**What it renders:**

- The 20×20 derived gridworld over the real field extent, each cell coloured by **`V*(cell)`**
  (viridis: dark = low, bright = high). The map glows brightest at the fold and fades outward —
  the Bellman tide, frozen.
- The **greedy policy** as white arrows in every cell (the `argmax` move), so you can see that
  acting greedily on `V*` always points you "uphill" toward the fold.
- The **fold** ringed in yellow at `pen_center`, with `pen_radius` to scale.
- A **red dashed contour** outlining the near-fence stress band (the welfare cost, weight
  `r_arousal`), so you can see the policy's arrows bend *away* from the rails — value routing
  around stress, just like a good dog keeps the mob off the fence.
- A note that the terminal fold cell sits at `V = 0` (no future), so the `+10` lives on the
  transitions *into* it — the subtlety the practice problem hinges on.

---

### Why an interviewer cares

If an interviewer asks you to "write the Bellman equation" and you can produce
`V*(s) = maxₐ [ r + γ·V*(s') ]` **and explain it as 'the value of here is the best move's reward
plus the value of there,'** you're ahead of most candidates. The deeper signal is the two things
this lesson drilled: (1) the **credit-assignment** story — that value iteration propagates reward
*backward* from the goal, which is why delayed rewards are learnable at all; and (2) knowing the
**limits** — table-based value iteration needs a known, enumerable model, which the real,
continuous, partially-observed herding task doesn't have, forcing you toward *learned*,
*model-free* methods. Being able to hand-trace a two-cell backup (the `9.99 → 9.88` we just did)
on a whiteboard is the kind of concrete fluency that ends the value-function portion of an
interview early, in your favour.

**Next lesson:** we drop the assumption that we know the dynamics. The dog has to **learn**
which moves are good *purely from trying them* — which forces the oldest dilemma in the field:
do you **exploit** the move you think is best, or **explore** a new one that might be better?
That's Q-learning and the explore/exploit trade-off, and it's the doorway to actually *training*
a policy in this gym.
