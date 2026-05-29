# Lesson 3 — Learning by trying: model-free RL and the explore/exploit gamble

*Free lesson · ~25 min · the doorway to actually training a dog*

---

## 1. The hook

I learned this herding my actual flock: a good young dog teaches itself, and it does it by
being *wrong* on purpose. Mine would try a line that didn't work, watch the mob squirt out the
side, and you could see the lesson land — *that approach was worse than I thought.* Nobody gave
him the physics of sheep. He never modelled flee forces. He just **tried things, felt the
outcome, and updated.** That's model-free reinforcement learning, and the only hard part is the
gamble underneath it: every run, do you do the thing that's worked, or risk a new thing that
might work better?

## 2. The question

> In Lesson 2 we computed values *assuming we knew how the sheep would react*. A real dog
> doesn't. So: how can an agent learn which moves are good **purely from the rewards it
> experiences** — no model of the flock at all — and how does it decide when to **trust** what
> it's learned versus **try** something new?

## 3. Intuition first

Two ideas, no math yet.

**Learning from surprise.** Picture grading not positions (Lesson 2) but *move-from-a-position*
— "swing wide *from here*." You walk in with a guess of how good that is. You make the move,
the muster plays out, and you see the actual result. If it turned out **better than you
guessed**, you nudge your estimate of that move **up**; worse, **down**. The size of the nudge
is proportional to your **surprise** — the gap between what you expected and what you got. Do
that a few thousand times across a flock and your estimates converge on the truth, and you never
once needed to know the physics. You only needed to *try the move and feel the gap.* That gap —
expectation minus reality — is the single most important quantity in this whole field.

**The gamble: explore vs exploit.** Here's the bind. Early on, your estimates are garbage. The
move that *looks* best is probably just the one you happened to try in an easy situation. If you
only ever do the current-best move (**exploit**), you'll never discover the wide cast that's
actually superior — you're stuck polishing a mediocre habit. But if you only ever try random new
things (**explore**), you never cash in on what you've learned; you flail forever. Every herding
run is a bet: spend it confirming what works, or spend it gathering information that might make
you better tomorrow? Get the balance wrong in either direction and you fail — too greedy and you
plateau, too random and you never converge. *That* is the explore/exploit trade-off, and it
never fully goes away.

## 4. The formalism — the Q-learning update, in herding terms

Give a name to "how good is move `a` from state `s`": the **action-value** `Q(s, a)`. (Value
`V(s)` from Lesson 2 was "how good is this position"; `Q` is "how good is *this move from* this
position." If you know `Q`, the best policy is trivially "pick the move with the highest `Q`.")

Now the famous update. After taking move `a` in state `s`, getting reward `r`, and landing in
`s'`:

```
Q(s, a)  ←  Q(s, a)  +  α · [ r + γ · maxₐ' Q(s', a')  −  Q(s, a) ]
                            └──────────── TD error δ ────────────┘
```

Unpack it as the surprise story from §3:

- `r + γ · maxₐ' Q(s', a')` is **what just happened**: the reward you got, plus the discounted
  value of the best you can do from where you landed. It's a fresh, partly-real estimate of
  `Q(s, a)` — better than your old guess because it contains an actual observed reward `r`.
- `Q(s, a)` is **what you expected**.
- Their difference, `δ`, is the **TD (temporal-difference) error** — the *surprise*. Positive:
  the move beat expectations. Negative: it disappointed.
- `α` (the **learning rate**) is how big a nudge you take toward the surprise — how much you let
  one experience move your belief.

Notice what's **not** in that equation: any model of the sheep, any `P`, any flee forces. You
never predict the next state; you just **use the one you landed in.** That's what *model-free*
means — you learn `Q` straight from `(s, a, r, s')` tuples the world hands you. And the
`γ · maxₐ' Q(s', a')` term is Lesson 2's Bellman recursion sneaking back in: credit still flows
**backward** from rewarding states, one TD update at a time, instead of by sweeping a known
table. Run enough episodes and `Q` converges to the same optimal values — without ever knowing
the dynamics.

**The explore/exploit knob, made concrete.** To collect those `(s, a, r, s')` tuples you need a
behaviour policy. The classic is **ε-greedy**: with probability `1 − ε` take your current-best
move (**exploit**), and with probability `ε` take a random move (**explore**). Start `ε` high
(you know nothing — wander) and **anneal it down** as `Q` sharpens (trust yourself more). In a
*continuous*-action gym like this one you don't flip a coin between discrete moves; you add
**noise to the action** and shrink the noise over training. That's exactly what this repo's PPO
trainer does — its policy is a Gaussian whose spread is a learned parameter (`Actor.log_std` in
`scripts/train.py`); early on the spread is wide (explore), and training narrows it (exploit).
Same knob, continuous dial.

## 5. Practice problem

The repo ships two scripted dogs (no learning — fixed strategies — but perfect for seeing what a
*learner* is choosing between). On seed 14, rolled out by
[`course/assets/l3_explore_vs_exploit.py`](../assets/l3_explore_vs_exploit.py):

| dog | strategy | result | mean arousal |
|---|---|---|---|
| `GreedyDriver` | always shove the flock's centre at the fold | **failed, 6/8 penned** | **0.791** |
| `FlankShepherd` | swing wide to collect stragglers, *then* drive | **penned 8/8** | **0.423** |

> An RL agent learning `Q` from scratch will, early on, almost always discover the *greedy*
> behaviour first — charge the centre — long before it finds the flanking behaviour. **Using the
> explore/exploit idea, explain (a) why greedy-charging is the move a fresh learner stumbles onto
> first, and (b) what specifically has to happen for it to ever discover the better flanking
> strategy instead of getting stuck.**

<details><summary>Hint 1</summary>

Which behaviour gets *some* immediate reward almost every time you try it, even by accident?
Charging the centre usually pens at least a few sheep (partial `r_pen`, positive `r_fetch`).
What does that do to its `Q` estimate relative to moves that initially look like "lose ground"?
</details>

<details><summary>Hint 2</summary>

The flanking move *starts* by going **away** from the fold (the wide cast), so its first few
steps earn **negative** `r_fetch` — it looks worse on the immediate reward. A purely greedy
learner (low `ε`) will down-weight it before its long-term payoff ever shows up. What knob
prevents that premature dismissal?
</details>

<details><summary>Hint 3</summary>

It has to **explore** the wide-cast line *enough times, despite its bad-looking early steps*, to
experience the `+10` it eventually sets up — so the TD updates can carry that credit backward
onto the wide cast. That needs high-enough `ε` (or wide-enough action noise) early, and a
discount `γ` near 1 so the delayed `+10` isn't crushed. (The repo's `RESULTS.md` reports PPO
**from scratch basically failed** — ~16% success — until they *warm-started* it by cloning the
flank expert. That's this exact wall.)
</details>

## 6. Worked solution

**(a) Why greedy comes first.** A fresh learner explores, stumbles into charging the centre, and
gets **rewarded right away** almost every time: even a clumsy charge pens a sheep or two
(`r_pen`) and shrinks distances-to-fold (`r_fetch`). Those positive rewards push `Q(charge)` up
fast. The flanking move, by contrast, *opens* with the wide cast — steps that earn **negative**
`r_fetch` because the dog is moving away from the fold — so its `Q` looks **bad early**, before
any of its eventual `+10` materialises. Greedy is the local optimum that's *easy to find*: it
pays immediately and often. That's why the learner finds it first, and why — without help — it
parks there.

**(b) What it takes to escape.** The only way the flanking strategy's true worth gets onto its
`Q` value is for the agent to **actually ride the wide cast all the way to the payoff, repeatedly**
— to *explore* that line despite its discouraging first steps, experience the clean 8/8 pen and
its `+10`, and let the TD error carry that credit **backward** (Lesson 2) onto the early wide-cast
moves until `Q(flank) > Q(charge)`. Two things must hold: **enough exploration** (high `ε` / wide
action noise early, so the agent doesn't abandon the line before it pays off) and **a long enough
horizon view** (`γ = 0.99`, so the delayed `+10` still outweighs the early `−r_fetch`). Get either
wrong and the agent rationally — and permanently — prefers the worse strategy. This is not
hypothetical: the repo's own result is that **PPO from scratch couldn't clear this wall (~16%
success)**, and the unlock was to **behaviour-clone the flank expert first** (hand the learner the
wide-cast trajectory so it doesn't have to discover it blind) and *then* let RL refine it. The
exploration problem was the whole ballgame.

And the punchline that makes this gym worth training on: once RL *did* get past the wall, it
found herding **gentler than the textbook expert** — flank pens 75% at arousal **0.459**, while
PPO at `r_arousal = 0.25` pens 63% at arousal **0.188**, roughly **59% less stress**
(`results/RESULTS.md`). Trial-and-error didn't just match the heuristic; tuned for welfare, it
*beat* it on the axis I actually care about. That's the moment the whole approach earns its keep.

## 7. The gotcha — exploration that quietly poisons what you're measuring

The trap with "just add noise to explore": **in this gym, exploration changes the very thing
you're scored on.** A random-ish, jittery dog doesn't just gather information — it spends the
whole time *inside the flock's flight zone, lurching unpredictably*, which is precisely the
worst case for the arousal model (in-zone pressure + a closing-speed bonus, with stress that
rises fast and decays slow). So naïve exploration **spikes the welfare cost** you're trying to
minimise. You can watch it in the exploration-trace figure: crank the action noise up and the
dog's path turns into a frantic scribble across the field. The deeper lesson is that
exploration is never free — every exploratory move is a real move in a real episode with real
consequences (here, real stress). Good RL **anneals** exploration (start wide, narrow as `Q`
sharpens) and, in welfare-sensitive settings, biases it toward *informative-but-safe* moves
rather than pure randomness. "Explore more" is not a universal cure; in a system where acting is
itself costly, *how* you explore is part of the problem.

## 8. Visual to build — two-policy comparison + an exploration trace

**Asset:** [`course/assets/l3_explore_vs_exploit.py`](../assets/l3_explore_vs_exploit.py) →
`out/l3_two_policy.png` and `out/l3_exploration_trace.png`

**Figure A — `l3_two_policy.png` (greedy vs flank, one seed, real env):**
- Two field panels side by side: `GreedyDriver` and `FlankShepherd` on the **same seed (14)**,
  each showing the **dog's full path**, the dog's start, and the **final sheep tinted by true
  arousal** (cream → red). Greedy's panel shows a frantic line and **stranded red sheep** outside
  the fold (failed, 6/8, arousal 0.791); flank's shows wide collecting loops and a clean **8/8**
  pen at much lower stress (0.423).
- A third panel plots **mean arousal** and **penned-fraction over time** for both: greedy's
  arousal pinned high with penning stalled below 1.0; flank's arousal lower and penning climbing
  to 1.0. This is the "eager dog scatters and stresses" failure vs the patient policy, *measured*.

**Figure B — `l3_exploration_trace.png` (the explore/exploit knob):**
- Three panels of the **same flank policy** with growing Gaussian **action noise**
  `ε ∈ {0.0, 0.35, 0.9}`. `ε = 0` is the clean, deterministic line (pure exploit); larger `ε`
  turns the dog's path into an ever-wilder scribble (explore) — a direct picture of the noise a
  learner injects early in training (and the same `log_std` dial PPO anneals here).

---

### Why an interviewer cares

This lesson is dense with the things RL interviews actually probe. **(1) Model-based vs
model-free:** being able to say "Q-learning needs no transition model — it learns straight from
`(s, a, r, s')`" and write the update with the **TD error** labelled is table-stakes for any RL
role. **(2) The explore/exploit trade-off** is a near-guaranteed question; ε-greedy with
annealing (and its continuous cousin, action-noise / entropy that decays) is the expected answer,
and bonus points for naming *why* it's hard — the agent can rationally lock onto a worse local
optimum, exactly the greedy-charge trap. **(3) On-policy vs off-policy, sparse/delayed reward,
and reward shaping** all fall out of the flank-discovery story (the wide cast's delayed `+10`, the
warm-start that got PPO past the exploration wall). If you can tell that story concretely — *why
the easy strategy is found first, what exploration and `γ` it takes to escape it, and what it
costs to explore in a welfare-sensitive world* — you're demonstrating the judgment that separates
"knows the algorithms" from "can actually get an agent to learn something."

**Where this goes next:** you now have the whole spine — the farm is an **MDP** (L1), good play
is **value** propagated by **Bellman** (L2), and an agent can **learn it model-free** by trying,
surprising itself, and balancing explore against exploit (L3). The natural next step is to *do
it*: `scripts/train.py` runs exactly this — PPO, `γ = 0.99`, a Gaussian policy whose exploration
noise anneals — and `results/RESULTS.md` is what comes out the other side. Clone the starter dog
in `starter_kit/my_sheepdog.py`, beat the baselines, then let a policy *learn* to out-herd your
hand-written one. That's the trial. 🐑
