# Lesson 1 — The farm as an MDP (and how a reward goes wrong)

*Free lesson · ~20 min · no math until you've earned it*

---

## 1. The hook

I learned this herding my actual flock: the dog that "wins" fastest is almost never the
dog you want. The first time I let an over-keen young dog work my ewes, he pinned the whole
mob in the corner in under a minute — and three of them came out limping, all of them came
out wild-eyed, and the next morning they wouldn't come near the gate. He'd solved the wrong
problem perfectly. That gap — between *the job got done* and *the job got done well* — is
the entire reason this gym exists, and it's the first thing reinforcement learning forces
you to make precise.

## 2. The question

> If I want to **teach** a dog (a piece of software, in our case) to herd — not by writing
> down the rules, but by letting it try and rewarding it — what exactly do I have to define
> before "reward" even means anything?

## 3. Intuition first

Before any formalism, here's how I think about it standing in the paddock.

To describe a herding situation completely enough that a dog could *decide what to do*, I
need four things, and you already know all four from watching real dogs work:

1. **What the dog can see right now.** Where it is, where the sheep are, how spread out they
   are, the look of them. Call this the **situation**.
2. **What the dog can actually do.** A dog can't teleport into the fold or shout commands.
   It can move — in a direction, at a speed. That's the whole repertoire. Call these the
   **moves**.
3. **What happens after a move.** I step left, the sheep nearest me bolt right, the mob
   tightens or splits. The world *responds*. Call this **what follows**.
4. **What counts as good.** Sheep in the fold: good. Sheep panicked: bad. Taking all day:
   bad. Call this the **scorecard**.

Crucially, item 3 — *what follows* — is the bit beginners forget. The dog doesn't move the
sheep like chess pieces. It moves **itself**, and the sheep flee from it. You herd by
*positioning*, not pushing. Every good herding decision is really a prediction about how the
flock will react to where you put yourself.

If I have those four things nailed down, I have a complete "world" the dog can learn in. RL
people have a name for this bundle.

## 4. The formalism (now that it'll mean something)

That bundle is a **Markov Decision Process (MDP)**, written **(S, A, P, R, γ)**. In herding
terms:

- **S — the state space.** Every situation the dog could be in. In *this* gym
  (`shepherd_gym/env.py`), the dog's view is a **38-number vector** (`obs_dim = 38`). It
  holds the dog's own position and velocity, the flock's centre and how spread out it is, the
  fraction already penned, the direction to the fold — and, for the five nearest sheep, each
  one's relative position, velocity, and **ear angle**. More on ears in a second.
- **A — the action space.** What the dog can do. Here it's **2 continuous numbers**,
  `a ∈ [-1, 1]²` (`action_dim = 2`): a steering direction. The env turns it into velocity:
  `dog_vel = a · dog_speed` (`dog_speed = 2.4`), then moves the dog by `dog_vel · dt`. No
  teleporting, no whistles — just "lean this way."
- **P — the transition dynamics.** Given a state and an action, what state follows. This is
  the flock physics: the sheep are **boids** (they separate, align, and cohere with
  neighbours) *plus* a **flee** force that fires only when the dog is within the **flight
  zone** (`flight_zone = 7.0` m). Inside that radius, each sheep accelerates directly away
  from the dog, harder the closer the dog is. That's the lever you actually pull.
- **R — the reward.** The scorecard, as one number per step. We'll dissect it below — it's
  where everything interesting happens.
- **γ — the discount.** How much I care about the future versus right now (this gym trains
  at `γ = 0.99`). We give γ a full lesson next time; for now read it as "the dog isn't only
  greedy for the very next instant."

**The "Markov" promise.** The "M" means the state is supposed to be *enough*: the best move
depends only on the current state, not the whole history of how you got there. That's a
modelling choice, and it's worth being suspicious of — we'll poke a hole in it in the gotcha.

### The ear angle — and the thing the dog can't see

Here's the design decision in this gym I find most honest. Each sheep carries a hidden number
called **arousal** ∈ [0, 1] — its true acute stress. The dog **never observes arousal
directly.** What the dog gets instead is a noisy **ear angle**: calm sheep hold their ears up
(~25°), stressed sheep pin them back (~−10°), and the reading is jittered by measurement
noise. That mirrors real deployment, where you'd estimate stress from video, not read it off a
sensor stapled to the animal.

But the **reward is computed on the true arousal** — the real welfare cost. So the dog is
graded on something it can only *infer*. Sit with that: **the scorecard measures more than the
state reveals.** That tension is the soul of this environment, and it's why "just pen them
fast" is a trap.

### Reading the actual reward

Straight from `step()` in `env.py`, every tick the dog earns:

| term | value | what it rewards / penalises |
|---|---|---|
| `r_pen · newly` | **+1.0** per sheep | a sheep that *just* crossed into the fold |
| `r_fetch · fetch` | **+0.15** × progress | every sheep getting closer to the fold (shaping) |
| `−r_spread · spread` | **−0.02** × spread | letting the flock smear out |
| `−r_arousal · mean_arousal · dt` | **−0.5** × stress × 0.1 | **welfare: the flock being stressed, every step** |
| `−r_time` | **−0.01** | dawdling |
| `+r_success` | **+10.0** (once) | the whole flock penned |

The welfare term, `−0.5 · mean_arousal · dt`, is small *per step* — but it's charged **every
single step**, on the **true** arousal. It's a **dose**: keep the flock at high stress for 200
steps and it quietly outweighs a couple of penning bonuses. That's how "calm" is encoded here —
not as a one-time check, but as an integral of suffering over the whole episode.

## 5. Practice problem

> A student writes a dog whose only instinct is: **sprint at the flock's centre and drive it
> into the fold as fast as possible.** On paper this looks great — it maximises `r_pen`,
> `r_fetch`, and `r_success`, and it minimises `r_time`. Yet on the leaderboard it scores
> *worse* than a slower, wider-working dog. **Using only the reward table above, explain why —
> and name the specific term that punishes the sprinter.**

Try it before peeking. Three hints, each revealing a little more:

<details><summary>Hint 1</summary>

Four of the six reward terms reward speed and penning. Only some terms can *fall* when the dog
goes faster and gets closer. Which inputs do those terms depend on?
</details>

<details><summary>Hint 2</summary>

Look at what *causes* arousal (the transition dynamics, §4 / `_update_arousal`): being inside
the flight zone, and the dog **closing fast**. A sprint maximises both. Now look at which
reward term reads arousal.
</details>

<details><summary>Hint 3</summary>

The welfare term is `−r_arousal · mean_arousal · dt`, charged **every step**. It's tiny once,
but it integrates. Also ask: when a tight sprint scatters the mob, what happens to `r_fetch`
(per-sheep progress) and to the chance some sheep *never* get penned (`r_success` never fires)?
</details>

## 6. Worked solution

The sprinter loses on two fronts, and the reward table predicts both.

**Front one — the welfare dose.** Sprinting straight at the centre means the dog is
continuously inside the flight zone (`flight_zone = 7.0`) *and* closing fast. In the arousal
model, in-zone pressure scales with `w_pred_arousal = 1.3` and the closing-speed bonus
`w_closing = 1.0`, and arousal **rises fast but recovers slow** (`arousal_rise = 1.6` vs
`arousal_decay = 0.22`). So the sprinter pins `mean_arousal` near 1.0 for most of the episode.
The welfare term `−0.5 · mean_arousal · dt` then bleeds roughly `−0.05` per step; over a
couple hundred steps that's `−10`-ish — on the order of the entire `+10` success bonus, wiped
out by stress alone.

**Front two — it doesn't even pen reliably.** A hard charge at the *centre* makes the flee
forces shove sheep **outward in all directions** — the mob splits. Stragglers that pop out of
the group lose the `r_fetch` progress reward (they're now moving the wrong way) and, worse,
they may never be collected, so `penned.all()` is never true and the big `+r_success = +10`
**never fires.** The sprinter forfeits the jackpot it was optimising for.

So the very terms the sprinter thought it was maximising (`r_success`, `r_fetch`) are the ones
it sabotages, *while* the welfare term it ignored quietly drains its score. The slower, wider
dog keeps arousal moderate, keeps the mob whole, and actually collects the `+10`. **The reward
didn't reward "go fast." It rewarded "finish the job gently" — they only look the same until
the flock panics.**

## 7. The gotcha — reward hacking, and where "Markov" cracks

**Reward hacking: "panic-ram the pen."** Suppose I'd been lazier and written the reward as just
`+1 per penned sheep, −0.01 per step`, with no welfare term. Now the sprinter is *correct* — it
genuinely maximises that reward, by terrorising the flock into the fold. The agent didn't
misbehave; **I specified the wrong thing.** That's reward hacking: the optimiser gives you
exactly what you asked for, including the parts you didn't mean. The fix in this gym is the
explicit welfare dose (`r_arousal`) *and* a competition score where **speed only counts if you
penned the whole flock** (`scoring.py`) — so "stall to dodge stress" and "panic-ram for speed"
are both dead ends. When you design a reward, your job isn't to reward the goal; it's to make
every shortcut to the goal cost more than the honest path.

**Where Markov cracks.** Remember the promise that the state is "enough"? It isn't, quite. The
dog observes a **noisy ear angle**, not true arousal, and a single noisy frame can't tell a
genuinely calming flock from one that just happens to look calm this instant. A dog that
reacts to one frame can be fooled; a real solution needs to integrate over time (which is why
serious agents here carry memory, and why the *critic* is given the true arousal during
training). Whenever someone hands you an MDP, ask: *is the state actually sufficient, or is the
*real* state partly hidden?* Here, it's partly hidden by design.

## 8. Visual to build — annotated state frame

**Asset:** [`course/assets/l1_annotated_frame.py`](../assets/l1_annotated_frame.py) → `out/l1_annotated_frame.png`

**What it renders, all from the live env** (`render_state()` + the real config):

- One real mid-muster frame (the expert run a few steps in), on the green field with the
  brown **fold** at `pen_center`.
- The **dog** as the dark dot, with a **yellow action arrow** drawn as `a · dog_speed` — *the
  only thing the agent controls* (A).
- A **red circle of radius `flight_zone = 7.0`** around the dog: the boundary inside which
  sheep flee (the lever on P).
- Every **sheep tinted by its TRUE arousal** (calm cream → stressed red), with the single
  most-aroused sheep labelled `arousal=… (hidden from policy)` — making the
  scorecard-vs-state gap visible.
- A header strip stating the MDP at a glance: `obs=38-d (no true stress) · A: 2-d in [-1,1]² ·
  penned k/8 · mean arousal …`.

The point of the picture: you can *see* S (positions, spread), A (the arrow), the mechanism of
P (the flight-zone ring), and R's welfare input (the red tint) — and you can see that the
red, the thing being scored, is exactly the thing the dog isn't told.

---

### Why an interviewer cares

"Define the MDP for this problem" is the most common opening question in any RL interview, and
the follow-up is almost always **"how would you design the reward, and how could it be
gamed?"** If you can lay out (S, A, P, R, γ) for a concrete task *and* name a plausible
reward-hacking failure and a fix — as we just did with panic-ramming and the welfare dose —
you've shown the two things that separate someone who's read about RL from someone who's
shipped it. Bonus signal: noticing that the **state is only partially observed** (true arousal
hidden behind a noisy ear angle) tells the interviewer you know an MDP can quietly be a POMDP.

**Next lesson:** the reward is per-step, but good herding is about *consequences that arrive
later* — the eager move that feels great now and scatters the flock ten steps on. To reason
about that, we need **value functions** and the **Bellman equation**.
