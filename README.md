# shepherd-gym 🐕🐑

*A working notebook. I'm figuring this out as I go — the thesis is firm, the model
is evolving, and the honest edges are marked as such.*

A reinforcement-learning environment where a dog learns to herd a flock into a fold
**while keeping the flock calm**. It's the control-side sibling of
[SamSeesSheep][sss] — the project where I measure sheep stress from video (ear angle).

---

## Where this came from

SamSeesSheep answers a *measurement* question: **how stressed is this sheep, right
now?** — by segmenting every sheep's head/ears/nose and reading ear angle, a posture
that the affective-state literature ties to arousal ([Reefmann 2009][reef],
[Boissy 2011][boissy]).

Once you can *measure* stress, a different question won't leave you alone: a flock can
be penned a hundred different ways, and they are not equally kind. The frantic,
zig-zagging gather and the patient, wide arc both end with sheep in the fold — but one
of them spikes every animal's arousal and the other barely registers. So:

> **If I can measure flock stress, can I *optimise against it* — and will a learning
> agent discover handling that's gentler than naïve driving, without failing the job?**

That's the thesis. shepherd-gym is the testbed for it.

## The thesis, stated plainly

1. For any herding goal, many policies succeed; they differ enormously in welfare cost.
2. That cost is *measurable* (ear-angle arousal, via SamSeesSheep).
3. Therefore it's *optimisable* — and RL should find policies on a better
   **speed-vs-stress frontier** than heuristics.
4. The simulated stress model can be made progressively *empirical* by feeding it real
   recordings (see the [data roadmap](docs/data-roadmap.md)).

## What it is right now

- A pure-numpy, Gymnasium-compatible, PufferLib-ready env (`Shepherd-v0`).
- A **flight-zone arousal model**: each sheep's latent stress rises inside its flight
  zone, more when the dog is *closing fast*, and when it's *isolated* from flockmates;
  it recovers slowly. Grounded in flight-zone/FID + gregariousness work.
- An **asymmetric actor–critic** setup: the policy only sees what's sensable —
  positions, velocities, flock geometry, and a **noisy ear-angle observable** (the sim
  analogue of SamSeesSheep's output) — while the critic gets the true latent arousal,
  and the reward is the true integrated stress. *(Mirrors deployment: at run time you
  only have the noisy CV estimate; ground truth is a training-only crutch.)*
- Scripted baselines (random / greedy / a [Strömbom][strom]-style flank shepherd) and a
  content renderer (ghost trails, swarm-of-attempts overlays, model-vs-model panels) in
  the [SamSeesSheep][sss] visual grammar. → full mechanism in [`docs/welfare-model.md`](docs/welfare-model.md).

## What's real, and what I'm still guessing

I'd rather be honest than oversell:

- ✅ The **framing** (measure → optimise) and the **asymmetric setup** are sound.
- ✅ The **locomotion** has empirical roots — [Strömbom et al. 2014][strom] fit
  shepherding heuristics to GPS tracks of a real dog and flock.
- ⚠️ The **arousal parameters are theory-grounded, not yet fit to my data** — so this
  is a *methods demonstration*, not a claim about real Katahdins. Yet.
- ⚠️ I'm modelling **arousal/fear** (the axis ear posture indexes), **not pain** — the
  [Sheep Pain Facial Expression Scale][mcl] is a related but distinct construct.

Making it empirical is the whole roadmap below.

## Run it (no training stack needed)

Needs only `numpy`, `pillow`, `opencv-python-headless`:

```bash
python scripts/demo.py 40     # benchmark the baselines + write visuals to out/
```

Early result (40 held-out seeds, 8 sheep): the flank shepherd pens **80%** but drives
mean arousal to **~0.49** — exactly the gap a gentler learned policy should close.

## Compete 🏆

shepherd-gym doubles as a **Kaggle-style competition**: submit a policy, get scored on
held-out seeds, and climb a leaderboard that gets progressively harder. A policy is any
callable `policy(env) -> action ∈ [-1,1]²` (the same contract as the baselines) **or** a
trained actor checkpoint — scripted and learned entries compete on one board.

**Score (0–100)** = `50·penning + 35·welfare + 15·speed`. Speed credit is awarded only on
a full pen, so stalling to keep the flock calm can't win — you must pen the flock *and*
keep it gentle (welfare = `1 − mean flock arousal`). This is the speed-vs-stress thesis,
turned into a ranking.

**Difficulty ladder** (`shepherd_gym/tiers.py`): `t0_pasture` (tutorial) → `t1_paddock`
(baseline) → `t2_range` → `t3_skittish` → `t4_big_muster`, plus an adaptive **endless**
mode and locked future tiers (wolves, obstacles) awaiting new env mechanics. Clear a
tier's threshold to unlock the next.

```bash
python scripts/compete.py score  --tier t1_paddock --registry flank      # dry run
python scripts/compete.py submit --tier t1_paddock --author you --title "gentle ppo" \
                                 --checkpoint out/checkpoints/ra0.25_bc/actor_X.pt --render
python scripts/compete.py leaderboard          # the board (also at leaderboard/README.md)
python scripts/compete.py ladder  --author you # your unlocked tiers
python scripts/compete.py endless --registry flank   # how deep can it go?
```

The committed board (`leaderboard/`) ships seeded with the scripted baselines as the bars
to beat. PRs that add a `submissions/*.json` are auto-scored by CI. Full details:
[`submissions/README.md`](submissions/README.md).

## Roadmap

**Make the model empirical** — the calibration ladder, gated on footage:
[`docs/data-roadmap.md`](docs/data-roadmap.md). Rungs: ear-angle observable → flock
locomotion → (the digital-twin moment) dog→stress dynamics from co-recorded clips →
proxy validation in welfare units.

**Train & show the learning** —
- **P2:** PPO with an asymmetric critic (consumes `info["privileged_state"]`); training
  curves + checkpoint-ghosting footage (step-1k vs 10k vs 100k racing one seed).
- **P3:** predators (wolves) — herd *and* screen them off.
- **P4:** a small LLM as the policy — can a language model learn to herd from text obs?

## Sources & kin

- [SamSeesSheep][sss] — the measurement sibling (ear-angle welfare CV). *The reason this exists.*
- [Reefmann, Wechsler & Gygax 2009][reef] — ear postures ↔ emotional valence/arousal in sheep.
- [Boissy et al. 2011][boissy] — ear postures and emotional states.
- [McLennan & Mahmoud 2019][mcl] — Sheep Pain Facial Expression Scale (the *pain* construct, for contrast).
- [Strömbom et al. 2014][strom] — shepherding heuristics fit to real GPS data (the locomotion model).
- [Temple Grandin — flight zone & handling][grandin] — the basis of the arousal model.
- [PufferLib][puffer] — the fast-RL target this env is built to plug into.

*(Locomotion + handling links verified; the sheep-affect clinical refs are carried
over from the SamSeesSheep README — give those a final check before they go in a paper.)*

## Docs

`docs/welfare-model.md` (the science) · `docs/data-roadmap.md` (heuristic→empirical
ladder) · `docs/design-system.md` + `docs/design-brief.md` (the visual identity).

---

*This README will keep changing as the model earns its grounding. If a claim here
sounds confident, check whether the matching rung in the data roadmap is actually ticked.*

[sss]: https://github.com/antonemking/SamSeesSheep
[reef]: https://www.sciencedirect.com/science/article/pii/S0168159109001610
[boissy]: https://www.sciencedirect.com/science/article/pii/S0031938411000369
[mcl]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6523241/
[strom]: https://royalsocietypublishing.org/doi/10.1098/rsif.2014.0719
[grandin]: https://www.grandin.com/behaviour/principles/flight.zone.html
[puffer]: https://github.com/PufferAI/PufferLib
