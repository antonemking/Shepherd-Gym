# Submitting an entry

Open a PR that adds a JSON file here. The `Score submission` GitHub Action scores it on
the chosen tier and publishes the result to the workflow run summary (the Action runs
submitted code, so it uses a read-only token and reports there rather than commenting on
the PR). To appear on the committed leaderboard, run `python scripts/compete.py submit ...`
locally and commit the updated `leaderboard/<tier>.json`.

## Format

```json
{
  "tier": "t1_paddock",
  "author": "your-handle",
  "title": "short description",
  "kind": "registry | callable | checkpoint",
  "ref":  "flank  |  my_module:MyPolicy  |  path/to/actor.pt"
}
```

- **registry** — one of the built-in baselines: `random`, `greedy`, `flank`.
- **callable** — a dotted path `module:attr` to a class or function implementing the
  policy contract `policy(env) -> action ∈ [-1,1]²` (see `shepherd_gym/baselines.py`).
  The module must be importable from the repo root.
- **checkpoint** — a PyTorch actor `state_dict` as produced by `scripts/train.py`. A
  checkpoint is **tier-specific** (its input width must match the tier's `obs_dim`).
  Add the `needs-torch` label to the PR so CI installs the training stack.

## Tiers

The ladder is defined in `shepherd_gym/tiers.py`: `t0_pasture` (tutorial) →
`t1_paddock` (baseline) → `t2_range` → `t3_skittish` → `t4_big_muster`, plus locked
future tiers (`t5_predators`, `t6_obstacles`) that need new env mechanics. Scripted
policies are tier-agnostic; learned checkpoints are submitted per tier.

## Scoring

Composite (0–100) = `50·penning + 35·welfare + 15·speed`. Speed credit is awarded only
on a full pen, so stalling to keep the flock calm can't win — you must actually pen the
flock *and* keep it calm. Welfare = `1 − mean flock arousal`.
