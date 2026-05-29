# 🐕🐑 Shepherd-Gym Leaderboard

Composite score (0–100) = `50·penning + 35·welfare + 15·speed`. Speed credit is awarded only on a full pen, so stalling to dodge stress can't win. Welfare = `1 − mean flock arousal` (gentler handling scores higher).

### Pasture (tutorial) — `t0_pasture`
_Small, calm flock with a roomy flight zone — learn the controls. · 30 held-out seeds · unlock next at 45_

| # | author | entry | score | pen | welfare | speed | success |
|---|---|---|---|---|---|---|---|
| 1 | baseline:flank | flank expert (Strömbom) (registry) | **66.1** | 97% | 0.347 | 0.374 | 93% |
| 2 | baseline:greedy | greedy driver (registry) | **60.9** | 92% | 0.323 | 0.234 | 73% |
| 3 | baseline:random | random walk (registry) | **56.1** | 59% | 0.735 | 0.048 | 20% |

### Paddock (baseline) — `t1_paddock`
_The canonical config the baselines and RESULTS.md are measured on. · 50 held-out seeds · unlock next at 55_

| # | author | entry | score | pen | welfare | speed | success |
|---|---|---|---|---|---|---|---|
| 1 | baseline:greedy | greedy driver (registry) | **65.0** | 88% | 0.544 | 0.139 | 62% |
| 2 | baseline:flank | flank expert (Strömbom) (registry) | **63.5** | 87% | 0.480 | 0.208 | 74% |
| 3 | baseline:random | random walk (registry) | **56.2** | 52% | 0.860 | 0.024 | 16% |

### Open Range — `t2_range`
_Bigger field, bigger flock — gathering distance grows. · 50 held-out seeds · unlock next at 55_

| # | author | entry | score | pen | welfare | speed | success |
|---|---|---|---|---|---|---|---|
| 1 | baseline:flank | flank expert (Strömbom) (registry) | **45.6** | 45% | 0.655 | 0.009 | 4% |
| 2 | baseline:random | random walk (registry) | **40.2** | 25% | 0.790 | 0.004 | 2% |
| 3 | baseline:greedy | greedy driver (registry) | **38.1** | 22% | 0.779 | 0.000 | 0% |

### Skittish Flock — `t3_skittish`
_Jumpy sheep: wide flight zone, fast flee, fast-rising arousal. · 50 held-out seeds · unlock next at 50_

| # | author | entry | score | pen | welfare | speed | success |
|---|---|---|---|---|---|---|---|
| 1 | baseline:greedy | greedy driver (registry) | **62.6** | 96% | 0.344 | 0.168 | 76% |
| 2 | baseline:random | random walk (registry) | **61.0** | 60% | 0.864 | 0.067 | 22% |
| 3 | baseline:flank | flank expert (Strömbom) (registry) | **60.4** | 87% | 0.421 | 0.137 | 68% |

### Big Muster — `t4_big_muster`
_Two dozen sheep on a large field — the endurance tier. · 50 held-out seeds · unlock next at 50_

| # | author | entry | score | pen | welfare | speed | success |
|---|---|---|---|---|---|---|---|
| 1 | baseline:flank | flank expert (Strömbom) (registry) | **49.6** | 47% | 0.741 | 0.008 | 4% |
| 2 | baseline:greedy | greedy driver (registry) | **38.4** | 18% | 0.843 | 0.000 | 0% |
| 3 | baseline:random | random walk (registry) | **34.7** | 10% | 0.854 | 0.000 | 0% |

### Wolves (LOCKED) — `t5_predators` 🔒
_Locked: Needs predator entities + screening reward in env.py (roadmap P3)._

### Obstacles (LOCKED) — `t6_obstacles` 🔒
_Locked: Needs static obstacles + collision in env.py. Stub only._
