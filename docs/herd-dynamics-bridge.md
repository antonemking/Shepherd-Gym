# Build ticket — `herd_dynamics`: Ezinu → shepherd-gym (Rung 2)

The seam that turns three repos into one system: an **Ezinu scenario** on the gable
cam captures dog+flock trajectories; a **gym calibrator** fits the sim's flock to that
real data ("my simulation moves like my actual flock"). This is data-roadmap **Rung 2**.

```
gable RTSP ─▶ Ezinu: detect(sheep,dog) ─▶ track ─▶ ground-plane(m) ─▶ herding_episode events
                                                                          │ (trajectories)
                                                                          ▼
                                          shepherd-gym: calibrate_dynamics.py ─▶ fit ShepherdConfig
```

## 0. The data contract (the only thing both sides must agree on)

Each detected herding episode → one JSONL record (`out/episodes/*.jsonl`), units = metres
on the pasture ground plane, with the sim's axes:

```json
{ "episode_id": "...", "site_id": "...", "camera_id": "...", "fps": 4.0, "units": "m",
  "pasture_extent_m": [W, H],
  "frames": [ { "t": 0.0, "dog": [x, y], "sheep": [[track_id, x, y], ...] }, ... ] }
```
`dog` may be `null` on frames where it's not detected. `track_id` is stable within an episode.

---

## 1. Ezinu side — the `herd_dynamics` scenario

Copy `scenarios/_template/` → `scenarios/herd_dynamics/`. Pure scenario, same contract
as `trough_visits`. It consumes detections already tagged with `attributes["zone_id"]`
**and** (new) `attributes["track_id"]`, projects each to the ground plane, segments
herding episodes, and emits one `herding_episode` event per episode (trajectory in payload).

**`config.py` — `HerdDynamicsConfig(ScenarioConfig)`**
- `pasture_zone_id: str` — zone the flock lives in.
- `flock_label: str = "sheep"`, `dog_label: str = "dog"` — open-vocab prompt labels.
- `homography: list[list[float]]` — 3×3 pixel→metre matrix (from the calibrator, §3).
- `pasture_extent_m: tuple[float, float]` — width/height in metres (sets sim `world`).
- `sample_fps: float = 4.0` — decimate to this rate before logging.
- `episode_dog_speed_mps: float = 0.8` — dog faster than this near the flock starts an episode.
- `episode_near_flock_m: float = 12.0` — and within this of the flock centroid.
- `episode_min_seconds: float = 4.0`, `episode_gap_seconds: float = 3.0` — close/duration gates.

**`state.py` — `HerdDynamicsState(BaseModel)`** (JSON round-trips)
- `episode_active: bool`, `episode_frames: list[dict]` (the buffer of `{t,dog,sheep}`),
  `episode_start_seq: int | None`, `idle_streak: int`, `prev_dog_xy: list[float] | None`,
  `last_frame_ts`, `decimate_acc: float`.

**`pipeline.py` — `HerdDynamicsScenario(Scenario[...])`** — `step(frame, detections)`:
1. Decimate to `sample_fps` (skip frame if under cadence).
2. Filter to `flock_label`/`dog_label` with `attributes["zone_id"] == pasture_zone_id`.
3. Project each detection's **ground-contact point** (bbox bottom-centre) through `homography` → (x,y) m.
4. Dog speed = ‖dog_xy − prev_dog_xy‖ / dt. Flock centroid = mean sheep_xy.
5. **Episode trigger:** `dog_speed > episode_dog_speed_mps AND dist(dog, centroid) < episode_near_flock_m`.
   - On start: open episode, reset buffer. While active: append `{t, dog, sheep:[[tid,x,y]…]}`.
   - On `episode_gap_seconds` of non-trigger: close. If duration ≥ `episode_min_seconds`,
     emit `herding_episode` event; else discard.
- `event_kinds = ("herding_episode",)`.
- `Event.payload` = the contract record (§0); `Evidence(frame_seqs=[start…end])`.
- `outcome_catalog`: one `OutcomeDescriptor` (category `"research_observation"`, indicator
  `"flock_response"`, neutral rationale — operational observation, *not* a welfare claim,
  per Ezinu's scope rule).
- `default_rules`: one `info`-severity rule, long cooldown (capture, not alarm).
- Register in `scenarios/registry.py` (`REGISTRY["herd_dynamics"]` + `_build_herd_dynamics_config`).

### 1a. Tracking (runtime addition — detector layer, keeps the scenario pure)
There's no MOT in Ezinu today. Add a tracking-capable backend (or extend
`ultralytics_open_vocab`): use ultralytics `model.track(..., persist=True)` (ByteTrack) and
write the returned id to `detection.attributes["track_id"]` — exactly how `zone_id` is
injected. The scenario then reads `track_id` like any other attribute; purity preserved.

## 2. Ezinu side — homography calibration (extend `barn_calibrator`)
Add a `homography` subcommand alongside the existing `snapshot`/polygon tooling:
capture a reference frame → operator supplies ≥4 pixel points and their real ground
coords (pen corners, fence posts, a tape laid in frame) → `cv2.findHomography` → write the
3×3 + `pasture_extent_m` into the site/scenario config. This is the gable-cam setup step.

---

## 3. Gym side — `scripts/calibrate_dynamics.py` (Rung 2)

Sibling to the ear `calibrate.py`. Fits the sim flock to the captured episodes by
**replaying the real dog through the sim** and matching the real flock's response.

- **Ingest:** read `out/episodes/*.jsonl`; set `ShepherdConfig.world = pasture_extent_m`.
- **Per episode:** seed a `ShepherdEnv` with the episode's initial sheep positions; each
  step, *override* the sim dog with the real dog's logged position; record the sim flock
  trajectory.
- **Objective:** minimise, over the episode, `Σ‖sim_centroid − real_centroid‖² +
  λ·(sim_spread − real_spread)²` (centroid path + bunching; robust, low-dimensional).
- **Fit (reduced, identifiable set):** `flight_zone` (flee-onset distance), `sheep_max_speed`
  (observed max flock speed), `social_dist` (median real nearest-neighbour distance),
  `w_flee`, `w_coh`. Derivative-free (scipy Nelder-Mead, or numpy coordinate descent if
  scipy absent) — the vectorised sim makes each evaluation cheap.
- **Guard** (like the ear calibrator): require ≥ ~5 episodes / ≥ ~500 frames and a sane fit
  (params within physical bounds, fit error below a threshold); else refuse to write and
  report what's missing.
- **Output:** patch the flock params in `ShepherdConfig`; write `out/dynamics_calibration.json`
  (fitted values + residuals) and a **sim-vs-real overlay** (the calibrator's headline figure:
  real flock path vs sim flock path under the same dog) — directly a LinkedIn artifact.
- **Roadmap:** flip Rung 2 to done; note the sim is now calibrated to *this* flock.

---

## Effort / sequencing
- **Reusable from Ezinu (~70%):** RTSP, open-vocab detect (sheep+dog by prompt), zones,
  scenario scaffold, evidence/persistence, `barn_calibrator` capture.
- **New builds:** (a) tracking backend (`.track(persist=True)` → `track_id`), (b) homography
  subcommand, (c) the `herd_dynamics` scenario, (d) the gym `calibrate_dynamics.py`.
- **Blocked on:** the gable cam being up + a few real herding passes captured.
- **Gotcha:** the gable view is great for geometry/tracking but too far for ear angle —
  welfare (Rung 1/3 stress signal) stays the close-cam/trough job. Two cameras, two halves.
```
