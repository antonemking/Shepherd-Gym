# Data roadmap — from heuristic → empirical

The gym's behaviour models start **theory-grounded** (parameters set from the
flight-zone / handling literature). This is the ladder that turns them **empirical**
as SamSeesSheep footage accrues. Each rung replaces a hand-set assumption with a
fitted one and unlocks a stronger claim. (Reminder on the word: more data makes the
sim *more* empirical, not less.)

Legend: `[x]` done · `[~]` partially possible with current footage · `[ ]` needs new capture.

## Rung 0 — theory-grounded `[x]`
Flight-zone arousal model + Strömbom-rooted locomotion, parameters from published
flight-zone/FID, gregariousness, and handling work. **Claim:** "a methods demo on a
plausible, literature-grounded stress model." (See `welfare-model.md`.)

## Rung 1 — ear-angle observable `[~]`
- **Data:** existing flock clips → SamSeesSheep ear-angle time series.
- **Fits:** measurement noise `ear_noise_deg` (you already have σ≈4° on v0.4), and the
  calm↔aroused ear-angle **range** (the `ear_neutral_deg` / `ear_aroused_deg` endpoints).
- **Needs:** a few calm clips and a few clearly-aroused clips to anchor the endpoints.
- **Unlocks:** "the observable the agent sees is calibrated to my pipeline's real output."

## Rung 2 — flock locomotion `[~]`
- **Data:** multi-sheep position tracks over time (segmentation → tracking).
- **Fits:** flee radius, boid weights, speed caps — how fast/far real sheep move and bunch.
- **Unlocks:** "the simulated flock *moves* like my flock." (Strömbom-style, from video instead of GPS.)

## Rung 3 — stress-response dynamics `[ ]`  ← the digital-twin moment
- **Data:** **dog/handler AND flock in the same clip, time-synced.** Paired series of
  (dog position rel. flock, dog closing speed, inter-sheep spacing) ↔ ear-angle arousal.
- **Fits:** `flight_zone`, `intrusion_exp`, `w_closing`, `w_isolation`, rise/decay.
- **Catch:** current clips look flock-only — this needs a deliberate capture (below).
- **Unlocks:** "the agent is trained against *my flock's* measured stress response."

### Capture protocol for Rung 3
- Both the working dog/handler and the flock visible the whole clip; fixed/known camera.
- A real herding session (approach, gather, drive, pen) — natural range of pressure.
- 1080p, 15–60 s, 2 fps is enough; note approximate scale (a known length in frame) so
  pixel distances convert to metres.
- A few sessions across calm and pushy handling to span the stimulus range.

## Rung 4 — proxy validation `[ ]`  (research-grade, future)
- **Data:** ground-truth stress alongside ear angle — physiological (cortisol, heart-rate)
  or documented handling-vs-grazing events.
- **Confirms:** that ear-angle arousal actually tracks welfare (your README's open gap).
- **Unlocks:** stating results in **welfare units**, not proxy units.

---

**The arc of the claim, rung by rung:** plausible model → calibrated observable →
realistic flock → *my flock's* stress response → validated welfare units.
