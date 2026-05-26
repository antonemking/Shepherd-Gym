# shepherd-gym — design-kit prompt

Paste the block below into your design tool (Claude's design/artifact mode, Figma
Make, or any UI/image generator). Trim sections to taste. Bracketed `[…]` are
knobs to set. The palette anchors come from the live renderer (`shepherd_gym/render.py`)
so the brand stays consistent with the actual footage.

---

**Design a brand + content design kit for "Shepherd-Gym."**

**What it is:** a reinforcement-learning research environment where an AI dog learns
to herd a flock of sheep into a fold *while keeping the flock calm*. It's a sibling
to a precision-livestock project ("SamSeesSheep") that measures sheep stress from
ear angle in real barn video — so this sim has a per-sheep "arousal/stress" signal,
and the research question is **"can RL discover low-stress herding?"** Output is a
benchmark + YouTuber-style learning visuals (agents ghost-trailing, failing, then
mastering the task) for a LinkedIn write-up series.

**Brand personality:** rigorous yet cozy — a Highland naturalist's field notebook
meets a modern ML lab. Calm, observational, quietly playful. Credible research, not
clip-art cute. Lean: [60% warm-naturalist / 40% clinical-lab — adjust].

**Audience & surfaces:** LinkedIn post cards & carousels, benchmark figures, animated
comparison loops (gif/webp), a GitHub README hero, slide/figure templates, and an
optional simple project web page. Must sit beside SamSeesSheep as one visual family.

**Palette anchors (from the live top-down sim renderer — keep brand legible when
overlaid on the green field):**
- Field/grass green `#3C6E3C` · fold brown `#967850` / ring `#5A462D`
- Dog "ink" `#1E1E32` (primary dark) · ghost-trail amber `#FFDC78` (signature accent)
- **Stress gradient** calm wool `#ECECE4` → stressed `#E04634` (the welfare signal — make it a first-class, colorblind-considered gradient)
- Semantic: success green `#78E68C` · failure red `#EB786E`
Derive a cohesive palette (primary, secondary, 1–2 accents, neutrals, semantic) from these.

**Typography:** pair a warm humanist sans for headings/body with a clean monospace for
metrics, code, and data labels (it's a research tool). Define a type scale.

**Deliverables (the kit):**
1. **Logo / wordmark** — a dog + sheep + a subtle "learning/RL" cue (e.g., a ghost
   trail or path). Must read at favicon size and in one color.
2. **Color tokens** — hex + names + usage, including the stress gradient and
   success/fail semantics.
3. **Type system** — families, scale, weights, sample headings.
4. **Icon style** — thin, geometric, friendly; set of ~10 (dog, sheep, fold, wolf,
   reward, stress/ear, play, compare, seed, chart).
5. **Data-viz system** — the most important piece: styles for reward & arousal
   training curves, success-rate bars, the speed-vs-stress scatter/Pareto plot, and
   "model-vs-model" comparison overlays (echo SamSeesSheep's ghosted v0.2-vs-v0.4
   keypoint look and σ-curves). Define line/marker/grid/annotation styling.
6. **Components** — metric cards, benchmark table, tags/chips (policy names, seeds),
   buttons, and a reusable "episode clip" frame chrome (top label bar + corner
   watermark) that wraps the rendered gifs/mp4s.
7. **Social / figure templates** — LinkedIn 1200×627 + square 1080×1080 + YouTube
   thumbnail 1280×720, each with slots for a looping clip, a headline, and a big
   metric callout (e.g., "80% penned • stress ↓ 31%").
8. **Diagram style** — for the environment loop (observation → policy → action →
   reward) and the herding/fold scene schematic.
9. **README hero / banner** for GitHub.

**Constraints:** all brand elements must stay legible over the green top-down field;
WCAG-AA contrast for text; the stress gradient should remain distinguishable for
deuteranopia (don't rely on red/green alone — vary lightness too).

**Output format:** a design-tokens sheet (hex, type scale, spacing), a component
sheet, and **3 finished example compositions** — (a) a LinkedIn post card announcing
the "low-stress herding" result, (b) a benchmark figure with a reward+arousal curve,
(c) the GitHub README hero.

**Avoid:** cartoon clip-art animals, barnyard kitsch, gradients-for-the-sake-of-it,
generic "AI" tropes (circuit brains, glowing blue). Keep it field-notebook credible.

---
