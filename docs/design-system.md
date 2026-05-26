# shepherd-gym — design system

> Source of truth: Figma file [`Shepherd-Gym — Design Kit`](https://www.figma.com/design/oQ77CQwjr7J3tTVX1foR9B)
> (page "Design Kit"). This doc mirrors it as code-ready tokens and component specs,
> and maps them to the live sim renderer (`shepherd_gym/render.py`) so design and
> simulation stay in sync.

**Brand personality:** rigorous yet cozy — a Highland naturalist's field notebook
meets an ML lab. Calm, observational, quietly playful. Credible research, not kitsch.
Sibling to the SamSeesSheep welfare-CV series.

---

## 1. Color tokens

| Token | Hex | Role | `render.py` |
|---|---|---|---|
| `grass` | `#3C6E3C` | Field / primary surface | `GRASS` |
| `fold-fill` | `#967850` | The fold (pen) | `PEN_FILL` |
| `fold-ring` | `#5A462D` | Fold outline | `PEN_RING` |
| `dog-ink` | `#1E1E32` | Primary dark · the dog · text on light | `DOG` |
| `ghost-amber` | `#FFDC78` | Signature accent · trails · highlights | trail `#FFDC78` |
| `calm-wool` | `#ECECE4` | Calm sheep · light neutral | `CALM` |
| `stress` | `#E04634` | Stressed sheep · danger · welfare-high | `STRESS` |
| `success` | `#78E68C` | Penned · success | swarm-success |
| `fail` | `#EB786E` | Failure | swarm-fail |
| `heather` | `#6E5A8C` | Secondary accent (Highland) · muted labels | — |
| `mist` | `#C9CFD2` | Borders · subtle text · gridlines | — |
| `paper` | `#F4F1E9` | Background · light surface | — |

**Stress gradient** (the welfare signal — sheep colour = arousal ∈ [0,1]). Don't
rely on hue alone; lightness also drops, so it survives deuteranopia:

| stop | hex |
|---|---|
| 0.0 calm | `#ECECE4` |
| 0.5 | `#E8B86A` |
| 1.0 stressed | `#E04634` |

```json
{
  "color": {
    "grass": "#3C6E3C", "fold-fill": "#967850", "fold-ring": "#5A462D",
    "dog-ink": "#1E1E32", "ghost-amber": "#FFDC78", "calm-wool": "#ECECE4",
    "stress": "#E04634", "success": "#78E68C", "fail": "#EB786E",
    "heather": "#6E5A8C", "mist": "#C9CFD2", "paper": "#F4F1E9"
  },
  "gradient": { "arousal": ["#ECECE4", "#E8B86A", "#E04634"] }
}
```

---

## 2. Typography

- **Headings / body:** Inter
- **Metrics / data / code:** Roboto Mono (fallback: Inter Bold)

| Style | Font | Size | Weight | Use |
|---|---|---|---|---|
| Display | Inter | 48 | Bold | Section heroes |
| Hero | Inter | 58–64 | Bold | Social-card headlines |
| H1 | Inter | 32 | Bold | Page titles |
| H2 | Inter | 24 | Semi Bold | Subsections |
| Body | Inter | 16 | Regular | Paragraphs |
| Caption | Inter | 13 | Regular | Footnotes, axis labels |
| Eyebrow | Inter | 18 | Bold | Card kickers (uppercase, amber) |
| Metric | Roboto Mono | 28–52 | Medium | Numbers, stats, data |

---

## 3. Spacing, radius, elevation

- **Spacing scale (px):** 4 · 8 · 12 · 16 · 20 · 24 · 32 · 64
- **Radius (px):** `sm` 8 · `md` 12 · `lg` 16 · `pill` 17 (chips) · `xl` 24 (social cards)
- **Iconography:** thin, geometric, 2px stroke, rounded caps; built from the same
  dot/ring/trail primitives as the logo.

---

## 4. Components

**Metric card** — 300×160, radius `lg`, fill `paper`/white, 8px left accent bar in a
semantic color. Metric (Mono 52, semantic color) + label (Inter Semi Bold 16, `heather`).

**Policy chip** — pill (radius 17), white fill, 12px status dot + Inter Semi Bold 14.
Dot colors: random `fail` · greedy `ghost-amber` · flank `success` · ppo `heather`.

**Benchmark table** — header row `dog-ink` fill / `paper` text (Semi Bold 15); body rows
alternate white / `paper` (radius 8); cells Mono 15; the `arousal ↓` cell uses `stress`
for the welfare-relevant row, else `heather`.

**Episode-clip frame** — the chrome that wraps a rendered gif/mp4. Grass fill; fold =
`fold-fill` disc + `fold-ring` 5px stroke; sheep = 16px dots colored via the arousal
gradient; dog = 22px `dog-ink` dot + 2px white stroke; ghost trail = fading `ghost-amber`
dots; top label bar `dog-ink` @ 85% with Mono 14 (`<policy> · seed N · t=… · penned x/N`);
bottom-right watermark `shepherd·gym` (Inter Bold 13, `ghost-amber`).

---

## 5. Data-viz rules

- **Reward** series → `ghost-amber`; **arousal** series → `stress`; **baseline** ref
  line → `success`.
- Points: 6px dots (or 2px line). Gridlines: `mist` 1px. Axes: `dog-ink` 2px.
- Legend top-right, 12px swatch dots + Caption labels.
- Title: H2. Axis labels: Caption in `heather`.
- Mirror the SamSeesSheep grammar: same scenario/seed, **multiple models ghosted**
  on one frame (e.g. checkpoint step-1k vs 10k vs 100k), plus a metric curve beneath.

---

## 6. Templates

- **LinkedIn card** 1200×627 — `dog-ink` bg, radius `xl`. Eyebrow (amber) → Hero
  (Inter Bold 58, `paper`) → subhead (Regular 22, `mist`) → 2 stat blocks (230×120,
  radius `md`, fill `#262640`) → footer wordmark. Field viz panel on the right (420×440).
- **Square** 1080×1080 and **YouTube thumb** 1280×720 — same system, re-flowed.
- **Benchmark figure** — white card, radius `lg`; see §5.

---

## 7. Voice & tone

Plain, precise, a little warm. Lead with the welfare angle ("herd *gently*"), back it
with numbers, stay honest about scope (sim ≠ real flock). Lowercase `shepherd·gym`
wordmark; sentence case headlines.

---

## 8. Keeping this in sync with Figma

This MD was authored from the kit's exact values. To re-export after Figma edits:

1. **Figma Dev Mode** → select a frame → *Copy as code* / inspect the **Local variables**
   and **styles** panels for hexes and type values.
2. **Via the Figma MCP** (what generated the kit): `get_variable_defs` returns variable
   tokens, `get_design_context` returns per-node specs. NB: the kit currently uses **paint
   & text *styles*** (not variables) — converting them to **Figma variables** would make
   `get_variable_defs` export a clean token JSON automatically. Say the word and I'll
   convert styles → variables so future exports are one call.
3. Treat **`render.py` color constants** as the runtime mirror of §1 — change them together.
