"""Leaderboard store: per-tier JSON + a regenerated markdown board, committed to the repo
(same spirit as results/RESULTS.md). No torch.

Layout:
    leaderboard/<tier_key>.json   — list of entry records (best-per-author kept)
    leaderboard/README.md         — the rendered board across all available tiers

Ladder unlock is *derived* from the store (not stored separately) so it's reproducible:
tier i+1 is unlocked for an author once they have an entry on tier i scoring >= its
unlock_score.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

from .tiers import AVAILABLE, LADDER, get_tier, next_tier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB_DIR = os.path.join(ROOT, "leaderboard")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _path(tier_key: str) -> str:
    return os.path.join(LB_DIR, f"{tier_key}.json")


def load_tier(tier_key: str) -> list[dict]:
    p = _path(tier_key)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)


def make_entry(tier_key: str, author: str, title: str, kind: str, ref: str,
               result: dict) -> dict:
    return {
        "author": author,
        "title": title,
        "kind": kind,
        "ref": ref,
        "tier": tier_key,
        "score": round(result["score"], 2),
        "P": round(result["P"], 4),
        "W": round(result["W"], 4),
        "S": round(result["S"], 4),
        "success": round(result["success"], 4),
        "arousal": round(result["arousal"], 4),
        "n_seeds": result["n"],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": _git_sha(),
    }


def record(entry: dict) -> list[dict]:
    """Record `entry`, keeping only each author's best-scoring run for the tier."""
    os.makedirs(LB_DIR, exist_ok=True)
    best = {e["author"]: e for e in load_tier(entry["tier"])}
    cur = best.get(entry["author"])
    # >= so a resubmission that ties the stored (2-dp) best still refreshes its
    # timestamp and code_version — the latest run for a given best score wins.
    if cur is None or entry["score"] >= cur["score"]:
        best[entry["author"]] = entry
    ranked = rank(list(best.values()))
    with open(_path(entry["tier"]), "w") as f:
        json.dump(ranked, f, indent=2)
    return ranked


def rank(entries: list[dict]) -> list[dict]:
    return sorted(entries, key=lambda e: e["score"], reverse=True)


def unlocked_tiers(author: str) -> list[str]:
    """Tier keys this author can play: always the first tier, then each next tier whose
    predecessor they've cleared (entry score >= predecessor unlock_score)."""
    unlocked = [AVAILABLE[0].key] if AVAILABLE else []
    for t in AVAILABLE:
        nxt = next_tier(t.key)
        if nxt is None or not nxt.available:
            continue
        cleared = any(e["author"] == author and e["score"] >= t.unlock_score
                      for e in load_tier(t.key))
        if cleared and nxt.key not in unlocked:
            unlocked.append(nxt.key)
    return unlocked


def tier_table(tier_key: str) -> str:
    t = get_tier(tier_key)
    entries = rank(load_tier(tier_key))
    rows = [f"### {t.name} — `{t.key}`",
            f"_{t.note} · {t.n_seeds} held-out seeds · unlock next at {t.unlock_score:.0f}_",
            "",
            "| # | author | entry | score | pen | welfare | speed | success |",
            "|---|---|---|---|---|---|---|---|"]
    if not entries:
        rows.append("| — | _(no entries yet)_ | | | | | | |")
    for i, e in enumerate(entries, 1):
        rows.append(
            f"| {i} | {e['author']} | {e['title']} ({e['kind']}) | **{e['score']:.1f}** | "
            f"{e['P']*100:.0f}% | {e['W']:.3f} | {e['S']:.3f} | {e['success']*100:.0f}% |"
        )
    return "\n".join(rows)


def render_board() -> str:
    md = ["# 🐕🐑 Shepherd-Gym Leaderboard", "",
          "Composite score (0–100) = `50·penning + 35·welfare + 15·speed`. "
          "Speed credit is awarded only on a full pen, so stalling to dodge stress can't win. "
          "Welfare = `1 − mean flock arousal` (gentler handling scores higher).", ""]
    for t in LADDER:
        if not t.available:
            md += [f"### {t.name} — `{t.key}` 🔒", f"_Locked: {t.note}_", ""]
            continue
        md += [tier_table(t.key), ""]
    return "\n".join(md)


def write_board() -> str:
    os.makedirs(LB_DIR, exist_ok=True)
    path = os.path.join(LB_DIR, "README.md")
    md = render_board()
    with open(path, "w") as f:
        f.write(md)
    return path
