"""Rung 1 calibration — fit the sim's ear-angle observable to real SamSeesSheep data.

Reads the reviewed keypoint labels (data/labels/*/review.json), computes ear angle
exactly as SamSeesSheep does — `normalize(ear_base→tip − nose→ear-mid)` in [-90,90],
with clinical thresholds UP>30° / DOWN<-10° — and reports the calm central angle and
the frame-to-frame measurement jitter. Recommends ShepherdConfig ear-angle values:

  ear_neutral_deg  ← measured calm central angle (these clips are calm sheep)
  ear_aroused_deg  ← SamSeesSheep's clinical DOWN threshold (no stressed frames here)
  ear_noise_deg    ← measured within-clip jitter σ

    .venv/bin/python scripts/calibrate.py            # report only
    .venv/bin/python scripts/calibrate.py --apply    # also patch shepherd_gym/env.py
"""
import sys, json, glob, math, argparse, pathlib, re
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SS = pathlib.Path("/home/toneking/dev/lorewood-advisors/sheep-seg")
LABELS = SS / "data" / "labels"
EAR_DOWN_THRESHOLD_DEG = -10.0   # from SamSeesSheep backend/config.py (clinical, SPFES)


def _ang(a, b):
    return math.degrees(math.atan2(-(b["y"] - a["y"]), b["x"] - a["x"]))


def _norm90(a):
    while a > 180: a -= 360
    while a < -180: a += 360
    if a > 90: a -= 180
    elif a < -90: a += 180
    return a


def _vis(kp):
    return kp.get("v", 0) > 0


def frame_ear_angles(kps):
    """kps = [nose, L_base, R_base, L_tip, R_tip]. Returns list of per-ear angles."""
    if not kps or len(kps) < 5:
        return []
    nose, lb, rb, lt, rt = kps[:5]
    bases = [p for p in (lb, rb) if _vis(p)]
    if not _vis(nose) or not bases:
        return []
    ear_mid = {"x": np.mean([p["x"] for p in bases]), "y": np.mean([p["y"] for p in bases])}
    head_axis = _ang(nose, ear_mid)
    out = []
    for base, tip in ((lb, lt), (rb, rt)):
        if _vis(base) and _vis(tip):
            out.append(_norm90(_ang(base, tip) - head_axis))
    return out


def main(apply=False):
    files = sorted(glob.glob(str(LABELS / "*" / "review.json")))
    if not files:
        print(f"No review.json under {LABELS} — is SamSeesSheep present?"); return
    all_ang, per_clip_std, n_frames = [], [], 0
    for f in files:
        d = json.load(open(f))
        clip = []
        for fr in d["frames"]:
            n_frames += 1
            clip += frame_ear_angles(fr["keypoints"])
        if len(clip) >= 3:
            per_clip_std.append(float(np.std(clip)))   # within-clip jitter (calm sheep ≈ stationary posture)
        all_ang += clip
    a = np.array(all_ang)
    print(f"Sampled {len(a)} ear-angle measurements over {n_frames} frames / {len(files)} clips\n")
    print(f"  ear angle: mean={a.mean():.1f}°  median={np.median(a):.1f}°  std={a.std():.1f}°")
    print(f"  range: [{a.min():.1f}, {a.max():.1f}]   p10={np.percentile(a,10):.1f}  p90={np.percentile(a,90):.1f}")
    up = (a > 30).mean() * 100; down = (a < -10).mean() * 100
    print(f"  posture mix (SamSeesSheep thresholds): UP {up:.0f}% · NEUTRAL {100-up-down:.0f}% · DOWN {down:.0f}%")
    jitter = float(np.median(per_clip_std)) if per_clip_std else float(a.std())
    print(f"  measurement jitter σ (median within-clip): {jitter:.1f}°")

    # --- sufficiency guard: never write non-physical values into the sim ---
    enough = len(a) >= 150
    plausible = jitter <= 12.0
    cal = {"n_measurements": int(len(a)), "n_frames": n_frames, "mean_deg": float(a.mean()),
           "median_deg": float(np.median(a)), "within_clip_jitter_deg": jitter,
           "sufficient": bool(enough and plausible), "source": "SamSeesSheep review.json"}
    (ROOT / "out").mkdir(exist_ok=True)
    json.dump(cal, open(ROOT / "out" / "ear_calibration.json", "w"), indent=2)

    if not (enough and plausible):
        print("\n⚠  INSUFFICIENT / UNCONTROLLED data — NOT fitting endpoints from this.")
        if not enough:
            print(f"   only {len(a)} measurements (want ~150+).")
        if not plausible:
            print(f"   within-clip jitter {jitter:.0f}° ≫ SamSeesSheep's validated ~4° — these clips")
            print(f"   aren't motionless, so this is real head motion, not measurement noise; and a")
            print(f"   from-scratch keypoint formula doesn't reproduce their validated mask metric.")
        print("   → Credible Rung-1 fit needs the VALIDATED pipeline (v0.4 weights + ear_angle.py)")
        print("     run on curated motionless clips. Until then, the only trustworthy value is")
        print("     ear_noise_deg ≈ 4.0 from the v0.4 held-out benchmark (docs/v0.4-benchmark.md);")
        print("     endpoints stay anchored to the SPFES clinical thresholds (UP 30 / DOWN -10).")
        print("   wrote out/ear_calibration.json (sufficient=false). No changes applied.")
        return

    neutral = round(float(np.median(a))); aroused = round(EAR_DOWN_THRESHOLD_DEG); noise = round(jitter)
    print(f"\nRecommended (Rung-1 calibrated): neutral={neutral} aroused={aroused} noise={noise}")
    if apply:
        env = ROOT / "shepherd_gym" / "env.py"; src = env.read_text()
        src = re.sub(r"ear_neutral_deg: float = [-\d.]+", f"ear_neutral_deg: float = {neutral}.0", src)
        src = re.sub(r"ear_noise_deg: float = [-\d.]+", f"ear_noise_deg: float = {noise}.0", src)
        env.write_text(src); print(f"applied → patched ear_* defaults in {env}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    main(ap.parse_args().apply)
