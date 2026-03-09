"""
ablation_config.py
==================
Edit the settings below, then run:

    python ablation_config.py

That's it. No command-line arguments needed.

Author : Amos Okpe  (MSc Computer Science, University of Hull)
"""

import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
#  SETTINGS — edit these
# ─────────────────────────────────────────────────────────────────────────────

# Path to the video you want to evaluate on.
# Can be absolute or relative to the folder you run this script from.
VIDEO_PATH = "data/input.mp4"

# How many frames to use from the video.
# More frames = more reliable results but slower.
# 300–500 is a good range for a dissertation result.
MAX_FRAMES = 400

# Minimum track length (frames) to be included in the evaluation.
# Tracks shorter than this are skipped — they're too short to split for re-entry.
MIN_TRACK_FRAMES = 25

# Re-ID matching threshold (cosine distance).
# Lower = stricter matching. Higher = more lenient.
# 0.28–0.35 works well for most videos.
REID_THRESHOLD = 0.30

# Random seed — keep this fixed so results are reproducible across runs.
SEED = 42

# Where to save outputs.  Both are optional — set to None to skip saving.
OUT_CSV    = "results/ablation_results.csv"   # None to skip
OUT_REPORT = "results/ablation_report.txt"    # None to skip

# ─────────────────────────────────────────────────────────────────────────────
#  ADVANCED — you probably don't need to change these
# ─────────────────────────────────────────────────────────────────────────────

# HSV histogram bins per strip (h_bins × s_bins features per strip).
# Increasing adds detail but slows comparison. 32×32 is the sweet spot.
H_BINS = 32
S_BINS = 32

# EMA update rates.
# alpha_near: update rate for large/close detections (lower = faster update)
# alpha_far:  update rate for small/distant detections (higher = more conservative)
EMA_ALPHA_NEAR = 0.75
EMA_ALPHA_FAR  = 0.93

# Maximum persons tracked simultaneously in the gallery.
MAX_GALLERY_SIZE = 64

# ─────────────────────────────────────────────────────────────────────────────
#  RUN — nothing to edit below this line
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Make sure we can import from the same directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from pathlib import Path
    from person_reid import PersonReIdentifier
    from reid_evaluator import extract_tracks_from_video, evaluate_config
    from run_ablation import CONFIGS, print_table, save_csv, save_report
    import time

    # ── Validate video path ───────────────────────────────────────────────
    if not Path(VIDEO_PATH).exists():
        print(f"\n  ERROR: Video not found at '{VIDEO_PATH}'")
        print(f"  Edit VIDEO_PATH in this file to point to your video.\n")
        sys.exit(1)

    # ── Create output directory if needed ────────────────────────────────
    for out_path in [OUT_CSV, OUT_REPORT]:
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Video        : {VIDEO_PATH}")
    print(f"  Max frames   : {MAX_FRAMES}")
    print(f"  Min track    : {MIN_TRACK_FRAMES} frames")
    print(f"  Threshold    : {REID_THRESHOLD}")
    print(f"  Seed         : {SEED}")

    t0 = time.time()

    # ── Extract tracks ────────────────────────────────────────────────────
    print(f"\n  Extracting tracks …")
    try:
        frames, tracks = extract_tracks_from_video(
            VIDEO_PATH,
            min_track_frames=MIN_TRACK_FRAMES,
            max_frames=MAX_FRAMES,
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    n_frames = len(frames)
    n_tracks = len(tracks)
    print(f"  → {n_frames} frames loaded, {n_tracks} tracks found")

    if n_tracks == 0:
        print("  ERROR: No tracks long enough to evaluate.")
        print(f"  Try lowering MIN_TRACK_FRAMES (currently {MIN_TRACK_FRAMES})")
        print(f"  or increasing MAX_FRAMES (currently {MAX_FRAMES}).")
        sys.exit(1)

    # ── Run each ablation config ──────────────────────────────────────────
    print()
    all_results = []
    for cfg in CONFIGS:
        thresh = cfg.get("threshold_override") or REID_THRESHOLD
        reid = PersonReIdentifier(
            reid_threshold   = thresh,
            h_bins           = H_BINS,
            s_bins           = S_BINS,
            ema_alpha_near   = EMA_ALPHA_NEAR,
            ema_alpha_far    = EMA_ALPHA_FAR,
            max_gallery_size = MAX_GALLERY_SIZE,
            use_bg_crop      = cfg["use_bg_crop"],
            use_strips       = cfg["use_strips"],
            use_occlusion    = cfg["use_occlusion"],
            use_quality_ema  = cfg["use_quality_ema"],
        )
        print("  Evaluating [{:20s}] (thresh={:.2f}) …".format(cfg["name"], thresh),
              end=" ", flush=True)
        t_cfg = time.time()
        result = evaluate_config(
            frames      = frames,
            tracks      = tracks,
            reid        = reid,
            config_name = cfg["name"],
            seed        = SEED,
        )
        print("done ({:.1f}s)  F1={:.4f}  ID-reduction={:.1f}%".format(
            time.time() - t_cfg, result.f1, result.id_reduction_pct))
        all_results.append(result)

    # ── Print tables ──────────────────────────────────────────────────────
    print_table(all_results)

    # ── Save outputs ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    if OUT_CSV:
        save_csv(all_results, OUT_CSV)
    if OUT_REPORT:
        save_report(
            all_results, CONFIGS, VIDEO_PATH,
            n_frames, n_tracks, elapsed, OUT_REPORT,
        )

    print(f"\n  Total time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
