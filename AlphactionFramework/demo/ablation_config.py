"""
ablation_config.py
==================
Edit the settings below, then run:

    python ablation_config.py

That's it. No command-line arguments needed.

Ablation study:
  Config 0: HSV BASELINE   -- colour histogram, no neural network
  Config 1: +BG_CROP (HSV) -- HSV + edge suppression
  Config 2: DEEP (ResNet50) -- deep features, no bg crop
  Config 3: +BG_CROP (Deep) -- deep features + edge suppression
  Config 4: +QUAL_EMA (FULL)-- full system: Deep + BG crop + quality EMA

Author : Amos Okpe  (MSc Computer Science, University of Hull)
"""

import sys
import os

# =============================================================================
#  SETTINGS -- edit these
# =============================================================================

# Path to the video you want to evaluate on.
VIDEO_PATH = "./test_video.mp4"

# How many frames to use from the video (300-500 recommended).
MAX_FRAMES = 400

# Minimum track length (frames) to be included in the evaluation.
MIN_TRACK_FRAMES = 25

# Random seed -- keep fixed so results are reproducible.
SEED = 42

# Where to save outputs (set to None to skip).
OUT_CSV    = "results/ablation_results.csv"
OUT_REPORT = "results/ablation_report.txt"

# =============================================================================
#  RUN -- nothing to edit below this line
# =============================================================================

def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import time
    import torch
    from pathlib import Path
    from person_reid import PersonReIdentifier
    from reid_evaluator import extract_tracks_from_video, evaluate_config
    from run_ablation import CONFIGS, print_table, save_csv, save_report

    if not Path(VIDEO_PATH).exists():
        print("\n  ERROR: Video not found at '{}'".format(VIDEO_PATH))
        print("  Edit VIDEO_PATH in this file.\n")
        sys.exit(1)

    for out_path in [OUT_CSV, OUT_REPORT]:
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n  Video        : {}".format(VIDEO_PATH))
    print("  Max frames   : {}".format(MAX_FRAMES))
    print("  Min track    : {} frames".format(MIN_TRACK_FRAMES))
    print("  Device       : {}".format(device))
    print("  Seed         : {}".format(SEED))

    t0 = time.time()

    print("\n  Extracting tracks ...")
    try:
        frames, tracks = extract_tracks_from_video(
            VIDEO_PATH,
            min_track_frames=MIN_TRACK_FRAMES,
            max_frames=MAX_FRAMES,
        )
    except Exception as e:
        print("  ERROR: {}".format(e))
        sys.exit(1)

    n_frames = len(frames)
    n_tracks = len(tracks)
    print("  -> {} frames loaded, {} tracks found".format(n_frames, n_tracks))

    if n_tracks == 0:
        print("  ERROR: No tracks long enough. Lower MIN_TRACK_FRAMES or increase MAX_FRAMES.")
        sys.exit(1)

    print()
    all_results = []

    for cfg in CONFIGS:
        thresh   = cfg.get("threshold_override", 0.40)
        relax    = cfg.get("relax_override",     thresh + 0.20)
        use_deep = cfg.get("use_deep_features",  False)

        reid = PersonReIdentifier(
            device               = device,
            reid_threshold       = thresh,
            reid_threshold_relax = relax,
            use_deep_features    = use_deep,
            use_bg_crop          = cfg["use_bg_crop"],
            use_quality_ema      = cfg["use_quality_ema"],
            max_gallery_size     = 64,
        )

        print("  [{:22s}] (deep={}, thresh={:.2f}) ...".format(
            cfg["name"], use_deep, thresh), end=" ", flush=True)

        t_cfg  = time.time()
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

    print_table(all_results)

    elapsed = time.time() - t0
    if OUT_CSV:
        save_csv(all_results, OUT_CSV)
    if OUT_REPORT:
        save_report(
            all_results, CONFIGS, VIDEO_PATH,
            n_frames, n_tracks, elapsed, OUT_REPORT,
        )

    print("\n  Total time: {:.1f}s\n".format(elapsed))


if __name__ == "__main__":
    main()
