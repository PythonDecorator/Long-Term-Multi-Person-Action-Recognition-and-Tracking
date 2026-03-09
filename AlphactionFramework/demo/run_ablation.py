#!/usr/bin/env python3
"""
run_ablation.py — Ablation Study: Re-ID Feature-by-Feature Evaluation
======================================================================

Runs five configurations of PersonReIdentifier on a video and produces
a dissertation-ready metrics table.

Each configuration adds ONE feature on top of the previous one:

  Config 0: BASELINE     — global HSV histogram, fixed EMA, no spatial info
  Config 1: +BG_CROP     — suppress background (outer 20% of box ignored)
  Config 2: +STRIPS      — spatial pyramid: 3 overlapping horizontal strips
  Config 3: +OCCLUSION   — skip strips too small (partial occlusion aware)
  Config 4: +QUAL_EMA    — quality-weighted EMA update (FULL SYSTEM)

Usage
-----
  # Basic — just run on a video:
  python run_ablation.py --video path/to/video.mp4

  # Save results to CSV and a text report:
  python run_ablation.py --video path/to/video.mp4 --out-csv results.csv --out-report report.txt

  # Use your own pre-extracted tracks (faster):
  python run_ablation.py --video path/to/video.mp4 --max-frames 300

Author : Amos Okpe  (MSc Computer Science, University of Hull)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from person_reid import PersonReIdentifier
from reid_evaluator import (
    EvalResults,
    TrackSegment,
    evaluate_config,
    extract_tracks_from_video,
)


# ---------------------------------------------------------------------------
# Configuration definitions
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name":               "BASELINE",
        "use_bg_crop":        False,
        "use_strips":         False,
        "use_occlusion":      False,
        "use_quality_ema":    False,
        "threshold_override": None,   # uses global REID_THRESHOLD
        "description":        "Global HSV histogram, fixed EMA — no improvements",
    },
    {
        "name":               "+BG_CROP",
        "use_bg_crop":        True,
        "use_strips":         False,
        "use_occlusion":      False,
        "use_quality_ema":    False,
        "threshold_override": None,
        "description":        "Adds background suppression (outer 20% of box width ignored)",
    },
    {
        "name":               "+STRIPS",
        "use_bg_crop":        True,
        "use_strips":         True,
        "use_occlusion":      False,
        "use_quality_ema":    False,
        # Spatial strips produce higher inter-strip distances than a global
        # histogram at the same threshold. 0.42 recalibrates the decision
        # boundary so precision and recall are both optimised.
        "threshold_override": 0.42,
        "description":        "Adds spatial pyramid: Head/Torso/Legs strips with overlap",
    },
    {
        "name":               "+OCCLUSION",
        "use_bg_crop":        True,
        "use_strips":         True,
        "use_occlusion":      True,
        "use_quality_ema":    False,
        "threshold_override": 0.42,
        "description":        "Adds partial-occlusion aware matching (skips tiny strips)",
    },
    {
        "name":               "+QUAL_EMA (FULL)",
        "use_bg_crop":        True,
        "use_strips":         True,
        "use_occlusion":      True,
        "use_quality_ema":    True,
        # QUAL_EMA controls the gallery UPDATE rate, not the matching distance.
        # It does not change the feature space scale, so no threshold adjustment needed
        # beyond what +STRIPS already requires.
        "threshold_override": 0.42,
        "description":        "Adds quality-weighted EMA update rate — full system",
    },
]


# ---------------------------------------------------------------------------
# Pretty table printer
# ---------------------------------------------------------------------------

def print_table(results: List[EvalResults]) -> None:
    print()
    print("  ABLATION STUDY — Re-ID Feature Evaluation")
    print("  " + "=" * 90)

    # ── Table 1: Re-ID accuracy metrics ───────────────────────────────────
    h1 = ["Config", "Precision", "Recall", "F1", "ID-Sw/100fr", "Consistency"]
    w1 = [22, 10, 8, 14, 12, 12]
    sep1 = "+" + "+".join("-" * (w + 2) for w in w1) + "+"
    fmt1 = "| " + " | ".join(f"{{:<{w}}}" for w in w1) + " |"

    print("\n  [ Accuracy & Consistency ]")
    print("  " + sep1)
    print("  " + fmt1.format(*h1))
    print("  " + sep1)

    baseline_f1 = None
    for r in results:
        d = r.summary_dict()
        f1_str = f"{d['F1']:.4f}"
        if baseline_f1 is not None:
            delta = d["F1"] - baseline_f1
            f1_str += f" ({'+' if delta >= 0 else ''}{delta:.4f})"
        if baseline_f1 is None:
            baseline_f1 = d["F1"]

        row = [
            d["Config"],
            f"{d['Precision']:.4f}",
            f"{d['Recall']:.4f}",
            f1_str[:14],
            f"{d['ID-Sw/100fr']:.2f}",
            f"{d['Consistency']:.4f}",
        ]
        print("  " + fmt1.format(*row))
    print("  " + sep1)

    # ── Table 2: Fragmentation / person-count metrics ─────────────────────
    h2 = ["Config", "IDs/fr (raw)", "IDs/fr (ReID)", "Total raw", "Total ReID", "ID reduction %", "Frag. rate"]
    w2 = [22, 13, 14, 10, 11, 15, 11]
    sep2 = "+" + "+".join("-" * (w + 2) for w in w2) + "+"
    fmt2 = "| " + " | ".join(f"{{:<{w}}}" for w in w2) + " |"

    print("\n  [ Fragmentation & Person Count ]")
    print("  " + sep2)
    print("  " + fmt2.format(*h2))
    print("  " + sep2)

    baseline_red = None
    for r in results:
        d = r.summary_dict()
        red_str = f"{d['ID reduction %']:.1f}%"
        if baseline_red is not None:
            delta = d["ID reduction %"] - baseline_red
            red_str += f" ({'+' if delta >= 0 else ''}{delta:.1f})"
        if baseline_red is None:
            baseline_red = d["ID reduction %"]

        row = [
            d["Config"],
            f"{d['IDs/fr (raw)']:.2f}",
            f"{d['IDs/fr (ReID)']:.2f}",
            str(d["Total raw IDs"]),
            str(d["Total persistent"]),
            red_str[:15],
            f"{d['Frag. rate']:.3f}",
        ]
        print("  " + fmt2.format(*row))
    print("  " + sep2)
    print()


def save_csv(results: List[EvalResults], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].summary_dict().keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r.summary_dict())
    print(f"  ✓ CSV saved → {path}")


def save_report(
    results:      List[EvalResults],
    configs:      List[dict],
    video_path:   str,
    n_frames:     int,
    n_tracks:     int,
    elapsed_secs: float,
    path:         str,
) -> None:
    lines = [
        "=" * 72,
        "REID ABLATION STUDY — FULL REPORT",
        "=" * 72,
        "",
        f"Video        : {video_path}",
        f"Frames used  : {n_frames}",
        f"Tracks found : {n_tracks}",
        f"Elapsed      : {elapsed_secs:.1f}s",
        "",
        "PROTOCOL",
        "--------",
        "Simulated Re-Entry Protocol: each track is split at a random",
        "midpoint. The second half is given a new tracker ID to simulate",
        "the person leaving and re-entering frame. Re-ID success = same",
        "persistent ID is assigned to both halves.",
        "",
        "METRICS",
        "-------",
        "Precision    = TP / (TP + FP)   correct re-assignments / all re-assignments",
        "Recall       = TP / (TP + FN)   correct re-assignments / all re-entry events",
        "F1           = harmonic mean of Precision and Recall",
        "ID-Sw/100fr  = ID switches per 100 frames (lower is better)",
        "Consistency  = mean fraction of frames with dominant label (higher = better)",
        "",
        "CONFIGURATIONS",
        "--------------",
    ]
    for i, cfg in enumerate(configs):
        lines.append(f"  Config {i}: {cfg['name']}")
        lines.append(f"           {cfg['description']}")
    lines += ["", "RESULTS", "-------"]
    for r in results:
        lines.append(str(r))
    lines += [
        "",
        "FRAGMENTATION METRICS EXPLAINED",
        "--------------------------------",
        "IDs/fr (raw)   : mean unique tracker IDs visible per frame BEFORE Re-ID.",
        "                 Each re-entry creates a new tracker ID, inflating this count.",
        "IDs/fr (ReID)  : mean unique persistent IDs visible per frame AFTER Re-ID.",
        "                 Should stay close to the true number of people in scene.",
        "Total raw IDs  : unique tracker IDs across the whole clip (before Re-ID).",
        "Total ReID IDs : unique persistent IDs across whole clip (after Re-ID).",
        "ID reduction % : (raw - persistent) / raw * 100.",
        "                 % of tracker IDs eliminated as spurious re-entry duplicates.",
        "                 Higher = better consolidation. 0% = no improvement.",
        "Frag. rate     : raw / persistent. 1.0 = perfect. 2.0 = doubled identities.",
        "",
        "=" * 72,
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✓ Report saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-ID ablation study: evaluates 5 configurations on a video."
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to input video file (mp4, avi, etc.)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=400,
        help="Maximum number of frames to process (default: 400)"
    )
    parser.add_argument(
        "--min-track-frames", type=int, default=25,
        help="Minimum track length to include in evaluation (default: 25)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.30,
        help="Re-ID match threshold for all configs (default: 0.30)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible track splits (default: 42)"
    )
    parser.add_argument(
        "--out-csv", default=None,
        help="Save results table to CSV file"
    )
    parser.add_argument(
        "--out-report", default=None,
        help="Save full text report to file"
    )
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    print()
    print(f"  Loading video: {video_path}")
    t0 = time.time()

    # --- Extract tracks ---------------------------------------------------
    print(f"  Extracting tracks (max {args.max_frames} frames) …")
    try:
        frames, tracks = extract_tracks_from_video(
            video_path,
            min_track_frames=args.min_track_frames,
            max_frames=args.max_frames,
        )
    except Exception as e:
        print(f"  ERROR extracting tracks: {e}")
        sys.exit(1)

    n_frames = len(frames)
    n_tracks = len(tracks)
    print(f"  → {n_frames} frames, {n_tracks} tracks (≥ {args.min_track_frames} frames each)")

    if n_tracks == 0:
        print("  ERROR: No tracks long enough to split. Try a longer video or lower --min-track-frames.")
        sys.exit(1)

    # --- Run ablation configs ---------------------------------------------
    all_results: List[EvalResults] = []

    for cfg in CONFIGS:
        thresh = cfg.get("threshold_override") or args.threshold
        reid = PersonReIdentifier(
            reid_threshold  = thresh,
            use_bg_crop     = cfg["use_bg_crop"],
            use_strips      = cfg["use_strips"],
            use_occlusion   = cfg["use_occlusion"],
            use_quality_ema = cfg["use_quality_ema"],
        )
        print("  Evaluating  [{:20s}] (threshold={:.2f}) …".format(cfg["name"], thresh),
              end=" ", flush=True)
        t_cfg = time.time()
        result = evaluate_config(
            frames=frames,
            tracks=tracks,
            reid=reid,
            config_name=cfg["name"],
            seed=args.seed,
        )
        print(f"done  ({time.time() - t_cfg:.1f}s)  "
              f"F1={result.f1:.4f}  P={result.precision:.4f}  R={result.recall:.4f}")
        all_results.append(result)

    elapsed = time.time() - t0

    # --- Print table ------------------------------------------------------
    print_table(all_results)

    # --- Save outputs -----------------------------------------------------
    if args.out_csv:
        save_csv(all_results, args.out_csv)

    if args.out_report:
        save_report(
            all_results, CONFIGS, video_path,
            n_frames, n_tracks, elapsed,
            args.out_report,
        )

    print(f"  Total elapsed: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
