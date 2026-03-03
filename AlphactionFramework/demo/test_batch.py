"""
test_batch.py — Robustness Evaluation Suite for AlphAction
===========================================================

Runs the action recognition demo across test videos organised by condition,
collects per-frame detection metrics, and writes:

  test_results/
    results_summary.csv     ← one row per video, all key metrics
    results_detailed.csv    ← one row per prediction event
    results_report.txt      ← dissertation-ready text report
    <condition>/            ← annotated output videos per condition

Folder layout required
----------------------
  demo/
    test_batch.py            ← this file
    test_videos/
      clean/                 ← 3 videos
      crowd/                 ← 3 videos
      fast_motion/           ← 3 videos
      occlusion/             ← 3 videos
      low_light/             ← 3 videos

Usage
-----
  python test_batch.py

  # Auto-resize + trim all videos to same resolution/length first:
  python test_batch.py --normalize --target-duration 10 --target-resolution 854x480

  # Custom model paths:
  python test_batch.py \\
      --cfg-path ../config_files/resnet50_4x16f_denseserial.yaml \\
      --weight-path ../data/models/aia_models/resnet50_4x16f_denseserial.pth
"""

from __future__ import annotations

# ── MUST be first — patches Conv3d before any alphaction import ──────────────
import torch.nn as nn
_orig_conv3d = nn.Conv3d.__init__
def _safe_conv3d(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', **kwargs):
    def _safe(x):
        if isinstance(x, str):           return x
        if isinstance(x, (tuple, list)): return tuple(int(i) for i in x)
        return int(x)
    _orig_conv3d(self, in_channels, out_channels,
                 _safe(kernel_size), _safe(stride), _safe(padding),
                 _safe(dilation), int(groups), bias, padding_mode, **kwargs)
nn.Conv3d.__init__ = _safe_conv3d
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import count
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Tuple

import cv2
import torch
from tqdm import tqdm

DEMO_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(DEMO_DIR))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONDITIONS = ["clean", "crowd", "fast_motion", "occlusion", "low_light"]

DEFAULT_CFG     = "../config_files/resnet50_4x16f_denseserial.yaml"
DEFAULT_WEIGHTS = "../data/models/aia_models/resnet50_4x16f_denseserial.pth"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VideoMetrics:
    condition:               str   = ""
    video_name:              str   = ""
    video_path:              str   = ""
    original_width:          int   = 0
    original_height:         int   = 0
    original_fps:            float = 0.0
    original_duration_s:     float = 0.0
    original_frames:         int   = 0
    total_frames_processed:  int   = 0
    frames_with_detections:  int   = 0
    detection_rate:          float = 0.0
    total_person_detections: int   = 0
    avg_persons_per_frame:   float = 0.0
    max_persons_in_frame:    int   = 0
    total_action_predictions:int   = 0
    unique_actions_detected: int   = 0
    avg_confidence:          float = 0.0
    max_confidence:          float = 0.0
    min_confidence:          float = 1.0
    actions_detected:        str   = ""
    processing_time_s:       float = 0.0
    processing_fps:          float = 0.0
    success:                 bool  = True
    error_msg:               str   = ""


@dataclass
class PredictionEvent:
    condition:    str
    video_name:   str
    timestamp_ms: float
    person_id:    int
    action:       str
    confidence:   float


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def get_video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = dict(
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps        = fps,
        n_frames   = n,
        duration_s = n / fps if fps > 0 else 0,
    )
    cap.release()
    return info


def normalize_video(src: str, dst: str, w: int, h: int, dur: int) -> bool:
    """Resize + trim a video using ffmpeg. Returns True on success."""
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-t", str(dur),
        "-vf", f"scale={w}:{h}",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-an", dst,
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0


# ---------------------------------------------------------------------------
# Metrics capture — monkey-patches visualizer without modifying its source
# ---------------------------------------------------------------------------

class MetricsCapture:
    """
    Intercepts AVAVisualizer._update_action_dict() and send_track()
    to collect metrics. Must be attached right after AVAVisualizer is
    created, before any frames are sent.
    """

    def __init__(self, visualizer, condition: str, video_name: str):
        self.condition  = condition
        self.video_name = video_name

        self.frame_person_counts:   List[int]             = []
        self.prediction_events:     List[PredictionEvent] = []
        self.action_confidence_all: List[float]           = []

        # Patch _update_action_dict
        orig_update = visualizer._update_action_dict

        def patched_update(scores, ids):
            if scores is not None and ids is not None:
                for score, pid in zip(scores, ids):
                    above = torch.nonzero(
                        score >= visualizer.thresh, as_tuple=False
                    ).squeeze(1).tolist()
                    for cid in above:
                        if cid in visualizer.excl_ids:
                            continue
                        conf  = float(score[cid])
                        label = visualizer.cate_list[cid]
                        self.action_confidence_all.append(conf)
                        self.prediction_events.append(PredictionEvent(
                            condition    = self.condition,
                            video_name   = self.video_name,
                            timestamp_ms = 0.0,
                            person_id    = int(pid),
                            action       = label,
                            confidence   = conf,
                        ))
            orig_update(scores, ids)

        visualizer._update_action_dict = patched_update

        # Patch send_track to count detections per frame
        orig_send_track = visualizer.send_track

        def patched_send_track(result):
            if isinstance(result, tuple):
                boxes, ids = result
                self.frame_person_counts.append(
                    len(boxes) if boxes is not None else 0
                )
            orig_send_track(result)

        visualizer.send_track = patched_send_track

    def fill(self, vm: VideoMetrics) -> VideoMetrics:
        counts = self.frame_person_counts
        vm.total_frames_processed   = len(counts)
        vm.frames_with_detections   = sum(1 for c in counts if c > 0)
        vm.detection_rate           = (
            vm.frames_with_detections / vm.total_frames_processed
            if vm.total_frames_processed > 0 else 0.0
        )
        vm.total_person_detections  = sum(counts)
        vm.avg_persons_per_frame    = (
            vm.total_person_detections / vm.total_frames_processed
            if vm.total_frames_processed > 0 else 0.0
        )
        vm.max_persons_in_frame     = max(counts) if counts else 0

        events = self.prediction_events
        vm.total_action_predictions = len(events)
        vm.unique_actions_detected  = len({e.action for e in events})
        vm.actions_detected         = ", ".join(sorted({e.action for e in events}))

        confs = self.action_confidence_all
        if confs:
            vm.avg_confidence = sum(confs) / len(confs)
            vm.max_confidence = max(confs)
            vm.min_confidence = min(confs)
        return vm


# ---------------------------------------------------------------------------
# Single-video runner — exact replica of demo.py main() loop
# ---------------------------------------------------------------------------

def run_single_video(
    video_path:  str,
    output_path: str,
    condition:   str,
    cfg_path:    str,
    weight_path: str,
    device:      torch.device,
) -> Tuple[VideoMetrics, List[PredictionEvent]]:
    """
    Mirrors demo.py main() exactly.
    The Conv3d patch at the top of this file is already applied
    before this function is ever called.
    """
    from action_predictor import AVAPredictorWorker
    from visualizer import AVAVisualizer

    video_name = Path(video_path).stem
    vm = VideoMetrics(condition=condition, video_name=video_name,
                      video_path=video_path)

    info = get_video_info(video_path)
    vm.original_width      = info.get("width",      0)
    vm.original_height     = info.get("height",     0)
    vm.original_fps        = info.get("fps",        0.0)
    vm.original_duration_s = info.get("duration_s", 0.0)
    vm.original_frames     = info.get("n_frames",   0)

    # Build args namespace exactly as demo.py does after argparse
    class Args:
        pass
    args = Args()
    args.input_path       = video_path
    args.output_path      = output_path
    args.device           = device
    args.realtime         = False
    args.start            = 0
    args.duration         = -1
    args.detect_rate      = 4
    args.common_cate      = False
    args.visual_threshold = 0.5
    args.cfg_path         = cfg_path
    args.weight_path      = weight_path
    args.gpus             = [0] if torch.cuda.device_count() >= 1 else [-1]
    args.min_box_area     = 0
    args.tracking         = True
    args.detector         = "tracker"
    args.debug            = False

    t_start = time.time()

    try:
        os.makedirs(Path(output_path).parent, exist_ok=True)

        # ── Visualiser ────────────────────────────────────────────────
        video_writer = AVAVisualizer(
            input_path           = args.input_path,
            output_path          = args.output_path,
            realtime             = False,
            start                = args.start,
            duration             = args.duration,
            show_time            = True,
            confidence_threshold = args.visual_threshold,
            common_cate          = args.common_cate,
        )

        # Attach metrics capture BEFORE any frames flow
        capture = MetricsCapture(video_writer, condition, video_name)

        # ── Predictor ─────────────────────────────────────────────────
        predictor = AVAPredictorWorker(args)
        pred_done = False
        frame_idx = 0

        # ── Main loop (identical to demo.py) ──────────────────────────
        try:
            for frame_idx in tqdm(
                count(),
                desc=f"  [{condition}] {video_name}",
                unit=" frame",
                leave=False,
            ):
                with torch.no_grad():
                    orig_img, boxes, scores, ids = predictor.read_track()

                if orig_img is None:
                    predictor.signal_tracking_done()
                    break

                video_writer.send_track((boxes, ids))
                while not pred_done:
                    result = predictor.read_result()
                    if result is None:
                        break
                    elif result == "done":
                        pred_done = True
                    else:
                        video_writer.send_result(result)

        except KeyboardInterrupt:
            print(f"\n  Interrupted during {video_name}.")

        # ── Flush (identical to demo.py) ──────────────────────────────
        video_writer.send_track("DONE")

        while not pred_done:
            result = predictor.read_result()
            if result is None:
                sleep(0.05)
            elif result == "done":
                pred_done = True
            else:
                video_writer.send_result(result)

        video_writer.send_result("DONE")
        video_writer.show_progress(frame_idx)
        video_writer.close()
        predictor.terminate()

        vm.processing_time_s = time.time() - t_start
        vm.processing_fps    = (
            frame_idx / vm.processing_time_s
            if vm.processing_time_s > 0 else 0.0
        )
        capture.fill(vm)

    except Exception as e:
        vm.success   = False
        vm.error_msg = str(e)
        vm.processing_time_s = time.time() - t_start
        print(f"\n  ERROR in {video_name}: {e}")
        traceback.print_exc()

    return vm, (capture.prediction_events if vm.success else [])


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = [
    "condition", "video_name",
    "original_width", "original_height", "original_fps",
    "original_duration_s", "original_frames",
    "total_frames_processed", "frames_with_detections", "detection_rate",
    "total_person_detections", "avg_persons_per_frame", "max_persons_in_frame",
    "total_action_predictions", "unique_actions_detected",
    "avg_confidence", "max_confidence", "min_confidence",
    "actions_detected",
    "processing_time_s", "processing_fps",
    "success", "error_msg",
]

DETAILED_FIELDS = [
    "condition", "video_name", "timestamp_ms",
    "person_id", "action", "confidence",
]


def write_csv(path: str, rows: list, fields: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def write_report(
    path: str, all_metrics: List[VideoMetrics], ts: str
) -> None:

    lines = []
    add = lines.append

    def hr1(t):
        add(""); add("=" * 70); add(t); add("=" * 70)

    def hr2(t):
        add(""); add(t); add("-" * len(t))

    hr1("ALPHACTION ROBUSTNESS EVALUATION REPORT")
    add(f"Generated  : {ts}")
    add(f"Videos     : {len(all_metrics)}")
    add(f"Conditions : {', '.join(CONDITIONS)}")

    by_cond: Dict[str, List[VideoMetrics]] = {}
    for vm in all_metrics:
        by_cond.setdefault(vm.condition, []).append(vm)

    # Per-condition detail
    hr1("RESULTS BY CONDITION")
    for cond in CONDITIONS:
        vms = by_cond.get(cond, [])
        ok  = [v for v in vms if v.success]
        hr2(f"Condition: {cond.upper().replace('_',' ')} "
            f"({len(vms)} videos, {len(ok)} succeeded)")
        if not ok:
            add("  No successful runs.")
            continue

        def avg(f):
            return sum(f(v) for v in ok) / len(ok)

        all_actions = sorted({
            a.strip() for v in ok
            for a in v.actions_detected.split(",") if a.strip()
        })

        add(f"  Avg detection rate       : {avg(lambda v: v.detection_rate):.1%}")
        add(f"  Avg persons / frame      : {avg(lambda v: v.avg_persons_per_frame):.2f}")
        add(f"  Avg action confidence    : {avg(lambda v: v.avg_confidence):.3f}")
        add(f"  Avg processing speed     : {avg(lambda v: v.processing_fps):.1f} fps")
        add(f"  Unique actions observed  : {len(all_actions)}")
        add(f"  Actions: {', '.join(all_actions) or 'none'}")
        add("")
        for v in ok:
            add(f"  [{v.video_name}]")
            add(f"    Resolution  : {v.original_width}x{v.original_height}"
                f" @ {v.original_fps:.1f} fps")
            add(f"    Duration    : {v.original_duration_s:.1f}s"
                f" ({v.original_frames} frames)")
            add(f"    Det. rate   : {v.detection_rate:.1%}"
                f" (max {v.max_persons_in_frame} people/frame)")
            add(f"    Predictions : {v.total_action_predictions}"
                f"  conf avg={v.avg_confidence:.2f}"
                f" max={v.max_confidence:.2f}"
                f" min={v.min_confidence:.2f}")
            add(f"    Actions     : {v.actions_detected or 'none'}")
            add(f"    Proc. time  : {v.processing_time_s:.1f}s"
                f" ({v.processing_fps:.1f} fps)")

    # Cross-condition table
    hr1("CROSS-CONDITION COMPARISON TABLE")
    add("(Averages per condition — paste directly into dissertation)")
    add("")
    hdr = (f"{'Condition':<15} {'N':>4} {'Det.Rate':>10}"
           f" {'Persons/f':>10} {'AvgConf':>9} {'ProcFPS':>9} {'Actions':>8}")
    add(hdr)
    add("-" * len(hdr))
    for cond in CONDITIONS:
        ok = [v for v in by_cond.get(cond, []) if v.success]
        if not ok:
            add(f"{cond:<15} {'0':>4}   (no data)")
            continue
        def avg(f):
            return sum(f(v) for v in ok) / len(ok)
        add(
            f"{cond:<15}"
            f" {len(ok):>4}"
            f" {avg(lambda v: v.detection_rate):>10.1%}"
            f" {avg(lambda v: v.avg_persons_per_frame):>10.2f}"
            f" {avg(lambda v: v.avg_confidence):>9.3f}"
            f" {avg(lambda v: v.processing_fps):>9.1f}"
            f" {int(avg(lambda v: v.unique_actions_detected)):>8}"
        )

    # Degradation analysis
    hr1("DEGRADATION ANALYSIS vs CLEAN BASELINE")
    clean_ok = [v for v in by_cond.get("clean", []) if v.success]
    if not clean_ok:
        add("  No clean baseline videos found — add videos to test_videos/clean/")
    else:
        def cavg(f):
            return sum(f(v) for v in clean_ok) / len(clean_ok)
        b_det  = cavg(lambda v: v.detection_rate)
        b_conf = cavg(lambda v: v.avg_confidence)
        b_per  = cavg(lambda v: v.avg_persons_per_frame)
        add(f"  Baseline (clean) — det_rate: {b_det:.1%} | "
            f"avg_conf: {b_conf:.3f} | avg_persons: {b_per:.2f}")
        add("")
        add(f"  {'Condition':<15} {'DetRate Δ':>12} {'Conf Δ':>10}"
            f" {'Persons Δ':>12}  Assessment")
        add("  " + "-" * 66)
        for cond in [c for c in CONDITIONS if c != "clean"]:
            ok = [v for v in by_cond.get(cond, []) if v.success]
            if not ok:
                add(f"  {cond:<15}  (no data)")
                continue
            def oavg(f):
                return sum(f(v) for v in ok) / len(ok)
            d_det  = oavg(lambda v: v.detection_rate)        - b_det
            d_conf = oavg(lambda v: v.avg_confidence)        - b_conf
            d_per  = oavg(lambda v: v.avg_persons_per_frame) - b_per
            if   d_det < -0.20: assess = "Significant degradation"
            elif d_det < -0.08: assess = "Moderate degradation"
            elif d_det < 0:     assess = "Mild degradation"
            else:               assess = "No degradation / improvement"
            add(f"  {cond:<15}"
                f" {d_det:>+11.1%}"
                f" {d_conf:>+9.3f}"
                f" {d_per:>+11.2f}"
                f"  {assess}")

    failed = [v for v in all_metrics if not v.success]
    if failed:
        hr1("FAILED VIDEOS")
        for v in failed:
            add(f"  {v.condition}/{v.video_name}: {v.error_msg}")

    add("")
    add("END OF REPORT")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AlphAction robustness batch evaluation"
    )
    parser.add_argument("--video-dir",         default="test_videos")
    parser.add_argument("--output-dir",        default="test_results")
    parser.add_argument("--normalize",         action="store_true",
                        help="Resize + trim all videos before running")
    parser.add_argument("--target-resolution", default="1920x1080",
                        help="WxH when --normalize (default: 1920x1080)")
    parser.add_argument("--target-duration",   default=20, type=int,
                        help="Seconds to trim to when --normalize (default: 20)")
    parser.add_argument("--cfg-path",    default=DEFAULT_CFG)
    parser.add_argument("--weight-path", default=DEFAULT_WEIGHTS)
    parser.add_argument("--cpu",         action="store_true")
    args = parser.parse_args()

    device    = torch.device("cpu" if args.cpu else "cuda")
    video_dir = Path(args.video_dir)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H   = (int(x) for x in args.target_resolution.split("x"))
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"  AlphAction Robustness Evaluation")
    print(f"  {run_ts}")
    print(f"{'='*60}\n")

    # Discover videos
    entries: List[Tuple[str, str, str]] = []
    for cond in CONDITIONS:
        cond_dir = video_dir / cond
        if not cond_dir.exists():
            print(f"  [skip] missing: {cond_dir}")
            continue
        videos = sorted(
            p for p in cond_dir.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        if not videos:
            print(f"  [skip] empty: {cond_dir}")
            continue
        proc_dir = out_dir / "preprocessed" / cond
        proc_dir.mkdir(parents=True, exist_ok=True)
        for v in videos:
            entries.append((cond, str(v), str(proc_dir / v.name)))

    if not entries:
        print("No videos found. Check your test_videos/ structure.")
        sys.exit(1)

    print(f"Found {len(entries)} videos across "
          f"{len({e[0] for e in entries})} conditions.\n")

    # Optional normalization
    if args.normalize:
        print("Normalizing videos ...")
        for cond, src, dst in entries:
            if Path(dst).exists():
                print(f"  [skip] {Path(dst).name}")
                continue
            ok = normalize_video(src, dst, W, H, args.target_duration)
            print(f"  [{'OK' if ok else 'FAIL'}] {cond}/{Path(src).name}")
        print()

    # Process each video
    all_metrics: List[VideoMetrics]    = []
    all_events:  List[PredictionEvent] = []

    for cond, orig, proc in entries:
        src  = proc if (args.normalize and Path(proc).exists()) else orig
        name = Path(src).stem
        dst  = str(out_dir / cond / f"{name}_output.mp4")

        print(f"\n[{cond}] {name}")

        vm, events = run_single_video(
            video_path  = src,
            output_path = dst,
            condition   = cond,
            cfg_path    = args.cfg_path,
            weight_path = args.weight_path,
            device      = device,
        )
        all_metrics.append(vm)
        all_events.extend(events)

        if vm.success:
            print(f"  ✓  det={vm.detection_rate:.0%}"
                  f"  persons/f={vm.avg_persons_per_frame:.1f}"
                  f"  conf={vm.avg_confidence:.2f}"
                  f"  actions={vm.unique_actions_detected}"
                  f"  {vm.processing_time_s:.0f}s @ {vm.processing_fps:.1f}fps")
        else:
            print(f"  ✗  {vm.error_msg}")

    # Write results
    print(f"\n{'='*60}")
    print("Writing results ...")
    write_csv(str(out_dir / "results_summary.csv"),
              [asdict(v) for v in all_metrics], SUMMARY_FIELDS)
    write_csv(str(out_dir / "results_detailed.csv"),
              [asdict(e) for e in all_events],  DETAILED_FIELDS)
    write_report(str(out_dir / "results_report.txt"), all_metrics, run_ts)

    # Console summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'Condition':<15} {'N':>4} {'Det.Rate':>10} {'AvgConf':>9} {'FPS':>7}")
    print("-" * 50)
    by_cond: Dict[str, List[VideoMetrics]] = {}
    for vm in all_metrics:
        by_cond.setdefault(vm.condition, []).append(vm)
    for cond in CONDITIONS:
        ok = [v for v in by_cond.get(cond, []) if v.success]
        if not ok:
            print(f"{cond:<15} {'0':>4}")
            continue
        def avg(f):
            return sum(f(v) for v in ok) / len(ok)
        print(f"{cond:<15}"
              f" {len(ok):>4}"
              f" {avg(lambda v: v.detection_rate):>10.1%}"
              f" {avg(lambda v: v.avg_confidence):>9.3f}"
              f" {avg(lambda v: v.processing_fps):>7.1f}")

    print(f"\nResults saved to: {out_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
