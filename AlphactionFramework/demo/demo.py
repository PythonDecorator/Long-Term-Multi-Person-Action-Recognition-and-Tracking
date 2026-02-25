"""
AlphAction Demo — modernised entry point.

Supports video file input and webcam (realtime) mode.
Uses threading throughout; no multiprocessing / POSIX semaphores.
Compatible with Python 3.9+, PyTorch 2.x, Pillow 10+.
"""

from __future__ import annotations

import argparse
from itertools import count
from time import sleep

import torch
from tqdm import tqdm

from visualizer import AVAVisualizer
from action_predictor import AVAPredictorWorker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphAction — Action Detection Demo")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--webcam", action="store_true",
                     help="Use webcam as input (realtime mode)")
    src.add_argument("--video-path", default="input.mp4",
                     help="Path to input video file (default: input.mp4)")

    p.add_argument("--output-path", default="output.mp4",
                   help="Path for the output video (default: output.mp4)")
    p.add_argument("--cpu", action="store_true",
                   help="Run on CPU instead of CUDA")
    p.add_argument("--cfg-path",
                   default="../config_files/resnet101_8x8f_denseserial.yaml",
                   help="Path to model config YAML")
    p.add_argument("--weight-path",
                   default="../data/models/aia_models/resnet101_8x8f_denseserial.pth",
                   help="Path to model weights (.pth)")
    p.add_argument("--visual-threshold", default=0.5, type=float,
                   help="Confidence threshold for visualisation (default: 0.5)")
    p.add_argument("--start", default=0, type=int,
                   help="Start offset in milliseconds (default: 0)")
    p.add_argument("--duration", default=-1, type=int,
                   help="Duration in milliseconds, -1 = full video (default: -1)")
    p.add_argument("--detect-rate", default=4, type=int,
                   help="Action label update rate in fps (default: 4)")
    p.add_argument("--common-cate", action="store_true",
                   help="Use the common-category (15-class) model")
    p.add_argument("--hide-time", action="store_true",
                   help="Hide the timestamp overlay")
    p.add_argument("--tracker-box-thres", default=0.1, type=float,
                   help="Box confidence threshold for tracker (default: 0.1)")
    p.add_argument("--tracker-nms-thres", default=0.4, type=float,
                   help="NMS threshold for tracker (default: 0.4)")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    # Derived settings
    args.input_path = 0 if args.webcam else args.video_path
    args.device     = torch.device("cpu" if args.cpu else "cuda")
    args.realtime   = args.webcam

    # Tracker settings (multi-GPU not supported in demo)
    args.gpus        = [0] if torch.cuda.device_count() >= 1 else [-1]
    args.min_box_area = 0
    args.tracking    = True
    args.detector    = "tracker"
    args.debug       = False

    if args.webcam:
        print("Starting webcam demo — press Ctrl+C to stop.")
    else:
        print(f"Starting video demo: {args.video_path}")

    # ── Visualiser ──────────────────────────────────────────────────────────
    video_writer = AVAVisualizer(
        input_path          = args.input_path,
        output_path         = args.output_path,
        realtime            = args.realtime,
        start               = args.start,
        duration            = args.duration,
        show_time           = not args.hide_time,
        confidence_threshold= args.visual_threshold,
        common_cate         = args.common_cate,
    )

    # ── Predictor worker (threading-based, no spawn) ─────────────────────
    predictor = AVAPredictorWorker(args)
    pred_done = False

    print("Tracking frames (action prediction runs in background threads)…")
    try:
        for frame_idx in tqdm(count(), desc="Tracker", unit=" frame"):
            with torch.no_grad():
                orig_img, boxes, scores, ids = predictor.read_track()

            if orig_img is None:
                if not args.realtime:
                    predictor.signal_tracking_done()
                break

            if args.realtime:
                result = predictor.read_result()
                keep_going = video_writer.write_realtime_frame(
                    result, orig_img, boxes, scores, ids
                )
                if not keep_going:
                    break
            else:
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
        print("\nInterrupted by user.")

    # ── Flush remaining results (non-realtime) ───────────────────────────
    if not args.realtime:
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
        tqdm.write("Writing output video…")
        video_writer.show_progress(frame_idx)

    video_writer.close()
    predictor.terminate()
    print("Done.")


if __name__ == "__main__":
    main()
