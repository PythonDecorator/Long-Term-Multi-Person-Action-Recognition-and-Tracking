"""
run_all.py  — One-shot evaluation suite for AlphAction+ project
================================================================
Drop your videos into folders, run this script, get everything:

  1. Re-ID ablation metrics (Table 4.4)
  2. Thermal FID evaluation  (Table 4.5)
  3. Visual evidence images  (figures for the dissertation)
       - Re-ID before/after frames strip (Figure A)
       - Thermal pipeline side-by-side   (Figure B)

FOLDER LAYOUT  (create these, put your videos inside)
──────────────────────────────────────────────────────
  videos/
    reid/          ← your Re-ID test videos (mp4/avi/mov)
    thermal/       ← your thermal / RGB videos for FID

OUTPUT
──────
  results/
    ablation_results.csv
    ablation_report.txt
    thermal_fid.csv
    thermal_report.txt
    figures/
      reid_before_after.png    ← Re-ID visual evidence
      reid_reentry_strip.png   ← Person label across re-entry
      thermal_pipeline.png     ← RGB→Lum→Blur→INFERNO→CLAHE

USAGE
─────
  # All defaults (videos in ./videos/reid/ and ./videos/thermal/):
  python run_all.py

  # Custom folders:
  python run_all.py --reid-dir my_reid_vids --thermal-dir my_thermal_vids

  # Skip Re-ID ablation (just do thermal + figures):
  python run_all.py --skip-ablation

  # Skip thermal FID (just do Re-ID + figures):
  python run_all.py --skip-thermal

Author: Amos Okpe  (MSc Artificial Intelligence and Data Science, University of Hull)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── optional rich progress bars ──────────────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(it, **kw): return it

# ── matplotlib (needed for INFERNO map + figure export) ──────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found — thermal pipeline figure will be skipped.\n"
          "         Install with:  pip install matplotlib")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ═════════════════════════════════════════════════════════════════════════════
# THERMAL ADAPTER  (standalone, no external deps beyond cv2 + matplotlib)
# ═════════════════════════════════════════════════════════════════════════════

class ThermalAdapter:
    def __init__(self, sigma=2.0, clahe_clip=2.0, clahe_tile=8):
        self.sigma = sigma
        ksize = int(6*sigma+1)|1
        self.ksize = (ksize, ksize)
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                      tileGridSize=(clahe_tile, clahe_tile))

    def _luminance(self, bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return lab[:,:,0]

    def _blur(self, L):
        return cv2.GaussianBlur(L, self.ksize, self.sigma)

    def _inferno(self, L_blurred):
        normed = L_blurred.astype(np.float32)/255.0
        if HAS_MPL:
            rgba = cm.inferno(normed)
            rgb  = (rgba[:,:,:3]*255).astype(np.uint8)
        else:
            r = np.clip(normed*2.0,   0,1)
            g = np.clip(normed*1.3-0.3,0,1)
            b = np.zeros_like(normed)
            rgb = (np.stack([r,g,b],2)*255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _clahe(self, bgr_inferno):
        out = bgr_inferno.copy()
        for c in range(3):
            out[:,:,c] = self.clahe.apply(bgr_inferno[:,:,c])
        return out

    def full_pipeline_stages(self, bgr):
        """Return all 5 stages as BGR images: RGB, Lum, Blur, INFERNO, CLAHE."""
        L       = self._luminance(bgr)
        blurred = self._blur(L)
        inferno = self._inferno(blurred)
        clahe   = self._clahe(inferno)
        lum_3ch  = cv2.merge([L, L, L])
        blur_3ch = cv2.merge([blurred, blurred, blurred])
        return bgr, lum_3ch, blur_3ch, inferno, clahe

    def convert(self, bgr):
        L = self._luminance(bgr)
        return self._clahe(self._inferno(self._blur(L)))

    def greyscale(self, bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.merge([g,g,g])

    def inferno_no_clahe(self, bgr):
        L = self._luminance(bgr)
        return self._inferno(self._blur(L))


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — THERMAL PIPELINE  (RGB → Lum → Blur → INFERNO → CLAHE)
# ═════════════════════════════════════════════════════════════════════════════

def make_thermal_pipeline_figure(
    video_paths: List[str],
    out_path: str,
    n_rows: int = 3,        # how many source frames to show
) -> bool:
    """
    Picks n_rows evenly-spaced frames from the first video that has content,
    runs each through ThermalAdapter, and saves a grid:

        Row 1: [RGB] [Luminance] [Gaussian blur] [INFERNO] [INFERNO+CLAHE]
        Row 2: ...
        ...

    Returns True on success.
    """
    if not HAS_MPL:
        print("  [skip] thermal figure — matplotlib not available")
        return False

    adapter = ThermalAdapter()

    # Find a usable video and extract sample frames
    sample_frames = []
    for vp in video_paths:
        cap   = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 10:
            cap.release(); continue
        indices = np.linspace(total//5, 4*total//5, n_rows, dtype=int)
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if ok:
                # Resize to 480p width for manageable figure
                h, w = frame.shape[:2]
                new_w = 480
                new_h = int(h * new_w / w)
                sample_frames.append(cv2.resize(frame, (new_w, new_h)))
        cap.release()
        if len(sample_frames) >= n_rows:
            break

    if not sample_frames:
        print("  [skip] thermal figure — no frames extracted")
        return False

    sample_frames = sample_frames[:n_rows]
    labels = ["Input Frame\n(Original / Thermal)", "Luminance (L*)", "Gaussian Blur\n(σ=2.0)",
              "INFERNO\nFalse-colour", "INFERNO + CLAHE\n(ThermalAdapter)"]

    n_cols = 5
    fig_w  = 5 * n_cols
    fig_h  = 4 * len(sample_frames) + 1.2   # extra for title + labels

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0f0f0f")
    # Detect whether input looks like thermal (check if mostly warm-toned)
    test_frame = sample_frames[0]
    r_mean = test_frame[:,:,2].mean()  # BGR: [2] = R
    b_mean = test_frame[:,:,0].mean()  # BGR: [0] = B
    input_type = "Thermal Input" if r_mean > b_mean + 30 else "RGB Input"
    fig.suptitle(
        f"Thermal Adaptation Pipeline: {input_type} → Pseudo-Thermal Conversion",
        fontsize=16, color="white", fontweight="bold", y=0.995
    )

    gs = gridspec.GridSpec(
        len(sample_frames), n_cols,
        figure=fig,
        hspace=0.08, wspace=0.04,
        left=0.01, right=0.99, top=0.93, bottom=0.06
    )

    stage_colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0"]

    for row_i, bgr in enumerate(sample_frames):
        stages = adapter.full_pipeline_stages(bgr)
        for col_i, (img_bgr, lbl, col) in enumerate(zip(stages, labels, stage_colors)):
            ax = fig.add_subplot(gs[row_i, col_i])
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(col); spine.set_linewidth(2.5)
            if row_i == 0:
                ax.set_title(lbl, fontsize=11, color=col,
                             fontweight="bold", pad=6, wrap=True)
            if col_i == 0:
                ax.set_ylabel(f"Frame {row_i+1}", fontsize=9,
                              color="#aaaaaa", rotation=90, labelpad=4)

    # Column labels and arrows are already drawn via ax.set_title above.
    # No extra arrows needed — titles serve as stage labels.

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  Thermal pipeline figure → {out_path}")
    return True


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Re-ID BEFORE / AFTER STRIP
# ═════════════════════════════════════════════════════════════════════════════

# Person colours (BGR) matching person_reid.py
# ═════════════════════════════════════════════════════════════════════════════
# FID  (Fréchet Inception Distance — no scipy needed)
# ═════════════════════════════════════════════════════════════════════════════

def _pca_project(fa, fb, n_components=64):
    """Reduce to n_components dims via PCA fitted on the union.
    Needed when n_samples < n_features (e.g. 300 frames, 2048 dims)
    — otherwise covariance is rank-deficient and FID collapses to 0."""
    n_components = min(n_components, fa.shape[0]-2, fb.shape[0]-2, fa.shape[1])
    if n_components >= fa.shape[1]:
        return fa, fb   # already low-dim enough
    combined = np.vstack([fa, fb])
    mu = combined.mean(0)
    _, _, Vt = np.linalg.svd(combined - mu, full_matrices=False)
    P = Vt[:n_components].T        # (2048, 64)
    return (fa - mu) @ P, (fb - mu) @ P


def _compute_fid(fa: np.ndarray, fb: np.ndarray) -> float:
    # Project to 64 dims — avoids rank-deficient covariance with few frames
    fa, fb = _pca_project(fa, fb, n_components=64)
    mu_a = np.mean(fa, 0);  mu_b = np.mean(fb, 0)
    eps  = np.eye(fa.shape[1]) * 1e-5
    sa   = np.cov(fa, rowvar=False) + eps
    sb   = np.cov(fb, rowvar=False) + eps
    diff = mu_a - mu_b
    vals, vecs = np.linalg.eigh(sa @ sb)
    vals  = np.maximum(vals, 0.0)
    sqrtm = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    return float(max(0.0, diff @ diff + np.trace(sa + sb - 2 * sqrtm)))


def run_fid_evaluation(video_paths: List[str], n_per_video=60) -> Dict[str,float]:
    import torch
    import torchvision.models  as models
    import torchvision.transforms as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  FID evaluation on {device} ...")

    # Sample frames
    rgb_frames: List[np.ndarray] = []
    for vp in video_paths:
        cap   = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 5: cap.release(); continue
        step  = max(1, total//n_per_video)
        for fi in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, f = cap.read()
            if ok: rgb_frames.append(cv2.resize(f, (299,299)))
            if len(rgb_frames) >= n_per_video*len(video_paths): break
        cap.release()
    if not rgb_frames:
        print("  [skip] FID — no frames extracted"); return {}
    print(f"  {len(rgb_frames)} frames sampled for FID")

    adapter = ThermalAdapter()
    grey_f  = [adapter.greyscale(f)          for f in rgb_frames]
    inf_f   = [adapter.inferno_no_clahe(f)   for f in rgb_frames]
    therm_f = [adapter.convert(f)            for f in rgb_frames]

    # Inception-v3 features
    inc = models.inception_v3(pretrained=True, transform_input=False)
    inc_feat = torch.nn.Sequential(
        inc.Conv2d_1a_3x3, inc.Conv2d_2a_3x3, inc.Conv2d_2b_3x3,
        torch.nn.MaxPool2d(3,2),
        inc.Conv2d_3b_1x1, inc.Conv2d_4a_3x3, torch.nn.MaxPool2d(3,2),
        inc.Mixed_5b, inc.Mixed_5c, inc.Mixed_5d,
        inc.Mixed_6a, inc.Mixed_6b, inc.Mixed_6c, inc.Mixed_6d, inc.Mixed_6e,
        inc.Mixed_7a, inc.Mixed_7b, inc.Mixed_7c,
        torch.nn.AdaptiveAvgPool2d((1,1)), torch.nn.Flatten(),
    ).eval().to(device)

    tf = T.Compose([T.ToPILImage(), T.Resize((299,299)), T.ToTensor(),
                    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    def get_feats(bgr_list, bs=32):
        out = []
        with torch.no_grad():
            for i in range(0, len(bgr_list), bs):
                batch = torch.stack(
                    [tf(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)) for f in bgr_list[i:i+bs]]
                ).to(device)
                out.append(inc_feat(batch).cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)

    print("  Extracting Inception features (takes ~2 min)...")
    fr  = get_feats(rgb_frames)
    fg  = get_feats(grey_f)
    fi  = get_feats(inf_f)
    ft  = get_feats(therm_f)
    h   = len(fr)//2
    fid_self = _compute_fid(fr[:h], fr[h:]) if h>1 else 0.0

    return {
        "RGB (self-split reference)":                  round(fid_self,  2),
        "Greyscale (3-ch replicated)":                 round(_compute_fid(fr,fg), 2),
        "Luminance + INFERNO (no CLAHE)":              round(_compute_fid(fr,fi), 2),
        "Luminance + INFERNO + CLAHE (ThermalAdapter)":round(_compute_fid(fr,ft), 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Re-ID ABLATION  (delegates to run_ablation.py logic inline)
# ═════════════════════════════════════════════════════════════════════════════

def run_reid_ablation(
    video_paths: List[str],
    out_csv:    str,
    out_report: str,
    max_frames: int = 400,
    min_track:  int = 20,
    seed:       int = 42,
) -> bool:
    try:
        from person_reid import PersonReIdentifier
        from reid_evaluator import (
            extract_tracks_from_video, evaluate_config,
            print_before_after_summary, EvalResults
        )
    except ImportError as e:
        print(f"  [skip] Re-ID ablation — {e}")
        return False

    CONFIGS = [
        {"name":"BASELINE",        "use_deep":False,"use_bg_crop":False,"use_qual":False,"thresh":0.40},
        {"name":"+BG_CROP (HSV)",  "use_deep":False,"use_bg_crop":True, "use_qual":False,"thresh":0.40},
        {"name":"+DEEP (ResNet50)","use_deep":True, "use_bg_crop":False,"use_qual":False,"thresh":0.30},
        {"name":"+BG_CROP (Deep)", "use_deep":True, "use_bg_crop":True, "use_qual":False,"thresh":0.30},
        {"name":"FULL SYSTEM",     "use_deep":True, "use_bg_crop":True, "use_qual":True, "thresh":0.30},
    ]

    all_results_by_video: Dict[str, List] = {}

    for vp in video_paths:
        vname = Path(vp).name
        print(f"\n  ─── Video: {vname} ───────────────────────────────")
        try:
            frames, tracks = extract_tracks_from_video(
                vp, min_track_frames=min_track, max_frames=max_frames
            )
        except Exception as e:
            print(f"  [skip] {vname}: {e}"); continue

        if not tracks:
            print(f"  [skip] {vname}: no tracks found"); continue
        print(f"  {len(frames)} frames  |  {len(tracks)} tracks (≥{min_track} frames each)")

        video_results = []
        for cfg in CONFIGS:
            reid = PersonReIdentifier(
                use_deep_features    = cfg["use_deep"],
                use_bg_crop          = cfg["use_bg_crop"],
                use_quality_ema      = cfg["use_qual"],
                reid_threshold       = cfg["thresh"],
                reid_threshold_relax = cfg["thresh"] + 0.20,
                lock_release_frames  = 3,   # short for simulation: tracks ~40-60 frames
            )
            t0 = time.time()
            print(f"  [{cfg['name']:<20}] ...", end=" ", flush=True)
            result = evaluate_config(frames, tracks, reid,
                                     cfg["name"], seed=seed)
            elapsed = time.time()-t0
            print(f"done ({elapsed:.0f}s)  "
                  f"F1={result.f1:.4f}  P={result.precision:.4f}  "
                  f"R={result.recall:.4f}  "
                  f"Events={result.n_reentry_events}  TP={result.n_tp}")
            video_results.append(result)
            print_before_after_summary(result, cfg["name"])

        all_results_by_video[vname] = video_results

    if not all_results_by_video:
        print("  No videos processed for Re-ID ablation."); return False

    # Average across videos
    config_names = [c["name"] for c in CONFIGS]
    avg_results  = []
    for i, cname in enumerate(config_names):
        per_vid = [all_results_by_video[vn][i]
                   for vn in all_results_by_video
                   if i < len(all_results_by_video[vn])]
        if not per_vid: continue
        from reid_evaluator import EvalResults
        avg = EvalResults(
            config_name        = cname + " (avg)",
            precision          = np.mean([r.precision          for r in per_vid]),
            recall             = np.mean([r.recall             for r in per_vid]),
            f1                 = np.mean([r.f1                 for r in per_vid]),
            id_switches_per100 = np.mean([r.id_switches_per100 for r in per_vid]),
            consistency        = np.mean([r.consistency        for r in per_vid]),
            id_reduction_pct   = np.mean([r.id_reduction_pct   for r in per_vid]),
            ids_per_frame_raw  = np.mean([r.ids_per_frame_raw  for r in per_vid]),
            ids_per_frame_reid = np.mean([r.ids_per_frame_reid for r in per_vid]),
            frag_rate          = np.mean([r.frag_rate          for r in per_vid]),
            total_raw_ids      = int(np.mean([r.total_raw_ids  for r in per_vid])),
            total_persistent   = int(np.mean([r.total_persistent for r in per_vid])),
            n_reentry_events   = int(np.mean([r.n_reentry_events for r in per_vid])),
            n_tp               = int(np.mean([r.n_tp            for r in per_vid])),
        )
        avg_results.append(avg)

    # Print summary table
    print("\n")
    print("  ╔══ ABLATION SUMMARY  (averaged across all videos) ══════════════════╗")
    print(f"  ║  {'Configuration':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ID-Sw/100':>10} {'Reduction':>10}  ║")
    print(f"  ╠{'═'*70}╣")
    for r in avg_results:
        print(f"  ║  {r.config_name:<22} {r.precision:>10.4f} {r.recall:>8.4f} "
              f"{r.f1:>8.4f} {r.id_switches_per100:>10.2f} "
              f"{r.id_reduction_pct:>9.1f}%  ║")
    print(f"  ╚{'═'*70}╝")

    # Save CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=avg_results[0].summary_dict().keys())
        w.writeheader()
        for r in avg_results: w.writerow(r.summary_dict())
    print(f"\n  ✓  Ablation CSV    → {out_csv}")

    # Save report
    lines = [
        "="*72, "RE-ID ABLATION STUDY — AVERAGED RESULTS", "="*72, "",
        f"Videos evaluated: {len(all_results_by_video)}",
        f"  " + ", ".join(all_results_by_video.keys()), "",
        "PROTOCOL",
        "  Simulated re-entry: each track split at random midpoint.",
        "  Second half given new tracker ID → Re-ID must restore original label.",
        "", "AVERAGED RESULTS", "-"*60,
    ]
    for r in avg_results:
        lines.append(str(r))
    lines += ["", "COPY INTO TABLE 4.4 OF DISSERTATION", "-"*40,
              f"{'Config':<26} {'Prec':>8} {'Rec':>8} {'F1':>8} "
              f"{'ID-Sw/100':>10} {'ID-Red%':>8}"]
    for r in avg_results:
        lines.append(f"{r.config_name:<26} {r.precision:>8.4f} {r.recall:>8.4f} "
                     f"{r.f1:>8.4f} {r.id_switches_per100:>10.2f} "
                     f"{r.id_reduction_pct:>7.1f}%")
    with open(out_report, "w") as f: f.write("\n".join(lines))
    print(f"  ✓  Ablation report → {out_report}")
    return True


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def find_videos(folder: str) -> List[str]:
    p = Path(folder)
    if not p.exists():
        return []
    return sorted(str(v) for v in p.iterdir() if v.suffix.lower() in VIDEO_EXTS)


def main():
    parser = argparse.ArgumentParser(
        description="AlphAction+ — statistics evaluation: Re-ID ablation + Thermal FID"
    )
    parser.add_argument("--reid-dir",    default="videos/reid",
                        help="Folder containing Re-ID test videos")
    parser.add_argument("--thermal-dir", default="videos/thermal",
                        help="Folder containing thermal/RGB videos for FID")
    parser.add_argument("--out-dir",     default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--max-frames",  type=int, default=500)
    parser.add_argument("--min-track",   type=int, default=40)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--skip-ablation",  action="store_true")
    parser.add_argument("--skip-thermal",   action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    reid_videos    = find_videos(args.reid_dir)
    thermal_videos = find_videos(args.thermal_dir)

    print("\n" + "="*65)
    print("  AlphAction+  One-Shot Evaluation Suite")
    print("="*65)
    print(f"  Re-ID videos    : {len(reid_videos)}  ({args.reid_dir})")
    print(f"  Thermal videos  : {len(thermal_videos)}  ({args.thermal_dir})")
    print(f"  Output dir      : {out}")
    print("="*65 + "\n")

    if not reid_videos and not thermal_videos:
        print("ERROR: No videos found. Create these folders and put your videos in:")
        print(f"  {args.reid_dir}/     ← for Re-ID ablation and figures")
        print(f"  {args.thermal_dir}/  ← for thermal FID evaluation")
        print("\nThen run:  python run_all.py\n")
        sys.exit(1)

    t_total = time.time()

    # ── 1. Re-ID Ablation ────────────────────────────────────────────────────
    if not args.skip_ablation:
        if reid_videos:
            print("\n[1/2]  RE-ID ABLATION STUDY")
            print("─"*50)
            run_reid_ablation(
                reid_videos,
                out_csv    = str(out/"ablation_results.csv"),
                out_report = str(out/"ablation_report.txt"),
                max_frames = args.max_frames,
                min_track  = args.min_track,
                seed       = args.seed,
            )
        else:
            print("[1/2]  Re-ID ablation — SKIPPED (no videos in reid folder)")
    else:
        print("[1/2]  Re-ID ablation — SKIPPED (--skip-ablation)")

    # ── 2. Thermal FID ───────────────────────────────────────────────────────
    if not args.skip_thermal:
        vids_for_fid = thermal_videos or reid_videos
        if vids_for_fid:
            print("\n[2/2]  THERMAL FID EVALUATION")
            print("─"*50)
            try:
                fid_scores = run_fid_evaluation(vids_for_fid, n_per_video=60)
                if fid_scores:
                    print("\n  Results (lower FID = smaller domain gap):")
                    for m, fid in fid_scores.items():
                        print(f"  {m:<50}  FID = {fid:>8.2f}")

                    fid_csv = str(out/"thermal_fid.csv")
                    with open(fid_csv,"w",newline="") as f:
                        w = csv.writer(f); w.writerow(["Method","FID"])
                        for m,fid in fid_scores.items(): w.writerow([m,fid])
                    print(f"\n  ✓  FID CSV → {fid_csv}")

                    fid_rpt = str(out/"thermal_report.txt")
                    lines   = ["="*70,"THERMAL FID REPORT","="*70,"",
                               "Paste into Table 4.5 of your dissertation","",
                               f"{'Method':<50} {'FID':>8}","-"*60]
                    for m,fid in fid_scores.items():
                        lines.append(f"{m:<50} {fid:>8.2f}")
                    lines += ["","Lower FID = smaller domain gap from RGB training data.",""]
                    with open(fid_rpt,"w") as f: f.write("\n".join(lines))
                    print(f"  ✓  FID report → {fid_rpt}")
            except ImportError as e:
                print(f"  [skip] FID — torch/torchvision not available: {e}")
        else:
            print("[2/2]  Thermal FID — SKIPPED (no videos found)")
    else:
        print("[2/2]  Thermal FID — SKIPPED (--skip-thermal)")

    # ── Done ─────────────────────────────────────────────────────────────────
    elapsed = time.time()-t_total
    print("\n" + "="*65)
    print(f"  All done in {elapsed:.0f}s")
    print(f"  Output folder: {out.resolve()}")
    print()
    print("  Files generated:")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            tag  = f"({size//1024}KB)" if size>1024 else f"({size}B)"
            print(f"    {str(f.relative_to(out.parent)):<55} {tag}")
    print("="*65+"\n")


if __name__ == "__main__":
    main()
