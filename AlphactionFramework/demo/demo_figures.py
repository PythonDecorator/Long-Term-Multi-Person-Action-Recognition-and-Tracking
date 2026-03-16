"""
demo_figures.py  — Automatic Figure Generation for AlphAction+ Demo
====================================================================
Attaches to AVAVisualizer during a demo run and saves dissertation-quality
figures automatically when the video finishes.

HOW TO USE  (in demo.py, after creating video_writer):
-------------------------------------------------------
    from demo_figures import DemoFigureSaver

    figure_saver = DemoFigureSaver(
        video_writer  = video_writer,    # your AVAVisualizer instance
        input_path    = args.input_path,
        output_dir    = "demo_figures",  # where to save figures
        n_reid_frames = 6,               # how many frames in before/after grid
        thermal_mode  = False,           # set True if running thermal video
    )

    # ... run demo as normal ...

    video_writer.close()
    figure_saver.save()                  # call AFTER close()

FIGURES SAVED
-------------
  demo_figures/
    <video_name>_reid_before_after.png   — before/after Re-ID comparison grid
    <video_name>_reid_reentry_strip.png  — person label surviving re-entry
    <video_name>_thermal_pipeline.png    — RGB→Lum→Blur→INFERNO→CLAHE
                                           (only if thermal_mode=True)

Author: Amos Okpe  (MSc Artificial Intelligence and Data Science, University of Hull)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── optional matplotlib (thermal figure only) ─────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — matches person_reid.py exactly
# ─────────────────────────────────────────────────────────────────────────────

_COLORS_BGR = [
    (113, 204,  46), (219, 152,  52), (182,  89, 155),
    ( 15, 196, 241), (156, 188,  26), ( 34, 126, 230),
    ( 60,  76, 231), ( 94,  73,  52), (166, 165, 149),
    (  0, 126, 211), ( 96, 174,  39), (173,  68, 142),
]

def _color(pid: int) -> Tuple[int, int, int]:
    return _COLORS_BGR[(pid - 1) % len(_COLORS_BGR)]


# ─────────────────────────────────────────────────────────────────────────────
# Thermal adapter  (same logic as in main pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class _ThermalAdapter:
    def __init__(self, sigma=2.0, clahe_clip=2.0, clahe_tile=8):
        self.sigma  = sigma
        ksize       = int(6 * sigma + 1) | 1
        self.ksize  = (ksize, ksize)
        self.clahe  = cv2.createCLAHE(clipLimit=clahe_clip,
                                       tileGridSize=(clahe_tile, clahe_tile))

    def stages(self, bgr: np.ndarray):
        """Return (rgb_orig, lum_3ch, blur_3ch, inferno_bgr, clahe_bgr)."""
        lab    = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L      = lab[:, :, 0]
        blur   = cv2.GaussianBlur(L, self.ksize, self.sigma)
        normed = blur.astype(np.float32) / 255.0
        if HAS_MPL:
            rgba   = cm.inferno(normed)
            rgb_inf = (rgba[:, :, :3] * 255).astype(np.uint8)
        else:
            r = np.clip(normed * 2.0, 0, 1)
            g = np.clip(normed * 1.3 - 0.3, 0, 1)
            b = np.zeros_like(normed)
            rgb_inf = (np.stack([r, g, b], 2) * 255).astype(np.uint8)
        inf_bgr = cv2.cvtColor(rgb_inf, cv2.COLOR_RGB2BGR)
        cla_bgr = inf_bgr.copy()
        for c in range(3):
            cla_bgr[:, :, c] = self.clahe.apply(inf_bgr[:, :, c])
        return bgr, cv2.merge([L,L,L]), cv2.merge([blur,blur,blur]), inf_bgr, cla_bgr


# ─────────────────────────────────────────────────────────────────────────────
# DemoFigureSaver
# ─────────────────────────────────────────────────────────────────────────────

class DemoFigureSaver:
    """
    Attach to an AVAVisualizer and save figures when the run completes.

    Records a buffer of (frame_bgr, boxes, pid_map, frame_idx) tuples
    during the run, then at save() time generates the figures.

    Parameters
    ----------
    video_writer  : AVAVisualizer instance (already constructed)
    input_path    : path to the input video (used for naming outputs)
    output_dir    : directory to write figure PNGs into
    n_reid_frames : how many source frames to show in the before/after grid
    thermal_mode  : if True, also generate the thermal pipeline figure
    max_buffer    : max frames to buffer (keeps memory bounded)
    """

    def __init__(
        self,
        video_writer,
        input_path:    str,
        output_dir:    str  = "demo_figures",
        n_reid_frames: int  = 6,
        thermal_mode:  bool = False,
        max_buffer:    int  = 500,
    ):
        self._vw           = video_writer
        self._video_name   = Path(input_path).stem
        self._out_dir      = Path(output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._n_frames     = n_reid_frames
        self._thermal_mode = thermal_mode
        self._max_buffer   = max_buffer
        self._thermal      = _ThermalAdapter()

        # Frame buffer: list of dicts
        # { "bgr": np.ndarray, "boxes": list, "pid_map": dict, "idx": int }
        self._buffer: List[dict] = []
        self._frame_idx = 0

        # Patch _run_reid so we intercept every frame that goes through Re-ID
        self._patch_visualizer()

        print(f"  [DemoFigures] Attached — figures will be saved to: {self._out_dir}/")

    # ── Patch the visualizer ──────────────────────────────────────────────────

    def _patch_visualizer(self) -> None:
        """Monkey-patch _run_reid to record frames + Re-ID state."""
        original_run_reid = self._vw._run_reid
        saver             = self           # capture ref

        def patched_run_reid(frame_bgr, boxes, ids):
            # Run the real Re-ID first
            original_run_reid(frame_bgr, boxes, ids)

            # Then record the result if we still have buffer space
            if len(saver._buffer) < saver._max_buffer:
                from visualizer import _ids_to_list, _boxes_to_list
                tid_list = _ids_to_list(ids)
                box_list = _boxes_to_list(boxes)

                # Build pid_map: tid -> persistent_id
                pid_map: Dict[int, int] = {}
                for tid in tid_list:
                    pid = saver._vw._reid.get_persistent_id(tid)
                    if pid is not None:
                        pid_map[tid] = pid

                # Only buffer frames that have at least one person with a pid
                if pid_map and box_list:
                    saver._buffer.append({
                        "bgr":     frame_bgr.copy(),
                        "boxes":   box_list,
                        "pid_map": pid_map,
                        "tids":    tid_list,
                        "idx":     saver._frame_idx,
                    })

            saver._frame_idx += 1

        self._vw._run_reid = patched_run_reid

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Call this after video_writer.close(). Saves all figures."""
        if not self._buffer:
            print("  [DemoFigures] No frames recorded — nothing to save.")
            return

        print(f"  [DemoFigures] Buffered {len(self._buffer)} frames — generating figures ...")

        self._save_reid_before_after()
        self._save_reid_reentry_strip()
        if self._thermal_mode:
            self._save_thermal_pipeline()

    # ── Figure 1: Before / After Re-ID grid ──────────────────────────────────

    def _save_reid_before_after(self) -> None:
        out_path = self._out_dir / f"{self._video_name}_reid_before_after.png"

        # Pick n evenly-spaced frames from the buffer, prefer multi-person frames
        buf = self._buffer
        rich = [b for b in buf if len(b["pid_map"]) >= 2]
        pool = rich if len(rich) >= self._n_frames else buf
        step = max(1, len(pool) // self._n_frames)
        picks = pool[::step][:self._n_frames]

        if not picks:
            print("  [DemoFigures] Before/after: no suitable frames"); return

        TARGET_W = 680
        rows = []
        for entry in picks:
            bgr  = entry["bgr"]
            H, W = bgr.shape[:2]
            sc   = TARGET_W / W
            sm   = cv2.resize(bgr, (TARGET_W, int(H * sc)))

            def sb(box): return tuple(v * sc for v in box)

            # Pair tids with boxes
            paired = list(zip(entry["tids"], entry["boxes"]))

            before = self._draw_frame(sm, paired, entry["pid_map"], sc, "before",
                                       entry["idx"])
            after  = self._draw_frame(sm, paired, entry["pid_map"], sc, "after",
                                       entry["idx"])
            div    = np.full((before.shape[0], 6, 3), (35, 35, 35), np.uint8)
            rows.append(np.hstack([before, div, after]))

        # Pad to same width
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            dw = max_w - r.shape[1]
            if dw > 0:
                r = np.hstack([r, np.zeros((r.shape[0], dw, 3), np.uint8)])
            padded.append(r)

        # Column header bar
        W_tot  = padded[0].shape[1]
        hdr_h  = 50
        header = np.zeros((hdr_h, W_tot, 3), np.uint8)
        header[:] = (12, 12, 12)
        hw    = (W_tot - 6) // 2
        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.80
        for txt, x_off, col in [
            ("BEFORE Re-ID", 0,      (170, 170, 170)),
            ("AFTER Re-ID",  hw + 6, ( 60, 210,  60)),
        ]:
            (tw, th), _ = cv2.getTextSize(txt, font, scale, 1)
            cv2.putText(header, txt,
                        (x_off + (hw - tw) // 2, (hdr_h + th) // 2),
                        font, scale, col, 1, cv2.LINE_AA)
        # Arrow
        mid_x = hw + 3
        cv2.arrowedLine(header, (mid_x - 18, hdr_h // 2),
                        (mid_x + 18, hdr_h // 2),
                        (120, 120, 120), 2, tipLength=0.5)

        grid = np.vstack([header] + padded)
        cv2.imwrite(str(out_path), grid)
        print(f"  [DemoFigures] ✓  Before/after → {out_path}")

    def _draw_frame(
        self,
        small:   np.ndarray,
        paired:  List[Tuple],   # [(tid, raw_box), ...]
        pid_map: Dict[int, int],
        scale:   float,
        mode:    str,
        fidx:    int,
    ) -> np.ndarray:
        vis   = small.copy()
        H, W  = vis.shape[:2]
        font  = cv2.FONT_HERSHEY_DUPLEX
        fscl  = max(0.50, min(W, H) / 720)
        thick = max(2, int(min(W, H) / 260))

        for tid, raw_box in paired:
            x1, y1, x2, y2 = [int(v * scale) for v in raw_box]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W-1, x2); y2 = min(H-1, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue

            if mode == "before":
                box_col   = (210, 210, 210)
                badge_col = (60, 60, 60)
                label     = f"ID:{tid}"
            else:
                pid       = pid_map.get(tid, 0)
                box_col   = _color(pid)
                badge_col = box_col
                label     = f"Person {pid}"

            # Box + corner accents
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_col, thick + 1)
            c = 16
            for cx, cy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                dx = 1 if cx == x1 else -1
                dy = 1 if cy == y1 else -1
                cv2.line(vis, (cx, cy), (cx + dx*c, cy), box_col, thick + 2)
                cv2.line(vis, (cx, cy), (cx, cy + dy*c), box_col, thick + 2)

            # Badge at bottom of box
            (tw, th), _ = cv2.getTextSize(label, font, fscl, 1)
            pad = 5
            by1 = y2; by2 = min(H-1, y2 + th + pad*2)
            ov  = vis.copy()
            cv2.rectangle(ov, (x1, by1), (x1 + tw + pad*2, by2), badge_col, -1)
            cv2.addWeighted(ov, 0.80, vis, 0.20, 0, vis)
            cv2.putText(vis, label, (x1+pad, by2-pad),
                        font, fscl, (255,255,255), 1, cv2.LINE_AA)

        # Banner
        bh     = max(28, int(H * 0.052))
        banner = np.zeros((bh, W, 3), np.uint8)
        banner[:] = (45, 45, 45) if mode == "before" else (18, 55, 18)
        txt  = ("BEFORE Re-ID  (raw tracker IDs)" if mode == "before"
                else "AFTER Re-ID  (persistent labels)")
        tcol = (180, 180, 180) if mode == "before" else (70, 220, 70)
        (tw2, th2), _ = cv2.getTextSize(txt, font, fscl * 0.82, 1)
        cv2.putText(banner, txt, ((W-tw2)//2, (bh+th2)//2),
                    font, fscl * 0.82, tcol, 1, cv2.LINE_AA)
        cv2.putText(vis, f"f{fidx}", (5, H-6),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (150,150,150), 1)
        return np.vstack([banner, vis])

    # ── Figure 2: Re-entry strip ──────────────────────────────────────────────

    def _save_reid_reentry_strip(self) -> None:
        out_path = self._out_dir / f"{self._video_name}_reid_reentry_strip.png"
        buf      = self._buffer
        if len(buf) < 8:
            print("  [DemoFigures] Re-entry strip: too few frames"); return

        # Find a person who appears consistently across the whole buffer
        # pid -> list of (buffer_idx, box)
        pid_appearances: Dict[int, List] = {}
        for bi, entry in enumerate(buf):
            for tid, pid in entry["pid_map"].items():
                # find matching box
                for t, b in zip(entry["tids"], entry["boxes"]):
                    if t == tid:
                        pid_appearances.setdefault(pid, []).append((bi, b, entry))
                        break

        # Pick the person with the most appearances
        if not pid_appearances:
            print("  [DemoFigures] Re-entry strip: no persistent persons found"); return

        best_pid      = max(pid_appearances, key=lambda p: len(pid_appearances[p]))
        appearances   = pid_appearances[best_pid]

        if len(appearances) < 6:
            print("  [DemoFigures] Re-entry strip: best person has too few frames"); return

        # Sample 4 frames from first half, 4 from second half with a gap in between
        mid   = len(appearances) // 2
        n_ea  = 4
        step1 = max(1, mid // n_ea)
        step2 = max(1, (len(appearances) - mid) // n_ea)
        first_half  = appearances[:mid:step1][:n_ea]
        second_half = appearances[mid::step2][:n_ea]

        # Check if there's a genuine temporal gap (re-entry evidence)
        gap_frames = 0
        if first_half and second_half:
            last_first  = first_half[-1][2]["idx"]
            first_second = second_half[0][2]["idx"]
            gap_frames   = first_second - last_first

        PANEL_W = 270
        panels  = []
        font    = cv2.FONT_HERSHEY_DUPLEX
        color   = _color(best_pid)

        def make_panel(bi, raw_box, entry, half_label):
            bgr   = entry["bgr"]
            H, W  = bgr.shape[:2]
            sc    = PANEL_W / W
            small = cv2.resize(bgr, (PANEL_W, int(H * sc)))
            fscl  = 0.55

            x1,y1,x2,y2 = [int(v * sc) for v in raw_box]
            x1=max(0,x1); y1=max(0,y1)
            x2=min(small.shape[1]-1,x2); y2=min(small.shape[0]-1,y2)
            if x2-x1 > 5 and y2-y1 > 5:
                cv2.rectangle(small,(x1,y1),(x2,y2),color,2)
                c=12
                for cx,cy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                    dx=1 if cx==x1 else -1; dy=1 if cy==y1 else -1
                    cv2.line(small,(cx,cy),(cx+dx*c,cy),color,2)
                    cv2.line(small,(cx,cy),(cx,cy+dy*c),color,2)

            # Person label badge
            lbl = f"Person {best_pid}"
            (lw,lh),_ = cv2.getTextSize(lbl, font, fscl, 1)
            pad = 4
            ly2 = min(small.shape[0]-2, y2 + lh + pad*2)
            ly1 = y2
            ov = small.copy()
            cv2.rectangle(ov,(x1,ly1),(x1+lw+pad*2,ly2),color,-1)
            cv2.addWeighted(ov,0.8,small,0.2,0,small)
            cv2.putText(small,lbl,(x1+pad,ly2-pad),font,fscl,(255,255,255),1,cv2.LINE_AA)

            # Half label top-left
            hcol = (180,180,180) if "1st" in half_label else (50,200,255)
            cv2.putText(small,half_label,(4,16),
                        cv2.FONT_HERSHEY_PLAIN,0.85,hcol,1,cv2.LINE_AA)
            cv2.putText(small,f"f{entry['idx']}",(4,small.shape[0]-6),
                        cv2.FONT_HERSHEY_PLAIN,0.75,(130,130,130),1)
            return small

        for bi, raw_box, entry in first_half:
            panels.append(make_panel(bi, raw_box, entry, "1st half"))

        # Re-entry divider
        if panels:
            ph = panels[0].shape[0]
            div = np.zeros((ph, 48, 3), np.uint8)
            div[:] = (0, 100, 200)
            gap_txt = f"+{gap_frames}f" if gap_frames > 0 else "RE"
            for yi, txt in enumerate(["RE", "EN", "TRY", gap_txt]):
                cv2.putText(div, txt, (4, ph//2 - 30 + yi*20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45,
                            (255, 220, 50), 1, cv2.LINE_AA)
            panels.append(div)

        for bi, raw_box, entry in second_half:
            panels.append(make_panel(bi, raw_box, entry, "2nd half"))

        # Pad all to same height
        max_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            dh = max_h - p.shape[0]
            if dh > 0:
                p = np.vstack([p, np.zeros((dh, p.shape[1], 3), np.uint8)])
            padded.append(p)
        strip = np.hstack(padded)

        # Title bar
        th = 46
        title_bar = np.zeros((th, strip.shape[1], 3), np.uint8)
        title_bar[:] = (10, 10, 10)
        n_appear  = len(appearances)
        gap_note  = f"  |  {gap_frames}-frame gap at re-entry" if gap_frames > 3 else ""
        title_txt = (f"Person {best_pid}: label maintained across {n_appear} appearances"
                     f"  |  blue divider = temporal gap / re-entry point{gap_note}"
                     f"  |  '{self._video_name}'")
        (ttw,tth),_ = cv2.getTextSize(title_txt, font, 0.50, 1)
        cv2.putText(title_bar, title_txt,
                    ((strip.shape[1]-ttw)//2, (th+tth)//2),
                    font, 0.50, color, 1, cv2.LINE_AA)
        final = np.vstack([title_bar, strip])
        cv2.imwrite(str(out_path), final)
        print(f"  [DemoFigures] ✓  Re-entry strip → {out_path}")

    # ── Figure 3: Thermal pipeline ────────────────────────────────────────────

    def _save_thermal_pipeline(self) -> None:
        if not HAS_MPL:
            print("  [DemoFigures] Thermal figure skipped — matplotlib not installed")
            return

        out_path = self._out_dir / f"{self._video_name}_thermal_pipeline.png"
        buf      = self._buffer
        n_rows   = 3

        # Pick evenly spaced frames with people in them
        step  = max(1, len(buf) // n_rows)
        picks = buf[::step][:n_rows]

        stage_labels = [
            "Input Frame",
            "Luminance (L*)",
            "Gaussian Blur\n(σ=2.0)",
            "INFERNO\nFalse-colour",
            "INFERNO + CLAHE\n(ThermalAdapter)",
        ]
        stage_colors = ["#4CAF50","#2196F3","#FF9800","#E91E63","#9C27B0"]
        n_cols = 5

        fig_w = 5 * n_cols
        fig_h = 3.8 * len(picks) + 1.2
        fig   = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d0d0d")
        fig.suptitle(
            f"Thermal Adaptation Pipeline — '{self._video_name}'",
            fontsize=15, color="white", fontweight="bold", y=0.995,
        )
        gs = gridspec.GridSpec(
            len(picks), n_cols, figure=fig,
            hspace=0.06, wspace=0.04,
            left=0.01, right=0.99, top=0.94, bottom=0.04,
        )
        for ri, entry in enumerate(picks):
            bgr    = entry["bgr"]
            stages = self._thermal.stages(bgr)
            for ci, (stage_bgr, lbl, scol) in enumerate(
                zip(stages, stage_labels, stage_colors)
            ):
                ax  = fig.add_subplot(gs[ri, ci])
                rgb = cv2.cvtColor(stage_bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(scol); sp.set_linewidth(2.5)
                if ri == 0:
                    ax.set_title(lbl, fontsize=10, color=scol,
                                 fontweight="bold", pad=5, wrap=True)
                if ci == 0:
                    ax.set_ylabel(f"Frame {ri+1}", fontsize=8,
                                  color="#aaaaaa", rotation=90, labelpad=3)

        self._out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  [DemoFigures] ✓  Thermal pipeline → {out_path}")
