"""
reid_evaluator.py  (v2 — fixed)
================================
Evaluation harness for PersonReIdentifier ablation study.

Simulated Re-Entry Protocol
----------------------------
Each track is split at a random midpoint.
  First half  → keeps original tracker ID
  Second half → gets a brand-new tracker ID (simulates re-entry)

Re-ID is CORRECT if both halves end up with the SAME persistent label.
  TP = re-entry correctly restored
  FP = re-entry got wrong / new persistent ID
  FN = re-entry with missing first or second half data

Per-frame before/after display is printed after each config run.

Author: Amos Okpe (MSc Computer Science, University of Hull)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackSegment:
    track_id:  int
    frame_ids: List[int]                            = field(default_factory=list)
    boxes:     List[Tuple[float,float,float,float]] = field(default_factory=list)
    def __len__(self): return len(self.frame_ids)


@dataclass
class EvalResults:
    config_name:        str
    precision:          float = 0.0
    recall:             float = 0.0
    f1:                 float = 0.0
    id_switches:        int   = 0
    total_frames:       int   = 0
    id_switches_per100: float = 0.0
    consistency:        float = 0.0
    total_raw_ids:      int   = 0
    total_persistent:   int   = 0
    id_reduction_pct:   float = 0.0
    ids_per_frame_raw:  float = 0.0
    ids_per_frame_reid: float = 0.0
    frag_rate:          float = 0.0
    n_reentry_events:   int   = 0
    n_tp:               int   = 0
    n_fp:               int   = 0
    n_fn:               int   = 0
    frame_snapshots:    List[Tuple[int,int,int]] = field(default_factory=list)

    def summary_dict(self) -> dict:
        return {
            "Config":           self.config_name,
            "Precision":        round(self.precision, 4),
            "Recall":           round(self.recall, 4),
            "F1":               round(self.f1, 4),
            "ID-Sw/100fr":      round(self.id_switches_per100, 2),
            "Consistency":      round(self.consistency, 4),
            "IDs/fr (raw)":     round(self.ids_per_frame_raw, 2),
            "IDs/fr (ReID)":    round(self.ids_per_frame_reid, 2),
            "Total raw IDs":    self.total_raw_ids,
            "Total persistent": self.total_persistent,
            "ID reduction %":   round(self.id_reduction_pct, 1),
            "Frag. rate":       round(self.frag_rate, 3),
        }

    def __str__(self):
        return (
            f"[{self.config_name}]\n"
            f"  Precision={self.precision:.4f}  Recall={self.recall:.4f}  F1={self.f1:.4f}\n"
            f"  ID-Sw/100fr={self.id_switches_per100:.2f}  Consistency={self.consistency:.4f}\n"
            f"  Re-entry events={self.n_reentry_events}  TP={self.n_tp}  FP={self.n_fp}  FN={self.n_fn}\n"
            f"  Raw IDs={self.total_raw_ids}  Persistent={self.total_persistent}  "
            f"Reduction={self.id_reduction_pct:.1f}%  Frag={self.frag_rate:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Track extraction
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    ax1,ay1,ax2,ay2 = a;  bx1,by1,bx2,by2 = b
    ix1=max(ax1,bx1); iy1=max(ay1,by1)
    ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter = max(0.0,ix2-ix1)*max(0.0,iy2-iy1)
    ua    = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/ua if ua>0 else 0.0


def _merge_boxes(boxes, thresh=0.25):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]), reverse=True)
    used  = [False]*len(boxes)
    out   = []
    for i,b in enumerate(boxes):
        if used[i]: continue
        grp = [b]
        for j in range(i+1,len(boxes)):
            if not used[j] and _iou(b,boxes[j])>thresh:
                grp.append(boxes[j]); used[j]=True
        out.append((min(g[0] for g in grp), min(g[1] for g in grp),
                    max(g[2] for g in grp), max(g[3] for g in grp)))
    return out


def extract_tracks_from_video(
    video_path: str,
    min_track_frames: int = 25,
    max_frames: int = 400,
) -> Tuple[List[np.ndarray], List[TrackSegment]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fgbg     = cv2.createBackgroundSubtractorMOG2(200, 50, False)
    min_area = 0.004 * W * H
    min_h    = 0.10  * H
    max_age  = 8

    active: Dict[int,dict]         = {}
    store:  Dict[int,TrackSegment] = {}
    next_tid = 1
    frames: List[np.ndarray] = []

    for frame_idx in range(max_frames):
        ok, frame = cap.read()
        if not ok: break
        frames.append(frame.copy())

        fg = fgbg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        fg = cv2.dilate(fg, np.ones((15,15),np.uint8), iterations=2)

        cnts,_ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands  = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w*h < min_area: continue
            if h   < min_h:    continue
            if h   < w:        continue
            cands.append((float(x),float(y),float(x+w),float(y+h)))
        cands = _merge_boxes(cands)

        used_c = set()
        for tid, ts in active.items():
            best_iou, best_ci = 0.0, -1
            for ci,cb in enumerate(cands):
                if ci in used_c: continue
                v = _iou(ts["box"], cb)
                if v > best_iou: best_iou,best_ci = v,ci
            if best_iou > 0.15 and best_ci >= 0:
                cb = cands[best_ci]
                ts["box"] = cb; ts["last"] = frame_idx
                store[tid].frame_ids.append(frame_idx)
                store[tid].boxes.append(cb)
                used_c.add(best_ci)

        for ci,cb in enumerate(cands):
            if ci not in used_c:
                tid = next_tid; next_tid += 1
                active[tid] = {"box":cb,"last":frame_idx}
                store[tid]  = TrackSegment(track_id=tid)
                store[tid].frame_ids.append(frame_idx)
                store[tid].boxes.append(cb)

        stale = [t for t,ts in active.items() if frame_idx-ts["last"] > max_age]
        for t in stale: del active[t]

    cap.release()
    valid = [t for t in store.values() if len(t) >= min_track_frames]
    return frames, valid


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_config(
    frames:      List[np.ndarray],
    tracks:      List[TrackSegment],
    reid,
    config_name: str,
    seed:        int = 42,
) -> EvalResults:
    rng = random.Random(seed)
    reid.reset()

    if not tracks:
        return EvalResults(config_name=config_name)

    # ── Step 1: assign split points ────────────────────────────────────────
    max_existing  = max(t.track_id for t in tracks)
    new_id_start  = max_existing + 100

    # split_info[orig_tid] = (split_index, new_tid_for_second_half)
    split_info: Dict[int, Tuple[int,int]] = {}
    counter = 0
    for t in tracks:
        if len(t) < 4:
            continue
        lo  = max(1, len(t)//3)
        hi  = min(len(t)-1, 2*len(t)//3)
        mid = rng.randint(lo, hi)
        split_info[t.track_id] = (mid, new_id_start + counter)
        counter += 1

    # ── Step 2: build per-frame lookup ─────────────────────────────────────
    # frame_lookup[fid] = list of (assigned_tid, box, orig_tid, is_second)
    frame_lookup: Dict[int, List] = {}
    for t in tracks:
        if t.track_id not in split_info:
            mid, new_tid = len(t), -1
        else:
            mid, new_tid = split_info[t.track_id]
        for i, (fid, box) in enumerate(zip(t.frame_ids, t.boxes)):
            if fid >= len(frames):
                continue
            is_second  = (i >= mid) and (new_tid != -1)
            assigned   = new_tid if is_second else t.track_id
            frame_lookup.setdefault(fid, []).append(
                (assigned, box, t.track_id, is_second)
            )

    # ── Step 3: run Re-ID frame by frame ──────────────────────────────────
    first_half_pids:  Dict[int, List[int]] = {}
    second_half_pids: Dict[int, List[int]] = {}
    snapshots: List[Tuple[int,int,int]] = []
    prev_pid: Dict[int,int] = {}
    id_switches = 0
    raw_ids_per_frame:  List[int] = []
    reid_ids_per_frame: List[int] = []
    all_raw_tids:  set = set()
    all_reid_pids: set = set()

    for fid in sorted(frame_lookup.keys()):
        entries   = frame_lookup[fid]
        bgr       = frames[fid]
        tids      = [e[0] for e in entries]
        boxes     = [e[1] for e in entries]
        orig_tids = [e[2] for e in entries]
        is_second = [e[3] for e in entries]

        all_raw_tids.update(tids)
        raw_ids_per_frame.append(len(set(tids)))

        reid.update(bgr, tids, boxes)

        frame_pids: set = set()
        for tid, orig_tid, second in zip(tids, orig_tids, is_second):
            pid = reid.get_persistent_id(tid)
            if pid is None:
                continue
            frame_pids.add(pid)
            all_reid_pids.add(pid)

            if second:
                second_half_pids.setdefault(orig_tid, []).append(pid)
            else:
                first_half_pids.setdefault(orig_tid, []).append(pid)
                if tid in prev_pid and prev_pid[tid] != pid:
                    id_switches += 1
                prev_pid[tid] = pid

        reid_ids_per_frame.append(len(frame_pids))
        snapshots.append((fid, len(set(tids)), len(frame_pids)))

    # ── Step 4: precision / recall ────────────────────────────────────────
    tp = fp = fn = 0
    reentry_events = 0
    for orig_tid in split_info:
        reentry_events += 1
        fh = first_half_pids.get(orig_tid, [])
        sh = second_half_pids.get(orig_tid, [])
        if not fh or not sh:
            fn += 1
            continue
        dom_fh = max(set(fh), key=fh.count)
        dom_sh = max(set(sh), key=sh.count)
        if dom_fh == dom_sh:
            tp += 1
        else:
            fp += 1
    fn = max(0, reentry_events - tp - fp)

    precision = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    recall    = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1        = (2*precision*recall/(precision+recall)
                 if (precision+recall) > 0 else 0.0)

    total_fr  = len(snapshots)
    sw_per100 = id_switches/total_fr*100 if total_fr > 0 else 0.0

    consistencies = []
    for orig_tid in split_info:
        pids = first_half_pids.get(orig_tid, [])
        if not pids: continue
        dom = max(set(pids), key=pids.count)
        consistencies.append(pids.count(dom)/len(pids))
    consistency = float(np.mean(consistencies)) if consistencies else 0.0

    total_raw  = len(all_raw_tids)
    total_reid = len(all_reid_pids)
    id_red_pct = (total_raw-total_reid)/total_raw*100 if total_raw > 0 else 0.0
    frag_rate  = total_raw/total_reid if total_reid > 0 else 1.0

    return EvalResults(
        config_name        = config_name,
        precision          = precision,
        recall             = recall,
        f1                 = f1,
        id_switches        = id_switches,
        total_frames       = total_fr,
        id_switches_per100 = sw_per100,
        consistency        = consistency,
        total_raw_ids      = total_raw,
        total_persistent   = total_reid,
        id_reduction_pct   = id_red_pct,
        ids_per_frame_raw  = float(np.mean(raw_ids_per_frame))  if raw_ids_per_frame  else 0.0,
        ids_per_frame_reid = float(np.mean(reid_ids_per_frame)) if reid_ids_per_frame else 0.0,
        frag_rate          = frag_rate,
        n_reentry_events   = reentry_events,
        n_tp               = tp,
        n_fp               = fp,
        n_fn               = fn,
        frame_snapshots    = snapshots,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Before / after display  (called from run_ablation.py)
# ─────────────────────────────────────────────────────────────────────────────

def print_before_after_summary(result: EvalResults, config_name: str) -> None:
    snaps = result.frame_snapshots
    if not snaps:
        print(f"  [{config_name}] No frame data.")
        return

    raw_vals  = [s[1] for s in snaps]
    reid_vals = [s[2] for s in snaps]

    print()
    print(f"  ╔══ BEFORE vs AFTER Re-ID  [{config_name}] ═══════════════════════════════╗")
    print(f"  ║  {'Frame':>6}  │  {'Before Re-ID':^28}  │  {'After Re-ID':^28}  ║")
    print(f"  ║  {'':>6}  │  {'(raw tracker IDs)':^28}  │  {'(persistent person labels)':^28}  ║")
    print(f"  ╠════════╪══════════════════════════════╪══════════════════════════════╣")

    step = max(1, len(snaps)//20)
    for fid, n_raw, n_reid in snaps[::step][:20]:
        bar_r = ("█" * min(n_raw,  24)).ljust(24) + ("…" if n_raw  > 24 else " ")
        bar_p = ("█" * min(n_reid, 24)).ljust(24) + ("…" if n_reid > 24 else " ")
        print(f"  ║  {fid:>6}  │  {n_raw:>2}  {bar_r}  │  {n_reid:>2}  {bar_p}  ║")

    print(f"  ╠════════╧══════════════════════════════╧══════════════════════════════╣")
    print(f"  ║  {'AGGREGATE STATISTICS':^62}  ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════╣")
    print(f"  ║  {'Metric':<38}  {'Before':>10}  {'After':>10}  ║")
    print(f"  ║  {'─'*38}  {'─'*10}  {'─'*10}  ║")
    print(f"  ║  {'Mean persons / frame':<38}  {np.mean(raw_vals):>10.2f}  {np.mean(reid_vals):>10.2f}  ║")
    print(f"  ║  {'Max persons in any frame':<38}  {max(raw_vals):>10}  {max(reid_vals):>10}  ║")
    print(f"  ║  {'Frames with ≥ 2 persons':<38}  {sum(1 for v in raw_vals if v>=2):>10}  {sum(1 for v in reid_vals if v>=2):>10}  ║")
    print(f"  ║  {'Total unique IDs across video':<38}  {result.total_raw_ids:>10}  {result.total_persistent:>10}  ║")
    print(f"  ║  {'ID reduction (re-entries merged)':<38}  {'—':>10}  {result.id_reduction_pct:>9.1f}%  ║")
    print(f"  ║  {'Re-entry events detected':<38}  {result.n_reentry_events:>10}  {'—':>10}  ║")
    print(f"  ║  {'Correct re-identifications (TP)':<38}  {'—':>10}  {result.n_tp:>10}  ║")
    print(f"  ║  {'Re-entry precision':<38}  {'—':>10}  {result.precision:>10.4f}  ║")
    print(f"  ║  {'Re-entry recall':<38}  {'—':>10}  {result.recall:>10.4f}  ║")
    print(f"  ║  {'Re-entry F1':<38}  {'—':>10}  {result.f1:>10.4f}  ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════════╝")
    print()
