"""
reid_evaluator.py — Simulated Re-Entry Evaluation Engine
=========================================================

Evaluates Re-ID quality WITHOUT manual ground-truth annotation by using
a "Simulated Re-Entry Protocol":

  1. Run the JDE tracker on the video to extract stable per-person tracks.
     Within a short clip, tracker_id == person_identity (before drift).

  2. For every track longer than MIN_TRACK_FRAMES, split it at a random
     midpoint and assign the second segment a BRAND NEW tracker_id.
     This faithfully simulates the person leaving and re-entering frame.

  3. Feed this modified sequence to PersonReIdentifier. Since the first
     segment established a gallery entry, the Re-ID should recognise the
     second segment and re-assign the SAME persistent_id.

  4. Count outcomes per split event:
       TP — second segment gets the SAME persistent_id as the first
       FP — second segment gets the persistent_id of a DIFFERENT person
       FN — second segment gets a BRAND NEW persistent_id (fragmentation)

  5. Compute Precision, Recall, F1, ID-Switch rate, Consistency score.

Usage
-----
  from reid_evaluator import evaluate_config
  results = evaluate_config(video_path, reid_config, tracker_data=None)
  print(results)

Or use run_ablation.py to evaluate all 5 configurations automatically.

Author : Amos Okpe  (MSc Computer Science, University of Hull)
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from person_reid import PersonReIdentifier


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackSegment:
    """One person's detections over a contiguous range of frames."""
    track_id:  int
    frames:    List[int]                                    # frame indices
    boxes:     List[Tuple[float, float, float, float]]      # (x1,y1,x2,y2)


@dataclass
class EvalResults:
    """Metrics for one configuration."""
    config_name:      str
    tp:               int   = 0
    fp:               int   = 0
    fn:               int   = 0
    id_switches:      int   = 0
    total_frames:     int   = 0
    n_events:         int   = 0   # number of simulated re-entry events
    consistency_scores: List[float] = field(default_factory=list)

    # ── Fragmentation / person-count metrics ─────────────────────────────
    # Per-frame unique ID counts — appended during evaluate_config main loop
    _raw_ids_per_frame:  List[int] = field(default_factory=list)  # unique tracker IDs/frame
    _reid_ids_per_frame: List[int] = field(default_factory=list)  # unique persistent IDs/frame
    total_raw_ids:        int = 0   # unique tracker IDs seen across whole clip
    total_persistent_ids: int = 0   # unique persistent IDs seen across whole clip

    @property
    def precision(self) -> float:
        return self.tp / max(1, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self.tp / max(1, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(1e-9, p + r)

    @property
    def id_switch_rate(self) -> float:
        """ID switches per 100 frames (normalised)."""
        return 100.0 * self.id_switches / max(1, self.total_frames)

    @property
    def consistency(self) -> float:
        """Mean track consistency: fraction of frames with the dominant label."""
        if not self.consistency_scores:
            return 0.0
        return float(np.mean(self.consistency_scores))

    # ── Fragmentation properties ──────────────────────────────────────────

    @property
    def mean_raw_ids_per_frame(self) -> float:
        """Average number of unique tracker IDs visible per frame (before Re-ID)."""
        if not self._raw_ids_per_frame:
            return 0.0
        return float(np.mean(self._raw_ids_per_frame))

    @property
    def mean_reid_ids_per_frame(self) -> float:
        """Average number of unique persistent IDs visible per frame (after Re-ID)."""
        if not self._reid_ids_per_frame:
            return 0.0
        return float(np.mean(self._reid_ids_per_frame))

    @property
    def id_reduction_pct(self) -> float:
        """
        Percentage reduction in total unique IDs after Re-ID.

        A value of 30% means Re-ID collapsed 30% of tracker IDs back into
        existing persons rather than creating new identities.
        Higher is better — it means fewer spurious new IDs were created.
        A value of 0% means Re-ID added no consolidation at all.
        """
        raw = self.total_raw_ids
        if raw == 0:
            return 0.0
        return 100.0 * (raw - self.total_persistent_ids) / raw

    @property
    def fragmentation_rate(self) -> float:
        """
        Ratio of raw tracker IDs to persistent IDs (across whole clip).

        1.0 = perfect (every re-entry correctly re-identified, no new IDs)
        2.0 = every person generated one extra spurious identity on re-entry
        Higher is worse.
        """
        return self.total_raw_ids / max(1, self.total_persistent_ids)

    def summary_dict(self) -> dict:
        return {
            "Config":              self.config_name,
            "Precision":           round(self.precision,             4),
            "Recall":              round(self.recall,                4),
            "F1":                  round(self.f1,                    4),
            "ID-Sw/100fr":         round(self.id_switch_rate,        2),
            "Consistency":         round(self.consistency,           4),
            # ── Fragmentation columns ──
            "IDs/fr (raw)":        round(self.mean_raw_ids_per_frame,  2),
            "IDs/fr (ReID)":       round(self.mean_reid_ids_per_frame, 2),
            "Total raw IDs":       self.total_raw_ids,
            "Total persistent":    self.total_persistent_ids,
            "ID reduction %":      round(self.id_reduction_pct,     1),
            "Frag. rate":          round(self.fragmentation_rate,    3),
            # ── Event counts ──
            "Events":              self.n_events,
            "TP":                  self.tp,
            "FP":                  self.fp,
            "FN":                  self.fn,
        }

    def __str__(self) -> str:
        d = self.summary_dict()
        return (
            f"[{d['Config']:20s}]  "
            f"P={d['Precision']:.4f}  R={d['Recall']:.4f}  F1={d['F1']:.4f}  "
            f"ID-Sw/100fr={d['ID-Sw/100fr']:.2f}  Consistency={d['Consistency']:.4f}  "
            f"IDs/fr: {d['IDs/fr (raw)']:.2f}→{d['IDs/fr (ReID)']:.2f}  "
            f"ID-reduction={d['ID reduction %']:.1f}%  "
            f"FragRate={d['Frag. rate']:.3f}  "
            f"Events={d['Events']}  TP={d['TP']} FP={d['FP']} FN={d['FN']}"
        )


# ---------------------------------------------------------------------------
# Tracker data extraction
# ---------------------------------------------------------------------------

def extract_tracks_from_video(
    video_path:       str,
    min_track_frames: int   = 30,
    max_frames:       int   = 500,
    nms_iou_thresh:   float = 0.45,
    conf_thresh:      float = 0.30,
) -> Tuple[List[np.ndarray], Dict[int, TrackSegment]]:
    """
    Extract approximate person tracks from video using background subtraction
    and IoU-based centroid tracking.

    Uses ONLY core OpenCV APIs (no cv2.legacy, no cv2.MultiTracker) so it
    works on any OpenCV version including 4.2.x on older HPC environments.

    This is used ONLY for the evaluation harness — production uses the JDE
    tracker. Here we just need stable enough tracks to split for re-entry.

    Returns
    -------
    frames : list of BGR frames (np.ndarray)
    tracks : {track_id: TrackSegment}  (only tracks >= min_track_frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: {}".format(video_path))

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=40, detectShadows=False
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    frames:     List[np.ndarray]         = []
    raw_tracks: Dict[int, TrackSegment]  = {}
    # active_boxes: {track_id: (x1,y1,x2,y2)} — last known box per track
    active_boxes: Dict[int, Tuple[float,float,float,float]] = {}
    next_tid = 1

    def _iou(a, b):
        """IoU between two (x1,y1,x2,y2) boxes."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / max(1.0, ua)

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())
        fh, fw = frame.shape[:2]

        fg_mask = bg_sub.apply(frame)

        # Let background model settle for first 20 frames
        if frame_idx < 20:
            frame_idx += 1
            continue

        # Morphological clean-up
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_CLOSE, kernel)

        # Find foreground blobs that look person-sized
        contours, _ = cv2.findContours(
            fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area_frac = (w * h) / float(fw * fh)
            if area_frac < 0.002 or area_frac > 0.45:
                continue
            if h < 0.05 * fh:
                continue
            if h / float(max(1, w)) < 0.4:
                continue
            x2, y2 = x + w, y + h
            detections.append((float(x), float(y), float(x2), float(y2)))

        # ── IoU-based matching (greedy, no cv2.legacy needed) ────────────
        matched_tids = set()
        matched_dets = set()

        # Build IoU matrix between active tracks and new detections
        tid_list = list(active_boxes.keys())
        scores = []   # (iou, tid, det_idx)
        for tid in tid_list:
            for di, det in enumerate(detections):
                iou = _iou(active_boxes[tid], det)
                if iou > 0.20:
                    scores.append((iou, tid, di))
        scores.sort(reverse=True)

        for iou_val, tid, di in scores:
            if tid in matched_tids or di in matched_dets:
                continue
            # Matched — update track with new box
            box = detections[di]
            raw_tracks[tid].frames.append(frame_idx)
            raw_tracks[tid].boxes.append(box)
            active_boxes[tid] = box
            matched_tids.add(tid)
            matched_dets.add(di)

        # Unmatched detections → new tracks
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            raw_tracks[next_tid] = TrackSegment(
                track_id=next_tid,
                frames=[frame_idx],
                boxes=[det],
            )
            active_boxes[next_tid] = det
            next_tid += 1

        # Remove tracks with no detection for > 10 frames
        # (they've left the scene — freeze them so they don't accumulate noise)
        stale = []
        for tid in list(active_boxes.keys()):
            if tid not in matched_tids:
                last = raw_tracks[tid].frames[-1] if raw_tracks[tid].frames else 0
                if frame_idx - last > 10:
                    stale.append(tid)
        for tid in stale:
            del active_boxes[tid]

        frame_idx += 1

    cap.release()

    if not frames:
        raise ValueError("No frames could be read from: {}".format(video_path))

    # Filter to tracks long enough to split
    long_tracks = {
        k: v for k, v in raw_tracks.items()
        if len(v.frames) >= min_track_frames
    }

    if not long_tracks:
        # Fallback: if background subtraction found nothing (static scene,
        # uniform background) return one synthetic full-frame track so the
        # evaluation can still run — results will be trivial but won't crash.
        print("  WARNING: No person-sized motion detected. "
              "Using synthetic full-frame track as fallback.")
        h, w = frames[0].shape[:2]
        synthetic = TrackSegment(
            track_id=1,
            frames=list(range(len(frames))),
            boxes=[(int(w*0.2), int(h*0.05), int(w*0.8), int(h*0.95))] * len(frames),
        )
        return frames, {1: synthetic}

    print("  Found {} candidate tracks, {} meet min {} frames.".format(
        len(raw_tracks), len(long_tracks), min_track_frames))
    return frames, long_tracks


# ---------------------------------------------------------------------------
# Simulated Re-Entry Protocol
# ---------------------------------------------------------------------------

def simulate_reentry_sequence(
    tracks:          Dict[int, TrackSegment],
    gap_min_frames:  int = 5,
    gap_max_frames:  int = 30,
    seed:            int = 42,
) -> Tuple[
    List[Tuple[int, int, Tuple[float, float, float, float]]],  # sequence
    Dict[int, int],   # new_tid → original_tid  (ground truth)
    Dict[int, int],   # original_tid → persistent_id (first-segment pid)
]:
    """
    Split each track at a random midpoint. The second half gets a new
    tracker_id. Returns the interleaved sequence plus ground-truth mappings.

    Returns
    -------
    sequence : list of (frame_idx, tracker_id, box) in frame order
    gt_map   : {new_tracker_id: original_tracker_id} — ground truth
    split_points : {original_tid: split_frame_idx}
    """
    rng = random.Random(seed)

    sequence:      List[Tuple[int, int, Tuple[float,float,float,float]]] = []
    gt_map:        Dict[int, int] = {}
    split_at:      Dict[int, int] = {}

    next_new_tid = max(tracks.keys()) + 100   # offset to avoid collision

    for tid, seg in tracks.items():
        n = len(seg.frames)
        if n < 2:
            for fi, box in zip(seg.frames, seg.boxes):
                sequence.append((fi, tid, box))
            continue

        # Split point: between 30% and 70% of the track
        lo = max(1, int(n * 0.30))
        hi = min(n - 1, int(n * 0.70))
        split = rng.randint(lo, hi)

        # First half — original tracker_id
        for fi, box in zip(seg.frames[:split], seg.boxes[:split]):
            sequence.append((fi, tid, box))

        # Simulate a gap (person off-screen)
        gap = rng.randint(gap_min_frames, gap_max_frames)
        # We just skip frames — no detections during the gap

        # Second half — NEW tracker_id
        new_tid = next_new_tid
        next_new_tid += 1
        gt_map[new_tid] = tid
        split_at[tid] = split

        second_start = split + gap
        for i, (fi, box) in enumerate(zip(seg.frames[split:], seg.boxes[split:])):
            sim_fi = fi + gap   # offset so frames don't overlap
            sequence.append((sim_fi, new_tid, box))

    # Sort by frame index
    sequence.sort(key=lambda x: x[0])

    return sequence, gt_map, split_at


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_config(
    frames:       List[np.ndarray],
    tracks:       Dict[int, TrackSegment],
    reid:         PersonReIdentifier,
    config_name:  str,
    seed:         int = 42,
) -> EvalResults:
    """
    Run one Re-ID configuration against the simulated re-entry protocol.

    Parameters
    ----------
    frames      : video frames as BGR numpy arrays
    tracks      : extracted tracks (from extract_tracks_from_video)
    reid        : a freshly reset PersonReIdentifier instance
    config_name : label for results table
    seed        : random seed for reproducible splits

    Returns
    -------
    EvalResults with all metrics filled in.
    """
    reid.reset()
    results = EvalResults(config_name=config_name)

    sequence, gt_map, _ = simulate_reentry_sequence(tracks, seed=seed)

    if not sequence:
        print(f"  WARNING: No sequence generated for {config_name}")
        return results

    # --- Run Re-ID on the sequence ----------------------------------------
    # Group events by frame to call reid.update() once per frame
    frame_events: Dict[int, List[Tuple[int, Tuple]]] = defaultdict(list)
    for (fi, tid, box) in sequence:
        frame_events[fi].append((tid, box))

    # Track what persistent_id was assigned to each original tracker_id
    # during its FIRST segment (ground truth anchor)
    first_seg_pid:    Dict[int, int]        = {}
    second_seg_pids:  Dict[int, List[int]]  = defaultdict(list)
    tid_all_pids:     Dict[int, List[int]]  = defaultdict(list)  # ALL pids per tid
    prev_labels:      Dict[int, int]        = {}

    results.total_frames = len(frame_events)
    max_fi = max(frame_events.keys()) if frame_events else 0

    # Accumulators for fragmentation metrics
    all_raw_ids:  set = set()   # every tracker ID seen across whole clip
    all_reid_ids: set = set()   # every persistent ID seen across whole clip

    for fi in sorted(frame_events.keys()):
        entries = frame_events[fi]
        tids = [e[0] for e in entries]
        boxes = [e[1] for e in entries]

        # Use the actual frame or a black frame if index out of range
        if fi < len(frames):
            frame = frames[fi]
        else:
            frame = np.zeros_like(frames[-1])

        labels = reid.update(frame, tids, boxes)

        # ── Per-frame fragmentation counting ─────────────────────────────
        frame_raw_ids  = set(tids)                                     # unique tracker IDs this frame
        frame_reid_ids = set(int(l.split()[-1]) for l in labels.values())  # unique persistent IDs

        results._raw_ids_per_frame.append(len(frame_raw_ids))
        results._reid_ids_per_frame.append(len(frame_reid_ids))

        all_raw_ids  |= frame_raw_ids
        all_reid_ids |= frame_reid_ids
        # ─────────────────────────────────────────────────────────────────

        for tid, label in labels.items():
            pid = int(label.split()[-1])
            tid_all_pids[tid].append(pid)   # track full history for consistency

            # Record first-segment anchor
            if tid in gt_map:
                # This is a second-segment (re-entry) detection
                second_seg_pids[tid].append(pid)
            else:
                # First segment — anchor this tracker_id
                if tid not in first_seg_pid:
                    first_seg_pid[tid] = pid

            # Count ID switches (unexpected label change mid-track)
            if tid in prev_labels and prev_labels[tid] != pid:
                results.id_switches += 1
            prev_labels[tid] = pid

    # Store whole-clip totals
    results.total_raw_ids        = len(all_raw_ids)
    results.total_persistent_ids = len(all_reid_ids)

    # --- Score re-entry events ---------------------------------------------
    for new_tid, orig_tid in gt_map.items():
        pids_seen = second_seg_pids.get(new_tid, [])
        if not pids_seen:
            results.fn += 1   # never matched at all
            results.n_events += 1
            continue

        results.n_events += 1
        # Majority label for this re-entry track
        majority_pid = max(set(pids_seen), key=pids_seen.count)
        expected_pid = first_seg_pid.get(orig_tid)

        if expected_pid is None:
            results.fn += 1   # first segment never seen — can't evaluate
            continue

        if majority_pid == expected_pid:
            results.tp += 1
        elif majority_pid in first_seg_pid.values():
            results.fp += 1   # assigned to a DIFFERENT known person
        else:
            results.fn += 1   # assigned a brand new ID (fragmentation)

    # --- Consistency score ------------------------------------------------
    # For every tracker_id (first or second segment), compute what fraction
    # of its frames carried the dominant label. 1.0 = perfectly stable.
    # Lower = the Re-ID system kept flipping its identity mid-track.
    for tid, pids in tid_all_pids.items():
        if not pids:
            continue
        dominant = max(set(pids), key=pids.count)
        frac = pids.count(dominant) / float(len(pids))
        results.consistency_scores.append(frac)

    return results
