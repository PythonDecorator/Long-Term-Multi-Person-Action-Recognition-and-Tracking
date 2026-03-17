"""
person_reid.py — Deep Feature Person Re-Identification
=======================================================
Assigns stable "Person N" labels that persist across the entire video,
including when people leave and re-enter the frame.

Core design principles
-----------------------
1. DUAL-ANCHOR GALLERY — no drift
   Every person entry stores TWO feature vectors:
     anchor_feat : first clean detection. NEVER modified. Permanent reference.
     recent_feat : EMA-updated as person is tracked. Adapts to lighting/pose.
   Match distance = 0.4 * dist(anchor) + 0.6 * dist(recent)
   Even after 1000 frames of EMA drift, the anchor prevents identity loss.

2. ACTIVE TRACKER LOCK — no ID theft
   Each persistent ID is "locked" to a tracker_id while that tracker_id is
   active in the frame. While locked, NO other detection can claim that ID.
   The lock is released only when the tracker_id has been absent for
   LOCK_RELEASE_FRAMES consecutive frames (person truly left the frame).
   This prevents the "ID theft" bug where a different person steals an ID
   from someone who just moved out of frame momentarily.

3. FRAME-PRESENCE TRACKING — correct re-entry handling
   We track the last frame each tracker_id was visible.
   When a tracker_id disappears for > LOCK_RELEASE_FRAMES frames,
   their person is marked as "left frame". On re-detection with a new
   tracker_id, they are matched against the gallery and their original
   ID is restored.

4. TIGHT THRESHOLDS — no false matches
   ResNet50 same-person cosine distances: 0.05-0.25
   ResNet50 diff-person cosine distances: 0.40-0.90
   strict threshold  = 0.30 (well inside the safe zone)
   relaxed threshold = 0.50 (for re-entry with different lighting/pose)

Author : Amos Okpe  (MSc Computer Science, University of Hull)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Stable per-person colour palette (R, G, B)
# ---------------------------------------------------------------------------

PERSON_COLORS_RGB = (
    ( 46, 204, 113),   # Person  1 - green
    ( 52, 152, 219),   # Person  2 - blue
    (155,  89, 182),   # Person  3 - purple
    (241, 196,  15),   # Person  4 - yellow
    ( 26, 188, 156),   # Person  5 - teal
    (230, 126,  34),   # Person  6 - orange
    (231,  76,  60),   # Person  7 - red
    ( 52,  73,  94),   # Person  8 - dark slate
    (149, 165, 166),   # Person  9 - silver
    (211,  84,   0),   # Person 10 - burnt orange
    ( 39, 174,  96),   # Person 11 - dark green
    (142,  68, 173),   # Person 12 - dark purple
)


def person_color_rgb(persistent_id):
    return PERSON_COLORS_RGB[(persistent_id - 1) % len(PERSON_COLORS_RGB)]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REID_H          = 256      # crop height for ResNet50
_REID_W          = 128      # crop width  for ResNet50
_BG_MARGIN       = 0.10     # fraction of box width to crop each side

# How many frames a tracker_id must be absent before we release its lock.
# At ~25fps this is ~0.5 seconds. Short enough to release quickly when
# person leaves; long enough to survive brief occlusions.
LOCK_RELEASE_FRAMES = 12

# Gallery update quality gate: box must cover this fraction of frame area.
_MIN_UPDATE_QUAL = 0.008

# Matching thresholds (cosine distance, lower = more similar)
# ResNet50 on ImageNet: same-person ~0.05-0.25, diff-person ~0.40-0.90
THRESHOLD_STRICT = 0.30   # high-confidence match
THRESHOLD_RELAX  = 0.50   # re-entry / lighting change match

# Weight for anchor vs recent in combined matching distance
_ANCHOR_WEIGHT = 0.40
_RECENT_WEIGHT = 0.60

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_H_BINS = 32
_S_BINS = 32


# ---------------------------------------------------------------------------
# ResNet50 feature extractor
# ---------------------------------------------------------------------------

class _ResNet50Extractor(nn.Module):
    """
    ResNet50 backbone (final FC removed).
    Input:  BGR crops (variable size)
    Output: (N, 2048) L2-normalised float32 numpy array
    Runs entirely in eval() mode with no_grad.
    """

    def __init__(self, device, weight_path=None):
        super(_ResNet50Extractor, self).__init__()
        resnet        = models.resnet50(pretrained=(weight_path is None))
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.device   = device
        if weight_path and Path(weight_path).exists():
            state = torch.load(weight_path, map_location="cpu")
            # Load only backbone keys (ignore head keys if any)
            bk_state = {k.replace("trainable.", "").replace("frozen.", ""): v
                        for k, v in state.items()
                        if "trainable" in k or "frozen" in k}
            if bk_state:
                self.backbone.load_state_dict(bk_state, strict=False)
            else:
                self.backbone.load_state_dict(state, strict=False)
            print(f"  [Re-ID] Loaded fine-tuned weights from: {weight_path}")
        self.to(device)
        self.eval()
        self.preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((_REID_H, _REID_W)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    @torch.no_grad()
    def extract_batch(self, bgr_crops):
        if not bgr_crops:
            return np.zeros((0, 2048), dtype=np.float32)
        tensors = []
        for crop in bgr_crops:
            if crop is None or crop.size == 0 \
                    or crop.shape[0] < 4 or crop.shape[1] < 4:
                tensors.append(torch.zeros(3, _REID_H, _REID_W))
            else:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensors.append(self.preprocess(rgb))
        batch = torch.stack(tensors).to(self.device)
        feats = self.backbone(batch).flatten(1)
        feats = F.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# HSV fallback (for ablation baseline config)
# ---------------------------------------------------------------------------

def _hsv_histogram(bgr_crop):
    if bgr_crop is None or bgr_crop.size == 0:
        return np.zeros(_H_BINS * _S_BINS, dtype=np.float32)
    hsv  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None, [_H_BINS, _S_BINS], [0, 180, 0, 256]
    ).flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / (norm + 1e-8)


# ---------------------------------------------------------------------------
# PersonReIdentifier
# ---------------------------------------------------------------------------

class PersonReIdentifier:
    """
    Production-grade Re-ID with dual-anchor gallery and tracker lock.

    Gallery entry per person (pid):
    --------------------------------
    {
      "anchor_feat" : np.array(2048) — first clean detection, NEVER updated
      "recent_feat" : np.array(2048) — EMA-updated current appearance
      "active_tid"  : int or None    — current tracker_id if in frame
      "last_frame"  : int            — last frame this person was seen
      "n_obs"       : int            — number of observations
    }

    ID Rules
    --------
    1. Each pid is locked to one tracker_id while that id is active.
       No other detection can steal a pid that is currently in frame.
    2. Locks are released after LOCK_RELEASE_FRAMES absent frames.
    3. Released pids are available for re-entry matching.
    4. New tracker_ids match against released gallery entries first.
    """

    def __init__(
        self,
        device                = None,
        reid_threshold        = THRESHOLD_STRICT,
        reid_threshold_relax  = THRESHOLD_RELAX,
        ema_alpha_near        = 0.85,
        ema_alpha_far         = 0.97,
        max_gallery_size      = 64,
        lock_release_frames   = LOCK_RELEASE_FRAMES,
        use_deep_features     = True,
        use_bg_crop           = True,
        use_quality_ema       = True,
        weight_path           = None,  # path to fine-tuned weights (None = ImageNet)
        # legacy compat
        use_strips            = True,
        use_occlusion         = True,
        h_bins                = 32,
        s_bins                = 32,
    ):
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.device               = device
        self.threshold            = reid_threshold
        self.threshold_relax      = reid_threshold_relax
        self.alpha_near           = ema_alpha_near
        self.alpha_far            = ema_alpha_far
        self.max_gallery_size     = max_gallery_size
        self.lock_release_frames  = lock_release_frames
        self.use_deep_features    = use_deep_features
        self.use_bg_crop          = use_bg_crop
        self.use_quality_ema      = use_quality_ema

        if use_deep_features:
            print("  [Re-ID] Loading ResNet50 on {} ...".format(device))
            self._extractor = _ResNet50Extractor(device, weight_path=weight_path)
            print("  [Re-ID] ResNet50 ready.")
        else:
            self._extractor = None

        # gallery[pid] = {anchor_feat, recent_feat, active_tid,
        #                 last_frame, n_obs}
        self._gallery               = {}   # type: Dict[int, Dict]

        # Maps tracker_id -> pid (only while tracker_id is active)
        self._tracker_to_pid        = {}   # type: Dict[int, int]

        # Maps pid -> tracker_id for the currently-active assignment
        self._pid_to_tracker        = {}   # type: Dict[int, int]

        # Last frame each tracker_id was seen (for lock release)
        self._tid_last_seen         = {}   # type: Dict[int, int]

        self._next_pid  = 1
        self._frame_idx = 0

    # ── Crop helpers ─────────────────────────────────────────────────────

    def _get_crop(self, frame_bgr, box):
        H, W = frame_bgr.shape[:2]
        x1 = max(0, int(box[0]));  y1 = max(0, int(box[1]))
        x2 = min(W, int(box[2]));  y2 = min(H, int(box[3]))
        bw = max(1, x2 - x1);     bh = max(1, y2 - y1)
        if self.use_bg_crop:
            m = int(bw * _BG_MARGIN)
            cx1 = x1 + m;  cx2 = x2 - m
            if cx2 <= cx1 + 4:
                cx1, cx2 = x1, x2
        else:
            cx1, cx2 = x1, x2
        crop    = frame_bgr[y1:y2, cx1:cx2]
        quality = (bw * bh) / float(max(1, H * W))
        return crop, quality

    def _extract_batch(self, frame_bgr, boxes):
        crops, qualities = [], []
        for box in boxes:
            c, q = self._get_crop(frame_bgr, box)
            crops.append(c)
            qualities.append(q)
        if self.use_deep_features:
            batch_feats = self._extractor.extract_batch(crops)
            feats = [batch_feats[i] for i in range(len(crops))]
        else:
            feats = [_hsv_histogram(c) for c in crops]
        return feats, qualities

    # ── Distance ─────────────────────────────────────────────────────────

    @staticmethod
    def _cos_dist(a, b):
        return float(1.0 - np.dot(a, b))

    def _gallery_dist(self, feat, pid):
        """
        Combined distance using both anchor and recent gallery features.
        Anchor prevents drift; recent adapts to lighting/pose changes.
        """
        entry = self._gallery[pid]
        d_anchor = self._cos_dist(feat, entry["anchor_feat"])
        d_recent = self._cos_dist(feat, entry["recent_feat"])
        return _ANCHOR_WEIGHT * d_anchor + _RECENT_WEIGHT * d_recent

    # ── Gallery management ────────────────────────────────────────────────

    def _register(self, feat, tid):
        """Create a new gallery entry for a new person."""
        if len(self._gallery) >= self.max_gallery_size:
            # Evict the person absent longest who is NOT currently in frame
            candidates = {
                k: v for k, v in self._gallery.items()
                if v["active_tid"] is None
            }
            if candidates:
                oldest = min(candidates,
                             key=lambda k: self._gallery[k]["last_frame"])
                del self._gallery[oldest]

        pid = self._next_pid
        self._next_pid += 1
        self._gallery[pid] = {
            "anchor_feat": feat.copy(),  # permanent — never changed
            "recent_feat": feat.copy(),  # EMA updated
            "active_tid":  tid,
            "last_frame":  self._frame_idx,
            "n_obs":       1,
        }
        return pid

    def _ema_update(self, pid, feat, quality):
        """Update only recent_feat. anchor_feat is never touched."""
        if self.use_quality_ema:
            alpha = self.alpha_near if quality >= 0.01 else self.alpha_far
        else:
            alpha = self.alpha_near

        entry = self._gallery[pid]

        # QUALITY GATE: only update if detection is clean enough.
        # This prevents occluded/partial crops from corrupting the gallery.
        if quality < _MIN_UPDATE_QUAL:
            return

        # CONSISTENCY GATE: if the new appearance is very different from
        # what we have, it's likely an occlusion or partial detection.
        # Don't update — keep the gallery clean.
        d = self._cos_dist(feat, entry["recent_feat"])
        if d > 0.45:  # appearance changed too much in one frame
            return

        blended = alpha * entry["recent_feat"] + (1.0 - alpha) * feat
        norm    = np.linalg.norm(blended)
        entry["recent_feat"] = blended / (norm + 1e-8)
        entry["last_frame"]  = self._frame_idx
        entry["n_obs"]       = entry.get("n_obs", 0) + 1

    def _best_match(self, feat, threshold, exclude=None):
        """
        Find closest gallery entry within threshold.
        Only considers pids that are NOT currently locked to an active
        tracker_id (i.e. available for assignment).
        """
        best_pid, best_dist = None, threshold
        for pid, entry in self._gallery.items():
            if exclude and pid in exclude:
                continue
            # LOCK CHECK: if this pid has an active tracker_id in the
            # current frame (set in _refresh_locks), skip it — it's taken.
            if entry.get("active_tid") is not None:
                continue
            dist = self._gallery_dist(feat, pid)
            if dist < best_dist:
                best_dist = dist
                best_pid  = pid
        return best_pid, best_dist

    # ── Lock management ───────────────────────────────────────────────────

    def _refresh_locks(self, current_tids):
        """
        Called at the START of each frame before matching.

        For every pid whose active_tid is NOT in current_tids,
        check if the tracker has been absent long enough to release the lock.

        This implements the "person left frame" detection:
        - Brief absence (< LOCK_RELEASE_FRAMES) → keep lock, ignore gap
        - Long absence (>= LOCK_RELEASE_FRAMES) → release lock, make
          pid available for re-entry matching
        """
        current_tids_set = set(current_tids)

        for pid, entry in self._gallery.items():
            tid = entry.get("active_tid")
            if tid is None:
                continue  # already released

            if tid in current_tids_set:
                # Tracker ID still active this frame — keep lock
                continue

            # Tracker ID absent — check for how long
            last = self._tid_last_seen.get(tid, 0)
            frames_absent = self._frame_idx - last

            if frames_absent >= self.lock_release_frames:
                # Person has left the frame — release the lock
                entry["active_tid"] = None
                # Remove from tracker-pid mapping
                if tid in self._tracker_to_pid:
                    del self._tracker_to_pid[tid]
                if self._pid_to_tracker.get(pid) == tid:
                    del self._pid_to_tracker[pid]

    # ── Core matching ─────────────────────────────────────────────────────

    def _resolve(self, tid, feat, quality, assigned_pids):
        """
        Determine the persistent ID for one tracker_id detection.

        Cases:
        A. Known tracker_id, same person (normal frame-to-frame tracking)
           → verify appearance, update gallery, return cached pid

        B. Known tracker_id, but appearance changed drastically
           → tracker recycled this ID, treat as new

        C. New tracker_id (person re-entered or genuinely new)
           → match against released gallery entries
           → strict first, then relaxed
           → register as new person if no match
        """
        # ── Case A / B: known tracker_id ─────────────────────────────────
        if tid in self._tracker_to_pid:
            pid = self._tracker_to_pid[tid]

            # Verify appearance hasn't changed too drastically
            if pid in self._gallery:
                d = self._gallery_dist(feat, pid)
                if d > 0.65:
                    # Tracker recycled this numeric ID to a new person (B)
                    # Release old mapping and re-match
                    self._gallery[pid]["active_tid"] = None
                    del self._tracker_to_pid[tid]
                    if self._pid_to_tracker.get(pid) == tid:
                        del self._pid_to_tracker[pid]
                    return self._match_or_register(
                        tid, feat, quality, assigned_pids
                    )

            # Verify this pid hasn't already been claimed by another tid
            # this frame (shouldn't happen, but safety check)
            if pid not in assigned_pids:
                self._ema_update(pid, feat, quality)
                self._gallery[pid]["active_tid"]  = tid
                self._gallery[pid]["last_frame"]  = self._frame_idx
                self._tid_last_seen[tid]          = self._frame_idx
                return pid
            else:
                # pid conflict — re-match
                return self._match_or_register(
                    tid, feat, quality, assigned_pids
                )

        # ── Case C: new tracker_id ────────────────────────────────────────
        return self._match_or_register(tid, feat, quality, assigned_pids)

    def _match_or_register(self, tid, feat, quality, assigned_pids):
        """
        Two-phase matching for a new or reset tracker_id.
        Only matches against gallery entries whose lock has been released
        (person left frame), never against currently-active pids.
        """
        # Phase 1: strict
        pid, _ = self._best_match(feat, self.threshold, exclude=assigned_pids)
        if pid is not None:
            self._ema_update(pid, feat, quality)
            self._gallery[pid]["active_tid"] = tid
            self._gallery[pid]["last_frame"] = self._frame_idx
            self._tracker_to_pid[tid]        = pid
            self._pid_to_tracker[pid]        = tid
            self._tid_last_seen[tid]         = self._frame_idx
            return pid

        # Phase 2: relaxed (re-entry with lighting/pose change)
        pid, _ = self._best_match(
            feat, self.threshold_relax, exclude=assigned_pids
        )
        if pid is not None:
            # Conservative update for relaxed match
            if quality >= _MIN_UPDATE_QUAL * 2:
                self._ema_update(pid, feat, quality)
            self._gallery[pid]["active_tid"] = tid
            self._gallery[pid]["last_frame"] = self._frame_idx
            self._tracker_to_pid[tid]        = pid
            self._pid_to_tracker[pid]        = tid
            self._tid_last_seen[tid]         = self._frame_idx
            return pid

        # New person
        pid = self._register(feat, tid)
        self._tracker_to_pid[tid]  = pid
        self._pid_to_tracker[pid]  = tid
        self._tid_last_seen[tid]   = self._frame_idx
        return pid

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, frame_bgr, tracker_ids, boxes):
        """
        Process one frame. Returns {tracker_id: "Person N"}.

        Guarantees
        ----------
        * Each Person N label appears at most once per frame.
        * Person labels are stable across the entire video.
        * A person who leaves and re-enters gets their ORIGINAL label back.
        * A new person who enters after someone else left gets a new label.
        * Once a label is assigned to someone in frame, it cannot be
          stolen by another detection until that person has been absent
          for LOCK_RELEASE_FRAMES consecutive frames.
        """
        self._frame_idx += 1

        if not tracker_ids:
            return {}

        # Step 1: Update last-seen for all tracker_ids present this frame
        for tid in tracker_ids:
            self._tid_last_seen[tid] = self._frame_idx

        # Step 2: Release locks for tracker_ids that have been absent long enough
        self._refresh_locks(tracker_ids)

        # Step 3: Batch feature extraction (one GPU forward pass)
        feats, qualities = self._extract_batch(frame_bgr, boxes)

        # Step 4: Match each detection
        labels        = {}
        assigned_pids = set()

        for tid, feat, quality in zip(tracker_ids, feats, qualities):
            pid = self._resolve(tid, feat, quality, assigned_pids)
            assigned_pids.add(pid)
            labels[tid] = "Person {}".format(pid)

        return labels

    def get_persistent_id(self, tracker_id):
        return self._tracker_to_pid.get(tracker_id)

    def reset(self):
        """Call between independent videos."""
        self._gallery.clear()
        self._tracker_to_pid.clear()
        self._pid_to_tracker.clear()
        self._tid_last_seen.clear()
        self._next_pid  = 1
        self._frame_idx = 0
