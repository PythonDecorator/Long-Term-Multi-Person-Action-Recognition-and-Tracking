"""
person_reid.py — Appearance-Based Person Re-Identification
===========================================================
Assigns stable "Person N" labels across video frames.

Key design decisions
---------------------
* Training-free: uses HSV colour histograms + weighted cosine distance.
* Robust to tracker ID reassignment: verifies cached assignments each frame.
* Two-phase matching: strict threshold for high-confidence matches, relaxed
  for re-entry after the person has been off-screen.
* Conservative gallery updates: only updates appearance model when the
  detection is clean (good size, appearance consistent with gallery).

Ablation flags (for dissertation evaluation)
--------------------------------------------
  use_bg_crop      — strip outer edges of box (background suppression)
  use_strips       — spatial pyramid: 3 overlapping horizontal strips
  use_occlusion    — skip strips that are too small (partial-occlusion aware)
  use_quality_ema  — adapt EMA rate to detection size

Author : Amos Okpe  (MSc Computer Science, University of Hull)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Stable per-person colour palette (R, G, B) — 12 distinct colours
# ---------------------------------------------------------------------------

PERSON_COLORS_RGB: Tuple[Tuple[int, int, int], ...] = (
    ( 46, 204, 113),   # Person  1 — green
    ( 52, 152, 219),   # Person  2 — blue
    (155,  89, 182),   # Person  3 — purple
    (241, 196,  15),   # Person  4 — yellow
    ( 26, 188, 156),   # Person  5 — teal
    (230, 126,  34),   # Person  6 — orange
    (231,  76,  60),   # Person  7 — red
    ( 52,  73,  94),   # Person  8 — dark slate
    (149, 165, 166),   # Person  9 — silver
    (211,  84,   0),   # Person 10 — burnt orange
    ( 39, 174,  96),   # Person 11 — dark green
    (142,  68, 173),   # Person 12 — dark purple
)


def person_color_rgb(persistent_id: int) -> Tuple[int, int, int]:
    """Return stable (R, G, B) for persistent_id (1-indexed)."""
    return PERSON_COLORS_RGB[(persistent_id - 1) % len(PERSON_COLORS_RGB)]


# ---------------------------------------------------------------------------
# Strip layout constants
# ---------------------------------------------------------------------------

# (row_start_frac, row_end_frac, matching_weight)
_STRIPS: Tuple[Tuple[float, float, float], ...] = (
    (0.00, 0.40, 0.20),  # Strip 0: Head / shoulders — weight 0.20
    (0.25, 0.75, 0.45),  # Strip 1: Torso           — weight 0.45 (dominant)
    (0.60, 1.00, 0.35),  # Strip 2: Legs / feet     — weight 0.35
)

_MIN_STRIP_PX: int   = 20    # minimum strip height to be included
_BG_MARGIN:   float  = 0.10  # fraction of box width to crop each side

# Minimum box area fraction to allow gallery UPDATE (not matching).
# Prevents polluting the gallery with occluded/distant detections.
_MIN_UPDATE_QUALITY: float = 0.008   # ~0.8% of frame area

# If distance to cached gallery entry exceeds this, the tracker has likely
# reassigned this numeric ID to a different physical person → re-match.
_REASSIGNMENT_DIST: float = 0.55


# ---------------------------------------------------------------------------
# PersonReIdentifier
# ---------------------------------------------------------------------------

class PersonReIdentifier:
    """
    Training-free HSV-histogram Re-ID with robust production features.

    Parameters
    ----------
    reid_threshold      : strict matching threshold (high-confidence match)
    reid_threshold_relax: relaxed threshold for re-entry after absence
    ema_alpha_near      : EMA weight for high-quality detections
    ema_alpha_far       : EMA weight for low-quality / distant detections
    max_gallery_size    : maximum simultaneous persons tracked in gallery

    Ablation flags (default: all True = full system)
    --------------------
    use_bg_crop     : suppress background by cropping box width edges
    use_strips      : spatial pyramid (3 overlapping strips)
    use_occlusion   : skip strips too small to be reliable
    use_quality_ema : quality-weighted EMA update rate
    """

    def __init__(
        self,
        reid_threshold:        float = 0.22,
        reid_threshold_relax:  float = 0.38,
        ema_alpha_near:        float = 0.80,
        ema_alpha_far:         float = 0.95,
        max_gallery_size:      int   = 64,
        h_bins:                int   = 32,
        s_bins:                int   = 32,
        # Ablation flags
        use_bg_crop:     bool = True,
        use_strips:      bool = True,
        use_occlusion:   bool = True,
        use_quality_ema: bool = True,
    ) -> None:
        self.threshold        = reid_threshold
        self.threshold_relax  = reid_threshold_relax
        self.alpha_near       = ema_alpha_near
        self.alpha_far        = ema_alpha_far
        self.max_gallery_size = max_gallery_size
        self.h_bins           = h_bins
        self.s_bins           = s_bins

        self.use_bg_crop     = use_bg_crop
        self.use_strips      = use_strips
        self.use_occlusion   = use_occlusion
        self.use_quality_ema = use_quality_ema

        # Gallery: {pid: {"strips": [...], "last_seen": int, "n_obs": int}}
        self._gallery:               Dict[int, Dict] = {}
        # Cached tracker_id → persistent_id mapping
        self._tracker_to_persistent: Dict[int, int]  = {}
        self._next_pid:  int = 1
        self._frame_idx: int = 0

    # ── Embedding extraction ──────────────────────────────────────────────

    def _strip_histogram(self, hsv_crop: np.ndarray) -> np.ndarray:
        """L2-normalised 2D HSV histogram for one crop."""
        if hsv_crop.size == 0:
            return np.zeros(self.h_bins * self.s_bins, dtype=np.float32)
        hist = cv2.calcHist(
            [hsv_crop], [0, 1], None,
            [self.h_bins, self.s_bins],
            [0, 180, 0, 256],
        ).flatten().astype(np.float32)
        norm = np.linalg.norm(hist)
        return hist / (norm + 1e-8)

    def _extract_strips(
        self,
        frame_bgr: np.ndarray,
        box:       Tuple[float, float, float, float],
    ) -> Tuple[List[Optional[np.ndarray]], float]:
        """
        Extract per-strip embeddings from frame_bgr at box.

        Returns
        -------
        strips  : list of 3 embeddings (or None if invalid/disabled)
        quality : box_area / frame_area (used for EMA rate + update gating)
        """
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(frame_bgr.shape[1], int(box[2]))
        y2 = min(frame_bgr.shape[0], int(box[3]))

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        if self.use_bg_crop:
            margin = int(bw * _BG_MARGIN)
            cx1 = x1 + margin
            cx2 = x2 - margin
            if cx2 <= cx1 + 4:
                cx1, cx2 = x1, x2   # box too narrow — skip crop
        else:
            cx1, cx2 = x1, x2

        full_crop = frame_bgr[y1:y2, cx1:cx2]
        ch = full_crop.shape[0]

        if ch == 0 or full_crop.shape[1] == 0:
            return [None, None, None], 0.0

        hsv_crop = cv2.cvtColor(full_crop, cv2.COLOR_BGR2HSV)

        if self.use_strips:
            strips: List[Optional[np.ndarray]] = []
            for r_start, r_end, _w in _STRIPS:
                row0 = int(ch * r_start)
                row1 = int(ch * r_end)
                crop = hsv_crop[row0:row1, :]
                if self.use_occlusion and crop.shape[0] < _MIN_STRIP_PX:
                    strips.append(None)
                else:
                    strips.append(self._strip_histogram(crop))
        else:
            global_hist = self._strip_histogram(hsv_crop)
            strips = [global_hist, global_hist, global_hist]

        frame_area = max(1, frame_bgr.shape[0] * frame_bgr.shape[1])
        quality    = (bw * bh) / frame_area
        return strips, quality

    # ── Distance ──────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))

    def _weighted_distance(
        self,
        query_strips:   List[Optional[np.ndarray]],
        gallery_strips: List[Optional[np.ndarray]],
    ) -> Optional[float]:
        """Weighted cosine distance across valid strip pairs, in [0, 1]."""
        active_w = 0.0
        dist_acc = 0.0
        for i, (_r0, _r1, w) in enumerate(_STRIPS):
            q = query_strips[i]
            g = gallery_strips[i]
            if q is None or g is None:
                continue
            dist_acc += w * self._cosine_dist(q, g)
            active_w += w
        if active_w < 1e-6:
            return None
        return dist_acc / active_w

    # ── Gallery management ────────────────────────────────────────────────

    def _register_new_person(self, strips: List[Optional[np.ndarray]]) -> int:
        """Add a new entry to the gallery and return its pid."""
        if len(self._gallery) >= self.max_gallery_size:
            # Evict the person seen least recently
            oldest = min(self._gallery,
                         key=lambda k: self._gallery[k]["last_seen"])
            del self._gallery[oldest]
        pid = self._next_pid
        self._next_pid += 1
        self._gallery[pid] = {
            "strips":    strips,
            "last_seen": self._frame_idx,
            "n_obs":     1,
        }
        return pid

    def _update_gallery(
        self,
        pid:     int,
        strips:  List[Optional[np.ndarray]],
        quality: float,
    ) -> None:
        """
        EMA update of gallery appearance — only called when detection is clean.

        alpha_near (higher keep-rate) = large/close detection → reliable appearance
        alpha_far  (lower keep-rate)  = small/distant detection → conservative
        """
        if self.use_quality_ema:
            alpha = self.alpha_near if quality >= 0.01 else self.alpha_far
        else:
            alpha = self.alpha_near

        stored     = self._gallery[pid]["strips"]
        new_strips: List[Optional[np.ndarray]] = []

        for i in range(len(_STRIPS)):
            s_new = strips[i]
            s_old = stored[i]
            if s_new is None:
                new_strips.append(s_old)
            elif s_old is None:
                new_strips.append(s_new)
            else:
                blended = alpha * s_old + (1.0 - alpha) * s_new
                norm    = np.linalg.norm(blended)
                new_strips.append(blended / (norm + 1e-8))

        self._gallery[pid]["strips"]    = new_strips
        self._gallery[pid]["last_seen"] = self._frame_idx
        self._gallery[pid]["n_obs"]     = self._gallery[pid].get("n_obs", 0) + 1

    def _best_match(
        self,
        query_strips: List[Optional[np.ndarray]],
        threshold:    float,
        exclude:      Optional[set] = None,
    ) -> Tuple[Optional[int], float]:
        """
        Find the closest gallery entry within *threshold*.

        Returns (best_pid, best_dist) — pid is None if no match found.
        """
        best_pid:  Optional[int] = None
        best_dist: float         = threshold
        for pid, entry in self._gallery.items():
            if exclude and pid in exclude:
                continue
            dist = self._weighted_distance(query_strips, entry["strips"])
            if dist is not None and dist < best_dist:
                best_dist = dist
                best_pid  = pid
        return best_pid, best_dist

    # ── Public API ────────────────────────────────────────────────────────

    def update(
        self,
        frame_bgr:   np.ndarray,
        tracker_ids: List[int],
        boxes:       List[Tuple[float, float, float, float]],
    ) -> Dict[int, str]:
        """
        Process one frame. Returns {tracker_id: "Person N"}.

        Guarantees
        ----------
        * Each persistent ID appears at most once per frame.
        * Tracker ID reassignment is detected and corrected.
        * Gallery is only updated from clean, high-quality detections.
        """
        self._frame_idx += 1
        labels:        Dict[int, str] = {}
        assigned_pids: set            = set()

        for tid, box in zip(tracker_ids, boxes):
            strips, quality = self._extract_strips(frame_bgr, box)
            pid = self._resolve(tid, strips, quality, assigned_pids)
            assigned_pids.add(pid)
            labels[tid] = "Person {}".format(pid)

        return labels

    def _resolve(
        self,
        tid:           int,
        strips:        List[Optional[np.ndarray]],
        quality:       float,
        assigned_pids: set,
    ) -> int:
        """
        Core matching logic — called once per detection per frame.

        Steps
        -----
        1. If this tracker_id has a cached mapping:
           a. Verify the cached assignment is still visually consistent.
              If the appearance changed too drastically (> _REASSIGNMENT_DIST),
              the tracker has recycled this numeric ID → treat as new.
           b. If pid is already assigned this frame (two tids → same pid),
              treat as new.
           c. Otherwise keep the cached mapping, update gallery if clean.

        2. If new or invalidated tracker_id:
           a. Try strict matching first (self.threshold).
           b. Fall back to relaxed matching (self.threshold_relax).
           c. Register as new person if no match found.
        """
        if tid in self._tracker_to_persistent:
            cached_pid = self._tracker_to_persistent[tid]

            # Check if tracker recycled this numeric ID (reassignment detection)
            if cached_pid in self._gallery:
                dist = self._weighted_distance(
                    strips, self._gallery[cached_pid]["strips"]
                )
                if dist is not None and dist > _REASSIGNMENT_DIST:
                    # Appearance is way off — tracker gave this ID to someone else
                    del self._tracker_to_persistent[tid]
                    return self._match_or_register(
                        tid, strips, quality, assigned_pids
                    )

            if cached_pid not in assigned_pids:
                # Normal update — known tracker_id, known person
                if (quality >= _MIN_UPDATE_QUALITY
                        and cached_pid in self._gallery):
                    self._update_gallery(cached_pid, strips, quality)
                self._gallery.setdefault(cached_pid, {
                    "strips": strips, "last_seen": self._frame_idx, "n_obs": 1
                })["last_seen"] = self._frame_idx
                return cached_pid
            else:
                # pid conflict: another tracker_id already claimed this pid.
                # Find the next best match.
                return self._match_or_register(
                    tid, strips, quality, assigned_pids
                )

        # Completely new tracker_id — try to match a known person
        return self._match_or_register(tid, strips, quality, assigned_pids)

    def _match_or_register(
        self,
        tid:           int,
        strips:        List[Optional[np.ndarray]],
        quality:       float,
        assigned_pids: set,
    ) -> int:
        """
        Two-phase matching for a tracker_id that needs a fresh gallery match.

        Phase 1: strict threshold (self.threshold ≈ 0.22)
          High-confidence match — update gallery immediately.

        Phase 2: relaxed threshold (self.threshold_relax ≈ 0.38)
          Likely the same person after re-entry / lighting change.
          Accept the match but update gallery conservatively.

        If both phases fail: register as a new person.
        """
        # Phase 1 — strict
        pid, dist = self._best_match(strips, self.threshold, exclude=assigned_pids)
        if pid is not None:
            if quality >= _MIN_UPDATE_QUALITY:
                self._update_gallery(pid, strips, quality)
            self._tracker_to_persistent[tid] = pid
            return pid

        # Phase 2 — relaxed (re-entry recovery)
        pid, dist = self._best_match(
            strips, self.threshold_relax, exclude=assigned_pids
        )
        if pid is not None:
            # Conservative update — appearance changed enough that we needed
            # the relaxed threshold, so don't update gallery aggressively
            if quality >= _MIN_UPDATE_QUALITY * 2:   # higher bar for relaxed match
                self._update_gallery(pid, strips, quality)
            self._tracker_to_persistent[tid] = pid
            return pid

        # No match — new person
        pid = self._register_new_person(strips)
        self._tracker_to_persistent[tid] = pid
        return pid

    # ── Utilities ─────────────────────────────────────────────────────────

    def get_persistent_id(self, tracker_id: int) -> Optional[int]:
        return self._tracker_to_persistent.get(tracker_id)

    def reset(self) -> None:
        """Call between independent videos."""
        self._gallery.clear()
        self._tracker_to_persistent.clear()
        self._next_pid  = 1
        self._frame_idx = 0
