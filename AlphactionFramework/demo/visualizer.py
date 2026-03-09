"""
AVAVisualizer — modernised, threading-based, with Person Re-ID.

Compatible with Python 3.9+, PyTorch 2.x, Pillow 10+.
Uses queue.Queue and threading.Thread throughout (no multiprocessing).
Pillow 10+ API: textbbox() replaces removed textsize().

Re-ID integration (new in this version)
----------------------------------------
Every frame that contains bounding boxes is passed through a
PersonReIdentifier (HSV colour-histogram + cosine distance).  The
re-identifier maps unstable tracker IDs onto persistent person numbers
("Person 1", "Person 2", …) that remain consistent even when a person
temporarily leaves the frame.

Visual changes vs original
---------------------------
* Each bounding box is drawn in a stable per-person colour
  (not a fixed dark red for everyone).
* A coloured name badge ("Person 1") is drawn at the top-left corner
  of every bounding box, always visible regardless of whether an action
  has been predicted for that person yet.
* Action labels above the box are unchanged.
"""

from __future__ import annotations

import queue
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from person_reid import PersonReIdentifier, person_color_rgb

cv2.setNumThreads(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _video_info(path: str | int) -> Dict:
    cap = cv2.VideoCapture(path)
    info = dict(
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps      = cap.get(cv2.CAP_PROP_FPS),
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    """Return (width, height) of *text* — works on Pillow 8, 9, and 10+."""
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    else:
        return draw.textsize(text, font=font)


def _ms_to_timestamp(ms: float) -> str:
    ms    = int(ms)
    msec  = ms % 1000
    ms  //= 1000
    sec   = ms % 60
    ms  //= 60
    mins  = ms % 60
    hrs   = ms // 60
    return f"{hrs:02d}:{mins:02d}:{sec:02d}.{msec:03d}"


def _boxes_to_list(boxes) -> List[Tuple[float, float, float, float]]:
    """Convert a tensor or list of boxes to a plain Python list of 4-tuples."""
    if boxes is None:
        return []
    if isinstance(boxes, torch.Tensor):
        return [tuple(float(v) for v in row) for row in boxes]
    return [tuple(float(v) for v in b) for b in boxes]


def _ids_to_list(ids) -> List[int]:
    """Convert a tensor or list of tracker IDs to a plain Python list of ints."""
    if ids is None:
        return []
    if isinstance(ids, torch.Tensor):
        return [int(v) for v in ids.view(-1)]
    return [int(v) for v in ids]


# ---------------------------------------------------------------------------
# AVAVisualizer
# ---------------------------------------------------------------------------

class AVAVisualizer:
    """
    Reads the original video frame-by-frame, composites bounding boxes and
    action labels, and writes the annotated video to *output_path*.

    In non-realtime mode two background threads handle:
      1. Frame loading      (_load_frames)
      2. Frame writing      (_write_frames)
    Communication happens via standard queue.Queue objects.

    Person Re-ID
    ------------
    A PersonReIdentifier instance is maintained for the lifetime of each
    video.  On every frame with detected persons, reid.update() is called
    with the current BGR frame, tracker IDs, and bounding boxes.  The
    returned mapping (tracker_id -> "Person N") is stored in
    self._current_labels and used by:

    * _render_labels — to draw the person-ID badge on each box and to
      look up cached action labels by persistent ID.
    * _update_action_dict — to re-key the action cache by persistent ID
      so that action history survives tracker ID resets.
    """

    # ── Category tables ───────────────────────────────────────────────────

    CATEGORIES: List[str] = [
        "bend/bow", "crawl", "crouch/kneel", "dance", "fall down",
        "get up", "jump/leap", "lie/sleep", "martial art", "run/jog",
        "sit", "stand", "swim", "walk",
        "answer phone", "brush teeth", "carry/hold sth.", "catch sth.",
        "chop", "climb", "clink glass", "close", "cook", "cut", "dig",
        "dress/put on clothing", "drink", "drive", "eat", "enter", "exit",
        "extract", "fishing", "hit sth.", "kick sth.", "lift/pick up",
        "listen to sth.", "open", "paint", "play board game",
        "play musical instrument", "play with pets", "point to sth.",
        "press", "pull sth.", "push sth.", "put down", "read", "ride",
        "row boat", "sail boat", "shoot", "shovel", "smoke", "stir",
        "take a photo", "look at a cellphone", "throw", "touch sth.",
        "turn", "watch screen", "work on a computer", "write",
        "fight/hit sb.", "give/serve sth. to sb.", "grab sb.", "hand clap",
        "hand shake", "hand wave", "hug sb.", "kick sb.", "kiss sb.",
        "lift sb.", "listen to sb.", "play with kids", "push sb.", "sing",
        "take sth. from sb.", "talk", "watch sb.",
    ]

    COMMON_CATES: List[str] = [
        "dance", "run/jog", "sit", "stand", "swim", "walk",
        "answer phone", "carry/hold sth.", "drive",
        "play musical instrument", "ride",
        "fight/hit sb.", "listen to sb.", "talk", "watch sb.",
    ]

    EXCLUSION: List[str] = [
        "crawl", "brush teeth", "catch sth.", "chop", "clink glass", "cook",
        "dig", "exit", "extract", "fishing", "kick sth.", "paint",
        "play board game", "play with pets", "press", "row boat", "shovel",
        "stir", "kick sb.", "play with kids",
    ]

    # Action category group colours (R, G, B) — movement / object / social
    ACTION_COLORS: Tuple[Tuple[int, ...], ...] = (
        (176, 85, 234),
        ( 87, 118, 198),
        ( 52, 189, 199),
    )

    # ── Init ──────────────────────────────────────────────────────────────

    def __init__(
        self,
        input_path:           str | int,
        output_path:          str,
        realtime:             bool,
        start:                int,
        duration:             int,
        show_time:            bool,
        confidence_threshold: float = 0.5,
        exclude_class:        Optional[List[str]] = None,
        common_cate:          bool = False,
    ) -> None:
        self.info      = _video_info(input_path)
        self.realtime  = realtime
        self.start     = start
        self.duration  = duration
        self.show_time = show_time
        self.thresh    = confidence_threshold

        fps = self.info["fps"]
        if fps == 0 or fps > 100:
            print(f"Warning: suspicious frame rate {fps} — output may be wrong.")

        # Category setup
        if common_cate:
            self.cate_list = self.COMMON_CATES
            self.cat_split = (6, 11)
        else:
            self.cate_list = self.CATEGORIES
            self.cat_split = (14, 63)

        self.cls2id   = {name: i for i, name in enumerate(self.cate_list)}
        excl          = exclude_class if exclude_class is not None else self.EXCLUSION
        self.excl_ids = {self.cls2id[c] for c in excl if c in self.cls2id}

        # Drawing params
        W, H           = self.info["width"], self.info["height"]
        long_side      = min(W, H)
        self.width     = W
        self.height    = H
        self.font_size = max(int(round(long_side / 40)), 1)
        self.box_width = max(int(round(long_side / 180)), 1)
        self.font      = ImageFont.truetype("./Roboto-Bold.ttf", self.font_size)
        badge_fs       = max(int(round(long_side / 52)), 1)
        self.badge_font = ImageFont.truetype("./Roboto-Bold.ttf", badge_fs)
        self.label_alpha = int(0.6 * 255)
        self.badge_alpha = int(0.85 * 255)

        # ── Re-ID state ──────────────────────────────────────────────────
        # Action cache keyed by persistent_id (NOT tracker id).
        # This means action labels survive tracker ID resets on re-entry.
        self._action_dict: Dict[int, Dict] = {}

        # Populated by _run_reid() on every frame that has detections.
        # Maps tracker_id (int) -> "Person N" label string.
        self._current_labels: Dict[int, str] = {}

        # Set of tracker IDs that passed box filtering this frame.
        # _render_labels and _update_action_dict skip IDs not in this set.
        self._valid_tracker_ids: set = set()

        # Person re-identifier lives for the duration of this video.
        self._reid = PersonReIdentifier(
            reid_threshold       = 0.22,   # strict — high-confidence match
            reid_threshold_relax = 0.38,   # relaxed — re-entry recovery
            ema_alpha_near       = 0.80,
            ema_alpha_far        = 0.95,
            max_gallery_size     = 64,
        )

        # ── Realtime or offline output ────────────────────────────────────
        if realtime:
            fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
            self._outvid = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        else:
            self._frame_q  = queue.Queue(maxsize=512)
            self._result_q = queue.Queue()
            self._track_q  = queue.Queue()
            self._done_q   = queue.Queue()

            self._loader = threading.Thread(
                target=self._load_frames, args=(input_path,),
                daemon=True, name="FrameLoader",
            )
            self._writer = threading.Thread(
                target=self._write_frames, args=(output_path,),
                daemon=True, name="FrameWriter",
            )
            self._loader.start()
            self._writer.start()

    # ── Re-ID helpers ─────────────────────────────────────────────────────

    # ── Box filtering ─────────────────────────────────────────────────────

    @staticmethod
    def _box_area(box: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _box_iou(a: Tuple, b: Tuple) -> float:
        """Intersection-over-area-of-a  (how much of box *a* is inside *b*)."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        return inter / area_a

    def _filter_boxes(
        self,
        box_list:    List[Tuple[float, float, float, float]],
        tracker_ids: List[int],
    ) -> Tuple[List[Tuple], List[int]]:
        """
        Remove spurious non-person detections before Re-ID and drawing.

        Rules (all must pass):

        1. Minimum area    — box must cover >= 1.5% of frame pixels.
           A 1080p frame is 1920x1080 = 2,073,600px. 1.5% ≈ 31,000px
           ≈ a 135x230px box, roughly a person at ~4m distance from camera.
           This aggressively removes hands, heads, floor fragments.

        2. Minimum height  — box must be >= 20% of frame height.
           On 720p that is 144px — the minimum height for a standing person
           visible from at least the waist up.

        3. Minimum width   — box must be >= 3% of frame width.
           Removes thin vertical slivers from edge detections.

        4. Aspect ratio    — height / width must be in [1.2, 5.0].
           People standing or walking are always taller than wide.
           Lower bound 1.2 removes wide horizontal objects (tables, cars).
           Upper bound 5.0 removes extreme slivers.

        Sorted largest-first so foreground people claim persistent IDs
        before any overlapping partial detections.
        """
        frame_area = self.width * self.height
        min_area   = 0.015  * frame_area   # 1.5 % of frame
        min_height = 0.20   * self.height  # 20 % of frame height
        min_width  = 0.030  * self.width   # 3 % of frame width

        paired = sorted(
            zip(box_list, tracker_ids),
            key=lambda x: self._box_area(x[0]),
            reverse=True,
        )

        kept_boxes: List[Tuple] = []
        kept_ids:   List[int]   = []

        for box, tid in paired:
            x1, y1, x2, y2 = box
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)

            # Rule 1 — minimum area
            if self._box_area(box) < min_area:
                continue
            # Rule 2 — minimum height
            if h < min_height:
                continue
            # Rule 3 — minimum width
            if w < min_width:
                continue
            # Rule 4 — aspect ratio (people are taller than wide)
            if not (1.2 <= h / w <= 5.0):
                continue

            kept_boxes.append(box)
            kept_ids.append(tid)

        return kept_boxes, kept_ids

    def _run_reid(
        self,
        frame_bgr: np.ndarray,
        boxes,
        ids,
    ) -> None:
        """
        Filter boxes, then run PersonReIdentifier for one frame and store
        the result in self._current_labels (tracker_id -> "Person N").

        Boxes are filtered before Re-ID so that:
          - Spurious non-person detections are never added to the gallery.
          - Largest boxes are processed first, so they claim persistent IDs
            before any overlapping partial-crop fragments.

        Parameters
        ----------
        frame_bgr : H x W x 3 BGR numpy array from OpenCV.
        boxes     : bounding boxes (tensor or list of 4-tuples).
        ids       : tracker IDs parallel to boxes (tensor or list of ints).
        """
        tracker_ids = _ids_to_list(ids)
        box_list    = _boxes_to_list(boxes)

        if not tracker_ids:
            self._current_labels = {}
            self._valid_tracker_ids: set = set()
            return

        # Filter and sort — this is the only source of truth for valid detections
        valid_boxes, valid_ids = self._filter_boxes(box_list, tracker_ids)

        # Store the set of valid tracker IDs so _render_labels can skip invalid ones
        self._valid_tracker_ids: set = set(valid_ids)

        if not valid_ids:
            self._current_labels = {}
            return

        self._current_labels = self._reid.update(frame_bgr, valid_ids, valid_boxes)

    def _persistent_id_for(self, tracker_id: int) -> int:
        """
        Return the persistent person number for tracker_id.

        Falls back to tracker_id itself if Re-ID has not yet seen this ID
        (e.g. the very first frame before _run_reid has been called).
        """
        pid = self._reid.get_persistent_id(tracker_id)
        return pid if pid is not None else tracker_id

    # ── Realtime API ──────────────────────────────────────────────────────

    def write_realtime_frame(
        self,
        result,
        orig_img: np.ndarray,
        boxes,
        scores,
        ids,
    ) -> bool:
        """Composite and display one frame.  Returns False when ESC pressed."""
        # orig_img is RGB from detector; convert to BGR for OpenCV/Re-ID
        frame_bgr = orig_img[:, :, ::-1].copy()

        if boxes is not None:
            # Run Re-ID first so _current_labels is ready for both
            # _update_action_dict and _render_labels below.
            self._run_reid(frame_bgr, boxes, ids)

        if result is not None:
            pred, _ts, pred_ids = result
            self._update_action_dict(pred.get_field("scores"), pred_ids)

        if boxes is not None:
            mask      = self._render_labels(frame_bgr, boxes, ids)
            frame_bgr = self._blend(frame_bgr, mask)

        cv2.imshow("AlphAction", frame_bgr)
        self._outvid.write(frame_bgr)
        return cv2.waitKey(1) != 27

    # ── Offline queue API ─────────────────────────────────────────────────

    def send_result(self, result) -> None:
        self._result_q.put(result)

    def send_track(self, result) -> None:
        self._track_q.put(result)

    def show_progress(self, total_frames: int) -> None:
        cnt = 0
        while True:
            try:
                self._done_q.get_nowait()
                cnt += 1
            except queue.Empty:
                break
        pbar = tqdm(total=total_frames, initial=cnt,
                    desc="Video Writer", unit=" frame")
        while cnt < total_frames:
            self._done_q.get()
            cnt += 1
            pbar.update(1)
        pbar.close()

    def close(self) -> None:
        if self.realtime:
            self._outvid.release()
            cv2.destroyAllWindows()
        else:
            self._writer.join()
        self._reid.reset()

    # ── Frame loader thread ───────────────────────────────────────────────

    def _load_frames(self, path: str | int) -> None:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, self.start)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if self.duration != -1 and ms > self.start + self.duration:
                break
            self._frame_q.put((frame, ms))
        cap.release()
        self._frame_q.put("DONE")

    # ── Frame writer thread ───────────────────────────────────────────────

    def _write_frames(self, output_path: str) -> None:
        W, H   = self.info["width"], self.info["height"]
        fps    = self.info["fps"]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

        pred_result = self._result_q.get()
        pred_ts     = float("inf")
        pred_ids    = None
        if not isinstance(pred_result, str):
            pred_result, pred_ts, pred_ids = pred_result

        while True:
            track = self._track_q.get()
            data  = self._frame_q.get()

            if isinstance(pred_result, str) and data == "DONE":
                break

            frame, ms = data

            if self.show_time:
                frame = self._draw_timestamp(frame, ms)

            if ms - pred_ts + 0.5 > 0:
                boxes  = pred_result.bbox
                scores = pred_result.get_field("scores")
                ids    = pred_ids

                pred_result = self._result_q.get()
                if not isinstance(pred_result, str):
                    pred_result, pred_ts, pred_ids = pred_result
                else:
                    pred_ts = float("inf")
            else:
                boxes, ids = track
                scores     = None

            if boxes is not None:
                # ── Re-ID: run BEFORE action dict update and rendering ────
                # frame is BGR (from cv2.VideoCapture) — correct for Re-ID.
                self._run_reid(frame, boxes, ids)
                self._update_action_dict(scores, ids)
                mask  = self._render_labels(frame, boxes, ids)
                frame = self._blend(frame, mask)

            out.write(frame)
            self._done_q.put(True)

        out.release()
        tqdm.write("Output video written.")

    # ── Drawing helpers ───────────────────────────────────────────────────

    def _update_action_dict(self, scores, ids) -> None:
        """
        Cache action label scores keyed by *persistent* person ID.

        By keying on persistent ID (not tracker ID), the action history
        is preserved when a person re-enters the frame with a new
        tracker ID — the Re-ID module maps the new tracker ID back to
        the same persistent number, so the cached labels are found.
        """
        if scores is None:
            return

        for score, tid_val in zip(scores, ids):
            tid = int(tid_val) if not isinstance(tid_val, int) else tid_val

            # Skip detections that were filtered out as non-person boxes
            if self._valid_tracker_ids and tid not in self._valid_tracker_ids:
                continue

            # Convert tracker ID to persistent person number
            pid = self._persistent_id_for(tid)

            above = torch.nonzero(
                score >= self.thresh, as_tuple=False
            ).squeeze(1).tolist()

            captions, colors = [], []
            for cid in above:
                if cid in self.excl_ids:
                    continue
                label = self.cate_list[cid]
                captions.append(f"{label} {score[cid]:.2f}")
                if cid < self.cat_split[0]:
                    colors.append(0)
                elif cid < self.cat_split[1]:
                    colors.append(1)
                else:
                    colors.append(2)

            self._action_dict[pid] = {"captions": captions, "colors": colors}

    def _render_labels(
        self,
        frame_bgr: np.ndarray,
        boxes,
        ids,
    ) -> Image.Image:
        """
        Build a transparent RGBA overlay for one frame containing:

        1. Per-person coloured bounding box.
           The colour is determined by the persistent person number and is
           stable across the entire video — you can see at a glance that
           "Person 1" stayed green when they re-entered the frame.

        2. Person ID badge at the top-left corner of each box.
           Format: "Person 1"  (always visible, even with no action label).

        3. Action captions above each box (unchanged from original).
           Looked up by persistent ID so they survive tracker resets.
        """
        canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw   = ImageDraw.Draw(canvas)

        tracker_ids = _ids_to_list(ids)
        box_list    = _boxes_to_list(boxes)

        # ── Pass 1: boxes + person ID badges ─────────────────────────────
        for box, tid in zip(box_list, tracker_ids):
            # Skip detections that failed box filtering
            if self._valid_tracker_ids and tid not in self._valid_tracker_ids:
                continue

            pid       = self._persistent_id_for(tid)
            label_str = self._current_labels.get(tid, f"Person {pid}")
            color_rgb = person_color_rgb(pid)

            x1, y1, x2, y2 = [round(v) for v in box]

            # Coloured bounding box
            draw.rectangle(
                (x1, y1, x2, y2),
                outline=color_rgb + (255,),
                width=self.box_width,
            )

            # Person ID badge — solid filled rectangle at top-left of box
            bw, bh = _text_size(draw, label_str, self.badge_font)
            pad    = max(self.font_size // 4, 2)
            draw.rectangle(
                (x1, y1, x1 + bw + pad * 2, y1 + bh + pad * 2),
                fill=color_rgb + (self.badge_alpha,),
            )
            draw.text(
                (x1 + pad, y1 + pad),
                label_str,
                fill=(255, 255, 255, 255),
                font=self.badge_font,
            )

        # ── Pass 2: action captions above box ────────────────────────────
        for box, tid in zip(box_list, tracker_ids):
            # Skip detections that failed box filtering
            if self._valid_tracker_ids and tid not in self._valid_tracker_ids:
                continue

            pid  = self._persistent_id_for(tid)
            info = self._action_dict.get(pid)
            if not info or not info["captions"]:
                continue

            x1, y1 = round(box[0]), round(box[1])
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            td      = ImageDraw.Draw(overlay)

            sizes          = [_text_size(td, c, self.font) for c in info["captions"]]
            widths, heights = zip(*sizes)
            max_h          = max(heights)
            rec_h          = int(round(1.8 * max_h))
            gap            = int(round(0.2 * max_h))
            pad            = max(self.font_size // 2, 1)
            total_h        = (rec_h + gap) * (len(info["captions"]) - 1) + rec_h
            start_y        = max(y1 - total_h, gap)

            for i, caption in enumerate(info["captions"]):
                rx1   = x1
                ry1   = start_y + (rec_h + gap) * i
                color = self.ACTION_COLORS[info["colors"][i]] + (self.label_alpha,)
                td.rectangle(
                    (rx1, ry1, rx1 + widths[i] + pad * 2, ry1 + rec_h),
                    fill=color,
                )
                td.text(
                    (rx1 + pad, ry1 + round((rec_h - heights[i]) / 2)),
                    caption,
                    fill=(255, 255, 255, self.label_alpha),
                    font=self.font,
                    align="center",
                )
            canvas = Image.alpha_composite(canvas, overlay)

        return canvas

    def _draw_timestamp(self, frame: np.ndarray, ms: float) -> np.ndarray:
        img     = Image.fromarray(frame[..., ::-1]).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)

        text   = _ms_to_timestamp(ms)
        tw, th = _text_size(draw, text, self.font)
        pad    = max(self.font_size // 2, 1)
        rec_h  = int(round(1.8 * th))
        y0     = img.height - rec_h

        draw.rectangle((0, y0, tw + pad * 2, img.height),
                        fill=(0, 0, 0, self.label_alpha))
        draw.text((pad, y0 + round((rec_h - th) / 2)), text,
                  fill=(255, 255, 255, self.label_alpha),
                  font=self.font, align="center")

        merged = Image.alpha_composite(img, overlay).convert("RGB")
        return np.array(merged)[..., ::-1]

    def _blend(self, frame: np.ndarray, mask: Image.Image) -> np.ndarray:
        """Alpha-composite *mask* onto *frame* (BGR numpy array)."""
        img    = Image.fromarray(frame[..., ::-1]).convert("RGBA")
        merged = Image.alpha_composite(img, mask).convert("RGB")
        return np.array(merged)[..., ::-1]
