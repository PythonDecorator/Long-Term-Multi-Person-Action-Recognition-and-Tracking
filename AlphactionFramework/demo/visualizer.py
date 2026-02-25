"""
AVAVisualizer — modernised, threading-based.

Compatible with Python 3.9+, PyTorch 2.x, Pillow 10+.
Uses queue.Queue and threading.Thread throughout (no multiprocessing).
Pillow 10+ API: textbbox() replaces removed textsize().
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
        # Pillow < 10: textsize() still available
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


# ---------------------------------------------------------------------------
# AVAVisualizer
# ---------------------------------------------------------------------------

class AVAVisualizer:
    """
    Reads the original video frame-by-frame, composites bounding boxes and
    action labels, and writes the annotated video to *output_path*.

    In non-realtime mode three background threads handle:
      1. Frame loading      (_load_frames)
      2. Frame writing      (_write_frames)
    Communication happens via standard queue.Queue objects.
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

    # Colours (R, G, B) for the three action category groups
    COLORS: Tuple[Tuple[int, ...], ...] = (
        (176, 85, 234),   # movement
        (87, 118, 198),   # object interaction
        (52, 189, 199),   # social
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
            self.cate_list     = self.COMMON_CATES
            self.cat_split     = (6, 11)
        else:
            self.cate_list     = self.CATEGORIES
            self.cat_split     = (14, 63)

        self.cls2id    = {name: i for i, name in enumerate(self.cate_list)}
        excl           = exclude_class if exclude_class is not None else self.EXCLUSION
        self.excl_ids  = {self.cls2id[c] for c in excl if c in self.cls2id}

        # Drawing params
        W, H             = self.info["width"], self.info["height"]
        long_side        = min(W, H)
        self.width       = W
        self.height      = H
        self.font_size   = max(int(round(long_side / 40)), 1)
        self.box_width   = max(int(round(long_side / 180)), 1)
        self.font        = ImageFont.truetype("./Roboto-Bold.ttf", self.font_size)
        self.box_color   = (191, 40, 41)
        self.label_alpha = int(0.6 * 255)

        # Action state (shared between load/write threads)
        self._action_dict: Dict[int, Dict] = {}

        # ── Realtime output ──────────────────────────────────────────────
        if realtime:
            fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
            self._outvid = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        else:
            # ── Offline: three queues, two background threads ─────────────
            self._frame_q  = queue.Queue(maxsize=512)   # raw frames
            self._result_q = queue.Queue()              # predictions
            self._track_q  = queue.Queue()              # tracking boxes
            self._done_q   = queue.Queue()              # written frame signals

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

    # ── Realtime API ──────────────────────────────────────────────────────

    def write_realtime_frame(
        self,
        result,
        orig_img: np.ndarray,
        boxes,
        scores,
        ids,
    ) -> bool:
        """Composite and display one frame. Returns False when ESC pressed."""
        orig_img = orig_img[:, :, ::-1]   # RGB → BGR for cv2

        if result is not None:
            pred, _ts, pred_ids = result
            self._update_action_dict(pred.get_field("scores"), pred_ids)

        if boxes is not None:
            mask     = self._render_labels(boxes, ids)
            orig_img = self._blend(orig_img, mask)

        cv2.imshow("AlphAction", orig_img)
        self._outvid.write(orig_img)
        return cv2.waitKey(1) != 27

    # ── Offline queue API (called from main thread) ───────────────────────

    def send_result(self, result) -> None:
        self._result_q.put(result)

    def send_track(self, result) -> None:
        self._track_q.put(result)

    def show_progress(self, total_frames: int) -> None:
        cnt  = 0
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

        # Initialise with first prediction result
        pred_result  = self._result_q.get()
        pred_ts      = float("inf")
        pred_ids     = None
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
                # New prediction has become current
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
                self._update_action_dict(scores, ids)
                mask  = self._render_labels(boxes, ids)
                frame = self._blend(frame, mask)

            out.write(frame)
            self._done_q.put(True)

        out.release()
        tqdm.write("Output video written.")

    # ── Drawing helpers ───────────────────────────────────────────────────

    def _update_action_dict(self, scores, ids) -> None:
        if scores is None:
            return
        for score, pid in zip(scores, ids):
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

            self._action_dict[int(pid)] = {
                "captions": captions,
                "colors":   colors,
            }

    def _render_labels(self, boxes, ids) -> Image.Image:
        """Build a transparent RGBA overlay with boxes and labels."""
        canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw   = ImageDraw.Draw(canvas)

        # Draw bounding boxes
        for box in boxes:
            draw.rectangle(
                box.tolist(),
                outline=self.box_color + (255,),
                width=self.box_width,
            )

        # Draw labels above each box
        for box, pid in zip(boxes, ids):
            info = self._action_dict.get(int(pid))
            if not info or not info["captions"]:
                continue

            x1, y1 = box.tolist()[:2]
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            td      = ImageDraw.Draw(overlay)

            sizes   = [_text_size(td, c, self.font) for c in info["captions"]]
            widths, heights = zip(*sizes)
            max_h   = max(heights)
            rec_h   = int(round(1.8 * max_h))
            gap     = int(round(0.2 * max_h))
            pad     = max(self.font_size // 2, 1)
            total_h = (rec_h + gap) * (len(info["captions"]) - 1) + rec_h
            start_y = max(round(y1) - total_h, gap)

            for i, caption in enumerate(info["captions"]):
                rx1  = round(x1)
                ry1  = start_y + (rec_h + gap) * i
                color = self.COLORS[info["colors"][i]] + (self.label_alpha,)
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
        img    = Image.fromarray(frame[..., ::-1]).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw   = ImageDraw.Draw(overlay)

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
