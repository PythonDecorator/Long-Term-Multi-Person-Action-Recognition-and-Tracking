"""
VideoDetectionLoader — modernised, threading-based.

Reads frames from a video file or webcam, runs person detection/tracking,
and feeds results into two queues consumed by downstream workers.

Compatible with Python 3.9+, PyTorch 2.x.
No multiprocessing — uses threading throughout to avoid POSIX semaphore
issues in Docker / Colab environments.
"""

from __future__ import annotations

import queue
import threading
from itertools import count
from time import sleep
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm

from detector.apis import get_detector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SharedValue:
    """Thread-safe integer value — replaces mp.Value('i', ...)."""

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock  = threading.Lock()

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    @value.setter
    def value(self, v: int) -> None:
        with self._lock:
            self._value = v


class Resize:
    """Aspect-ratio-preserving resize with an optional max-size cap."""

    def __init__(self, min_size: int, max_size: Optional[int]) -> None:
        self.min_size = min_size
        self.max_size = max_size

    def _target_size(self, w: int, h: int) -> Tuple[int, int]:
        size     = self.min_size
        max_size = self.max_size
        if max_size is not None:
            lo, hi = float(min(w, h)), float(max(w, h))
            if hi / lo * size > max_size:
                size = int(round(max_size * lo / hi))
        if (w <= h and w == size) or (h <= w and h == size):
            return h, w
        if w < h:
            return int(size * h / w), size
        return size, int(size * w / h)

    def __call__(self, image):          # PIL image
        h, w  = self._target_size(*image.size)
        return F.resize(image, (h, w))


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class VideoDetectionLoader:
    """
    Streams frames from *input_path*, runs a person detector on each frame,
    and pushes results onto two queues:

    * ``track_queue``  — (orig_img, boxes, scores, ids) for the main thread
    * ``action_queue`` — (frame_data, video_size) for the predictor worker
    """

    def __init__(
        self,
        cfg,
        track_queue:  queue.Queue,
        action_queue: queue.Queue,
        predictor_process: SharedValue,
    ) -> None:
        self.cfg        = cfg
        self.input_path = cfg.input_path
        self.start_ms   = cfg.start
        self.dur_ms     = cfg.duration
        self.realtime   = cfg.realtime
        self.detector   = None          # lazy — created inside worker thread

        # Read video metadata once
        cap = cv2.VideoCapture(self.input_path)
        assert cap.isOpened(), f"Cannot open video source: {self.input_path}"
        self.videoinfo = {
            "fourcc":    int(cap.get(cv2.CAP_PROP_FOURCC)),
            "fps":       cap.get(cv2.CAP_PROP_FPS),
            "frameSize": (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        }
        cap.release()

        self._stop       = threading.Event()
        self.track_queue = track_queue
        self.action_queue= action_queue
        self.predictor   = predictor_process

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> "VideoDetectionLoader":
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="VDL-worker"
        )
        self._worker.start()
        return self

    def terminate(self) -> None:
        self._stop.set()
        self._worker.join(timeout=10)
        _drain(self.track_queue)
        _drain(self.action_queue)

    # ── Public read helpers ───────────────────────────────────────────────

    def read_track(self):
        return self.track_queue.get()

    # ── Internal ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main loop: read frames → detect → push to queues."""
        cv2.setNumThreads(0)
        torch.set_num_threads(1)

        self.detector = get_detector(self.cfg)

        cap = cv2.VideoCapture(self.input_path)
        assert cap.isOpened(), f"Cannot open video source: {self.input_path}"
        if not self.realtime:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_ms)

        prev_ms = 0.0

        for frame_i in count():
            if self._stop.is_set():
                break
            if self.realtime and self.predictor.value == -1:
                break
            if self.track_queue.full():
                sleep(0.001)
                continue

            ok, frame = cap.read()
            prev_ms   = cur_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            if not ok or (
                not self.realtime
                and self.dur_ms != -1
                and cur_ms > self.start_ms + self.dur_ms
            ):
                # Signal end of stream
                self._put(self.track_queue,  (None, None, None, None))
                self._put(self.action_queue, ("Done", self.videoinfo["frameSize"]))
                self._await_predictor(prev_ms)
                break

            # Preprocess & detect
            img_t = self.detector.image_preprocess(frame)
            if isinstance(img_t, np.ndarray):
                img_t = torch.from_numpy(img_t)
            if img_t.dim() == 3:
                img_t = img_t.unsqueeze(0)

            orig_img   = frame[:, :, ::-1]                          # BGR→RGB view
            im_dim     = torch.FloatTensor(
                (frame.shape[1], frame.shape[0])
            ).repeat(1, 2)

            with torch.no_grad():
                det_result = self._detect(img_t, orig_img, im_dim)

            self._postprocess(det_result, frame, cur_ms, frame_i)

        cap.release()

    def _detect(
        self,
        img:      torch.Tensor,
        orig_img: np.ndarray,
        im_dim:   torch.Tensor,
    ):
        with torch.no_grad():
            dets = self.detector.images_detection(img, im_dim)

        if isinstance(dets, int) or dets.shape[0] == 0:
            return orig_img, None, None, None

        if isinstance(dets, np.ndarray):
            dets = torch.from_numpy(dets)
        dets = dets.cpu()

        mask   = dets[:, 0] == 0
        boxes  = dets[mask, 1:5]
        scores = dets[mask, 5:6]
        ids    = dets[mask, 6:7]

        if boxes.shape[0] == 0:
            return orig_img, None, None, None

        return orig_img, boxes, scores, ids

    def _postprocess(
        self,
        detection,
        raw_frame: np.ndarray,
        cur_ms:    float,
        frame_i:   int,
    ) -> None:
        orig_img, boxes, scores, ids = detection

        if orig_img is None or self._stop.is_set():
            self._put(self.track_queue, (None, None, None, None))
            return

        self._put(
            self.action_queue,
            ((raw_frame, cur_ms, boxes, scores, ids), self.videoinfo["frameSize"]),
        )
        self._put(self.track_queue, (orig_img, boxes, scores, ids))

    def _put(self, q: queue.Queue, item) -> None:
        if not self._stop.is_set():
            q.put(item)

    def _await_predictor(self, last_ms: float) -> None:
        """Show a progress bar while the predictor worker catches up."""
        pred_val = self.predictor.value
        if pred_val == -1:
            return

        tqdm.write("Tracking done. Waiting for feature extraction…")
        initial  = max(pred_val - self.start_ms, 0)
        total    = max(int(last_ms) - self.start_ms, 1)
        pbar     = tqdm(total=total, initial=initial, desc="Feature Extraction")
        last_pos = initial

        while self.predictor.value != -1:
            cur = self.predictor.value
            pbar.update(cur - last_pos)
            last_pos = cur
            if self._stop.is_set():
                break
            sleep(0.1)

        pbar.update(total - last_pos)
        pbar.close()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _drain(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break
