"""
AVAPredictor + AVAPredictorWorker — modernised, threading-based.

Compatible with Python 3.9+, PyTorch 2.x.
No multiprocessing — uses threading throughout.

Key changes vs original:
  - mp.Queue / mp.Process → queue.Queue / threading.Thread
  - Lazy model initialisation inside the worker thread
  - All feature tensors explicitly cast to float32 on GPU transfer
  - BoxList fields correctly cast (avoid conv3d dtype errors in PyTorch 2.x)
  - Timestamps stored as int (OpenCV returns float ms; range() needs int)
"""

from __future__ import annotations

import copy
import queue
import threading
from bisect import bisect_right
from itertools import count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from alphaction.config import cfg as base_cfg
from alphaction.dataset.collate_batch import batch_different_videos
from alphaction.dataset.transforms import object_transforms as OT
from alphaction.dataset.transforms import video_transforms as T
from alphaction.modeling.detector import build_detection_model
from alphaction.structures.bounding_box import BoxList
from alphaction.structures.memory_pool import MemoryPool
from alphaction.utils.IA_helper import has_memory, has_object
from alphaction.utils.checkpoint import ActionCheckpointer
from detector.apis import get_detector
from video_detection_loader import SharedValue, VideoDetectionLoader, _drain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(v: Any, device: torch.device) -> Any:
    """
    Move *v* to *device* as float32.

    Handles:
      - torch.Tensor           → .to(device).float()
      - BoxList                → rebuild with all tensor fields cast to float32
      - anything with .to()   → .to(device)  (no dtype coercion)
      - everything else        → returned unchanged
    """
    if isinstance(v, torch.Tensor):
        return v.to(device).float()

    if isinstance(v, BoxList):
        new_bl = BoxList(v.bbox.to(device).float(), v.size, v.mode)
        for name, val in v.extra_fields.items():
            if isinstance(val, torch.Tensor):
                new_bl.add_field(name, val.to(device).float())
            elif hasattr(val, "to"):
                new_bl.add_field(name, val.to(device))
            else:
                new_bl.add_field(name, val)
        return new_bl

    if hasattr(v, "to"):
        return v.to(device)

    return v


# ---------------------------------------------------------------------------
# AVAPredictor — wraps the action recognition model
# ---------------------------------------------------------------------------

class AVAPredictor:
    """
    Builds and runs the AlphAction model.

    The model is initialised lazily inside the worker thread to avoid
    CUDA context issues when the main process hasn't set one up yet.
    """

    def __init__(
        self,
        cfg_path:     str,
        weight_path:  str,
        detect_rate:  int,
        common_cate:  bool,
        device:       torch.device,
    ) -> None:
        cfg = base_cfg.clone()
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.WEIGHT                 = weight_path
        cfg.MODEL.IA_STRUCTURE.MEMORY_RATE *= detect_rate
        if common_cate:
            cfg.MODEL.ROI_ACTION_HEAD.NUM_CLASSES                      = 15
            cfg.MODEL.ROI_ACTION_HEAD.NUM_PERSON_MOVEMENT_CLASSES      = 6
            cfg.MODEL.ROI_ACTION_HEAD.NUM_OBJECT_MANIPULATION_CLASSES  = 5
            cfg.MODEL.ROI_ACTION_HEAD.NUM_PERSON_INTERACTION_CLASSES   = 4
        cfg.freeze()
        self.cfg = cfg

        self.has_memory = has_memory(cfg.MODEL.IA_STRUCTURE)
        self.mem_len    = cfg.MODEL.IA_STRUCTURE.LENGTH
        self.mem_rate   = cfg.MODEL.IA_STRUCTURE.MEMORY_RATE
        self.has_object = has_object(cfg.MODEL.IA_STRUCTURE)

        self.device     = device
        self.cpu_device = torch.device("cpu")

        # Feature stores
        self.mem_pool      = MemoryPool()
        self.object_pool   = MemoryPool()
        self.mem_timestamps: List[int] = []
        self.obj_timestamps: List[int] = []
        self.pred_pos = 0

        self.transforms, self.person_transforms, self.object_transforms = (
            self._build_transforms()
        )

        self.model = None   # initialised lazily in worker thread

    # ── Lazy init ─────────────────────────────────────────────────────────

    def ensure_model(self) -> None:
        if self.model is not None:
            return
        self.model = build_detection_model(self.cfg)
        self.model.eval()
        ckpt = ActionCheckpointer(self.cfg, self.model)
        print(f"Loading weights: {self.cfg.MODEL.WEIGHT}")
        ckpt.load(self.cfg.MODEL.WEIGHT)
        self.model.to(self.device)
        print("Model ready.")

    # ── Transforms ────────────────────────────────────────────────────────

    def _build_transforms(self):
        cfg = self.cfg
        video_tf = T.Compose([
            T.TemporalCrop(cfg.INPUT.FRAME_NUM, cfg.INPUT.FRAME_SAMPLE_RATE),
            T.Resize(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST),
            T.ToTensor(),
            T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN,
                std=cfg.INPUT.PIXEL_STD,
                to_bgr=cfg.INPUT.TO_BGR,
            ),
            T.SlowFastCrop(cfg.INPUT.TAU, cfg.INPUT.ALPHA, False),
        ])
        person_tf = OT.Resize()
        object_tf = OT.Compose([
            OT.PickTop(cfg.MODEL.IA_STRUCTURE.MAX_OBJECT),
            OT.Resize(),
        ])
        return video_tf, person_tf, object_tf

    # ── Feature update ────────────────────────────────────────────────────

    def update_feature(
        self,
        video_data,
        boxes,
        objects,
        timestamp: int,
        transform_randoms,
    ) -> None:
        """Extract and cache features for *timestamp*."""
        self.ensure_model()

        if self.mem_timestamps:
            assert timestamp > self.mem_timestamps[-1], \
                "Features must be updated in chronological order."

        slow = batch_different_videos(
            [video_data[0]], self.cfg.DATALOADER.SIZE_DIVISIBILITY
        ).to(self.device)
        fast = batch_different_videos(
            [video_data[1]], self.cfg.DATALOADER.SIZE_DIVISIBILITY
        ).to(self.device)
        boxes_gpu = [self.person_transforms(boxes, transform_randoms).to(self.device)]
        objs_gpu  = (
            [self.object_transforms(objects, transform_randoms).to(self.device)]
            if objects is not None else None
        )

        with torch.no_grad():
            feat = self.model(slow, fast, boxes_gpu, objs_gpu, part_forward=0)
            p_feat = [f.to(self.cpu_device) for f in feat[0]][0]
            o_feat = (
                None if feat[1] is None
                else [f.to(self.cpu_device) for f in feat[1]][0]
            )

        self.mem_pool["SingleVideo", timestamp] = p_feat
        self.mem_timestamps.append(timestamp)
        if o_feat is not None:
            self.object_pool["SingleVideo", timestamp] = o_feat
            self.obj_timestamps.append(timestamp)

    # ── Ready-to-predict count ────────────────────────────────────────────

    def n_ready(self) -> int:
        if not self.mem_timestamps:
            return 0
        if self.has_memory:
            before, after = self.mem_len
            last_ready = self.mem_timestamps[-1] - after * self.mem_rate
            return bisect_right(self.mem_timestamps, last_ready) - self.pred_pos
        return len(self.mem_timestamps) - self.pred_pos

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, timestamp: int, vid_size: Tuple[int, int]):
        """Run the second model pass (attention + classification)."""
        self.ensure_model()

        td = lambda v: _to_device(v, self.device)   # noqa: E731

        current_p = [td(self.mem_pool["SingleVideo", timestamp])]
        current_o = (
            [td(self.object_pool["SingleVideo", timestamp])]
            if ("SingleVideo", timestamp) in self.object_pool else None
        )

        gpu_pool = MemoryPool()
        for k, v in self.mem_pool.items():
            gpu_pool[k] = td(v)

        extras: Dict[str, Any] = dict(
            person_pool   = gpu_pool,
            movie_ids     = ["SingleVideo"],
            timestamps    = [timestamp],
            current_feat_p= current_p,
            current_feat_o= current_o,
        )

        with torch.no_grad():
            out = self.model(None, None, None, None, extras=extras, part_forward=1)
            out = [o.resize(vid_size).to(self.cpu_device) for o in out]

        self.pred_pos += 1
        return out[0]

    # ── Feature cleanup ───────────────────────────────────────────────────

    def release_feature(self, timestamp: Optional[int] = None) -> None:
        """Free cached features up to (and including) *timestamp*.
        Pass ``None`` to clear everything."""
        if timestamp is None:
            self.mem_pool      = MemoryPool()
            self.object_pool   = MemoryPool()
            self.mem_timestamps = []
            self.obj_timestamps = []
            self.pred_pos       = 0
            return

        last_unused = (
            timestamp - self.mem_len[0] * self.mem_rate
            if self.has_memory else timestamp
        )
        n = bisect_right(self.mem_timestamps, last_unused)
        for t in self.mem_timestamps[:n]:
            del self.mem_pool["SingleVideo", t]
        self.mem_timestamps = self.mem_timestamps[n:]
        self.pred_pos       = max(self.pred_pos - n, 0)

        m = bisect_right(self.obj_timestamps, timestamp)
        for t in self.obj_timestamps[:m]:
            del self.object_pool["SingleVideo", t]
        self.obj_timestamps = self.obj_timestamps[m:]


# ---------------------------------------------------------------------------
# AVAPredictorWorker — orchestrates detection loader + predictor
# ---------------------------------------------------------------------------

class AVAPredictorWorker:
    """
    Owns the VideoDetectionLoader thread and the prediction thread.
    Exposes a simple read/write interface to the main (demo) thread.
    """

    def __init__(self, cfg) -> None:
        self.realtime    = cfg.realtime
        self.detect_rate = cfg.detect_rate
        self.interval    = 1000 // cfg.detect_rate

        self.predictor = AVAPredictor(
            cfg_path    = cfg.cfg_path,
            weight_path = cfg.weight_path,
            detect_rate = cfg.detect_rate,
            common_cate = cfg.common_cate,
            device      = cfg.device,
        )

        # Optional secondary object detector
        self.obj_det = None
        if self.predictor.has_object:
            obj_cfg          = copy.deepcopy(cfg)
            obj_cfg.detector = "yolo"
            self._obj_cfg    = obj_cfg
        else:
            self._obj_cfg = None

        # Inter-thread queues (pure Python — no shared memory)
        self.track_queue  = queue.Queue(maxsize=1)
        self.input_queue  = queue.Queue(maxsize=30)
        self.output_queue = queue.Queue()

        # Shared progress counter read by the detection loader
        self._pred_pos = SharedValue(0)

        # Start detection loader
        VideoDetectionLoader(
            cfg, self.track_queue, self.input_queue, self._pred_pos
        ).start()

        # Frame buffer
        ava_cfg = self.predictor.cfg
        self._buf_len     = ava_cfg.INPUT.FRAME_NUM * ava_cfg.INPUT.FRAME_SAMPLE_RATE
        self._center      = self._buf_len // 2
        self._frames: List = []
        self._extras: List = []
        self._timestamps: List[Tuple[int, Tuple, Any]] = []
        self._last_ms     = -2000
        self._pred_cnt    = 0
        self._vid_tf      = self.predictor.transforms

        # Thread control
        self._stop_event      = threading.Event()
        self._tracking_done   = threading.Event()

        self._thread = threading.Thread(
            target=self._predict_loop, daemon=True, name="Predictor"
        )
        self._thread.start()

    # ── Public interface ──────────────────────────────────────────────────

    def read_track(self):
        """Blocking read of a (orig_img, boxes, scores, ids) tuple."""
        return self.track_queue.get()

    def read_result(self):
        """Non-blocking read from the output queue. Returns None if empty."""
        if self._stop_event.is_set():
            return None
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def signal_tracking_done(self) -> None:
        """Call once all tracking frames have been consumed (non-realtime)."""
        assert not self.realtime
        self._tracking_done.set()

    def terminate(self) -> None:
        self._stop_event.set()
        self._pred_pos.value = -1
        self._thread.join(timeout=15)
        _drain(self.input_queue)

    # ── Prediction loop ───────────────────────────────────────────────────

    def _predict_loop(self) -> None:
        """Background thread: consume frames, extract features, predict."""
        import cv2 as _cv2
        _cv2.setNumThreads(0)
        torch.set_num_threads(1)

        if self._obj_cfg is not None:
            self.obj_det = get_detector(self._obj_cfg)

        stream_done = False
        pred_cnt    = 0

        for _ in count():
            if self._stop_event.is_set():
                return

            # ── Flush remaining predictions once stream is fully buffered ─
            if self._tracking_done.is_set() and stream_done:
                tqdm.write("Flushing remaining action predictions…")
                remaining = self._timestamps[pred_cnt:]
                for ts, vid_size, ids in tqdm(
                    remaining, initial=pred_cnt,
                    total=len(self._timestamps), desc="Action Prediction",
                ):
                    feat_idx = ts // self.interval
                    preds    = self.predictor.predict(feat_idx, vid_size)
                    self.output_queue.put((preds, ts, ids))
                    self.predictor.release_feature(feat_idx)

                self.predictor.release_feature()
                tqdm.write("Action prediction complete.")
                self.output_queue.put("done")
                self._tracking_done.clear()
                return

            # ── Read next frame batch ─────────────────────────────────────
            try:
                extra, vid_size = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue

            if extra == "Done":
                stream_done              = True
                self._pred_pos.value = -1
                continue

            frame, cur_ms, boxes, scores, ids = extra

            # Accumulate sliding window
            self._frames.append(frame)
            self._extras.append((cur_ms, boxes, scores, ids))
            self._frames = self._frames[-self._buf_len:]
            self._extras = self._extras[-self._buf_len:]

            if len(self._frames) < self._buf_len:
                continue

            ctr_ms, p_boxes, p_scores, p_ids = self._extras[self._center]

            # Throttle to detect_rate
            if ctr_ms < self._last_ms + self.interval:
                continue
            self._last_ms = ctr_ms

            if not self.realtime:
                self._pred_pos.value = int(cur_ms)

            if p_boxes is None or len(p_boxes) == 0:
                continue

            frame_arr = np.stack(self._frames)[..., ::-1]

            # Optional object detection
            obj_boxes = self._detect_objects(vid_size)

            # Build input for feature extractor
            vid_data, _, rands = self._vid_tf(frame_arr, None)
            p_box              = BoxList(p_boxes, vid_size, "xyxy").clip_to_image()
            feat_idx           = int(ctr_ms) // self.interval

            self.predictor.update_feature(vid_data, p_box, obj_boxes, feat_idx, rands)

            if self.realtime:
                preds = self.predictor.predict(feat_idx, vid_size)
                self.output_queue.put((preds, ctr_ms, p_ids[:, 0]))
                self.predictor.release_feature(feat_idx)
                pred_cnt += 1
            else:
                self._timestamps.append((int(ctr_ms), vid_size, p_ids[:, 0]))
                n_ready = self.predictor.n_ready()
                for i in range(pred_cnt, pred_cnt + n_ready):
                    ts, vs, ts_ids = self._timestamps[i]
                    fi   = ts // self.interval
                    pred = self.predictor.predict(fi, vs)
                    self.output_queue.put((pred, ts, ts_ids))
                    self.predictor.release_feature(fi)
                pred_cnt += n_ready

    def _detect_objects(self, vid_size) -> Optional[BoxList]:
        if self.obj_det is None or not self._frames:
            return None
        kframe    = self._frames[self._center]
        kdata     = self.obj_det.image_preprocess(kframe)
        im_dim    = torch.FloatTensor(
            (kframe.shape[1], kframe.shape[0])
        ).repeat(1, 2)
        dets      = self.obj_det.images_detection(kdata, im_dim)
        if isinstance(dets, int) or dets.shape[0] == 0:
            obj_t = torch.zeros((0, 4))
        else:
            obj_t = dets[:, 1:5].cpu()
        return BoxList(obj_t, vid_size, "xyxy").clip_to_image()
