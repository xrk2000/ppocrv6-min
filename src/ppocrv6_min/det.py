# -*- coding: utf-8 -*-
"""DB text-line detector (PP-OCRv6 *_det ONNX).

Post-processing follows PaddleOCR ``ppocr/postprocess/db_postprocess.py``
exactly: threshold -> contours -> minAreaRect -> polygon-mean score ->
pyclipper ``unclip`` expansion.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper

# ImageNet stats (PaddleX inference.yml: NormalizeImage)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DBPostProcess:
    """Contour -> scored unclipped quads, identical to PaddleOCR's DBPostProcess."""

    def __init__(self, thresh: float = 0.2, box_thresh: float = 0.4,
                 unclip_ratio: float = 1.4, max_candidates: int = 3000,
                 min_size: int = 3):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates
        self.min_size = min_size

    def unclip(self, box, ratio: float):
        box = np.asarray(box, dtype=np.float64)
        x = box[:, 0]
        y = box[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        closed = np.vstack([box, box[:1]])
        length = float(np.linalg.norm(np.diff(closed, axis=1), axis=1).sum())
        distance = area * ratio / (length + 1e-9)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return offset.Execute(distance)

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        i1, i2, i3, i4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            i1, i4 = 0, 1
        else:
            i1, i4 = 1, 0
        if points[3][1] > points[2][1]:
            i2, i3 = 2, 3
        else:
            i2, i3 = 3, 2
        box = [points[i1], points[i2], points[i3], points[i4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, pred: np.ndarray, box) -> float:
        h, w = pred.shape
        box = np.array(box, dtype=np.float32).copy()
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return float(cv2.mean(pred[ymin:ymax + 1, xmin:xmax + 1], mask)[0])

    def __call__(self, pred: np.ndarray, dest_w: int, dest_h: int):
        """``pred``: (H, W) probability map at network output resolution."""
        mask = (pred > self.thresh).astype(np.uint8)
        contours, _ = cv2.findContours(mask * 255, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        h, w = pred.shape
        boxes, scores = [], []
        for contour in contours[: self.max_candidates]:
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points)
            if self.box_thresh > score:
                continue
            box = self.unclip(points, self.unclip_ratio)
            if len(box) > 1:
                continue
            box = np.array(box).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box, dtype="int32")
            box[:, 0] = np.clip(np.round(box[:, 0] / w * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * dest_h), 0, dest_h)
            boxes.append(box)
            scores.append(score)
        return boxes, scores


class DetModel:
    """DB text-line detector.

    Args:
        model_dir: directory holding ``inference.onnx`` (PaddleX det export).
        limit_side_len: long-side target for ``DetResizeForTest`` (default 960;
            the long side is never up-scaled beyond the original).
        thresh / box_thresh / unclip_ratio: DBPostProcess parameters
            (defaults match the PaddleX ``inference.yml`` of PP-OCRv6_tiny_det).
        providers: onnxruntime execution providers.
    """

    def __init__(self, model_dir: str, limit_side_len: int = 960,
                 thresh: float = 0.2, box_thresh: float = 0.4,
                 unclip_ratio: float = 1.4, providers=None):
        onnx_path = os.path.join(model_dir, "inference.onnx")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(onnx_path)
        self.limit = limit_side_len
        self.post = DBPostProcess(thresh=thresh, box_thresh=box_thresh,
                                  unclip_ratio=unclip_ratio)
        self.sess = ort.InferenceSession(
            onnx_path, providers=providers or ["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def _resize(self, h: int, w: int):
        scale = min(self.limit / max(h, w), 1.0)
        nh = max(32, (int(h * scale) // 32) * 32)
        nw = max(32, (int(w * scale) // 32) * 32)
        return nh, nw

    def detect(self, img_bgr: np.ndarray):
        """Detect text lines.

        Args:
            img_bgr: HxWx3 uint8 BGR image.

        Returns:
            list of ``(points, score)`` where ``points`` is a (4, 2) float32
            quad (tl, tr, br, bl order is NOT guaranteed) in original image
            coordinates and ``score`` is the mean probability inside the box.
        """
        h, w = img_bgr.shape[:2]
        nh, nw = self._resize(h, w)
        im = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = im.astype(np.float32)
        x = (x / 255.0 - MEAN) / STD
        x = x.transpose(2, 0, 1)[None]
        prob = self.sess.run(None, {self.in_name: x})[0][0, 0]  # (nh, nw)
        boxes, scores = self.post(prob, w, h)
        return [(b.astype(np.float32), float(s)) for b, s in zip(boxes, scores)]
