# -*- coding: utf-8 -*-
"""End-to-end OCR pipeline: DB detection -> perspective warp -> CTC recognition."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .det import DetModel
from .io import imread_unicode
from .rec import RecModel


@dataclass
class TextItem:
    text: str
    score: float
    box: tuple  # (x0, y0, x1, y1) axis-aligned bbox of the quad, int
    quad: np.ndarray  # (4, 2) float32 in original image coords


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order four points to tl, tr, br, bl (sum/diff corner method)."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl])


def warp_quad(img: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
    """Perspective-warp a 4-point quad into a horizontal strip (same channel layout).

    The strip width comes from the longer of the two top/bottom edges; a
    minimum height of 8 px keeps the recognizer stable on very thin lines.
    """
    q = order_quad(quad)
    w1 = float(np.linalg.norm(q[0] - q[1]))
    w2 = float(np.linalg.norm(q[3] - q[2]))
    h1 = float(np.linalg.norm(q[0] - q[3]))
    h2 = float(np.linalg.norm(q[1] - q[2]))
    W = int(max(w1, w2))
    H = max(int(max(h1, h2)), 8)
    if W < 2:
        return None
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (W, H),
                               flags=cv2.INTER_LINEAR)


class OCR:
    """PP-OCRv6 detect + recognize pipeline.

    Args:
        det_dir: directory of the ``*_det_onnx`` model.
        rec_dir: directory of the ``*_rec_onnx`` model.
        box_thresh: detection ``box_thresh`` (higher = fewer, stricter boxes).
        limit_side_len: detection long-side target.
        use_det: disable detection to feed the whole image to the recognizer
            (useful for a single already-cropped line).
        providers: onnxruntime providers passed to both models.
    """

    def __init__(self, det_dir: str, rec_dir: str, box_thresh: float = 0.4,
                 limit_side_len: int = 960, use_det: bool = True,
                 providers=None):
        self.det = DetModel(det_dir, limit_side_len=limit_side_len,
                            box_thresh=box_thresh, providers=providers) \
            if use_det else None
        self.rec = RecModel(rec_dir, providers=providers)
        self.use_det = use_det

    def detect(self, img_bgr: np.ndarray):
        if self.det is None:
            h, w = img_bgr.shape[:2]
            quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            return [(quad, 1.0)]
        return self.det.detect(img_bgr)

    def predict(self, image: "str | np.ndarray") -> List[TextItem]:
        """Run the full pipeline on a path or BGR ndarray. Returns items
        sorted top-to-bottom, left-to-right."""
        if isinstance(image, str):
            img_bgr = imread_unicode(image)
        else:
            img_bgr = np.asarray(image)
        img_rgb = img_bgr[:, :, ::-1]
        t_start = time.time()
        found = self.detect(img_bgr)
        items: List[TextItem] = []
        for quad, score in found:
            strip = warp_quad(img_rgb, quad)
            if strip is None:
                continue
            text, rec_score, _dt, _sz = self.rec.predict_line(strip)
            x0 = int(np.min(quad[:, 0])); y0 = int(np.min(quad[:, 1]))
            x1 = int(np.max(quad[:, 0])); y1 = int(np.max(quad[:, 1]))
            items.append(TextItem(text=text, score=rec_score,
                                  box=(x0, y0, x1, y1), quad=quad))
        items.sort(key=lambda it: (it.box[1], it.box[0]))
        return items

    def predict_file(self, path: str) -> List[TextItem]:
        return self.predict(path)
