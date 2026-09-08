# -*- coding: utf-8 -*-
"""Integration tests against the real PP-OCRv6 ONNX models (skipped if absent)."""
import numpy as np
import cv2

from conftest import DET_DIR, REC_DIR, IMAGE, _need_det, _need_rec, _need_image
from ppocrv6_min.det import DetModel
from ppocrv6_min.rec import RecModel
from ppocrv6_min.io import imread_unicode
from ppocrv6_min.pipeline import OCR


@_need_rec()
def test_rec_synthetic_line(tmp_path):
    """Recognize a clean ASCII single line with high confidence."""
    text = "PP-OCRv6 tiny rec 2026"
    img = np.full((60, 520, 3), 255, np.uint8)
    cv2.putText(img, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    rec = RecModel(REC_DIR)
    got, score, _dt, (w, h) = rec.predict_line(img[:, :, ::-1])
    assert h == 60
    assert "2026" in got
    assert score > 0.7


@_need_det()
def test_det_finds_lines():
    img = imread_unicode(IMAGE)
    det = DetModel(DET_DIR)
    found = det.detect(img)
    assert len(found) >= 3, f"expected >=3 text lines, got {len(found)}"
    for quad, score in found:
        assert quad.shape == (4, 2)
        assert 0.0 < score <= 1.0


@_need_det()
@_need_rec()
@_need_image()
def test_end_to_end_ocr():
    ocr = OCR(DET_DIR, REC_DIR)
    items = ocr.predict(IMAGE)
    assert len(items) >= 3
    # the test image is Chinese; at least the activity-deadline line should match
    joined = "".join(it.text for it in items)
    assert "2026" in joined
    assert "奖券" in joined or "活动" in joined
    # boxes must be within the image
    img = imread_unicode(IMAGE)
    H, W = img.shape[:2]
    for it in items:
        x0, y0, x1, y1 = it.box
        assert 0 <= x0 < x1 <= W
        assert 0 <= y0 < y1 <= H
