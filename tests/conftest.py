# -*- coding: utf-8 -*-
"""Test configuration: locate the local PP-OCRv6 ONNX models.

Override with env vars if your models live elsewhere:
  PPOCR_TEST_DET_DIR, PPOCR_TEST_REC_DIR, PPOCR_TEST_IMAGE
Tests are skipped automatically when the models are not present.
"""
import os

import pytest

DET_DIR = os.environ.get("PPOCR_TEST_DET_DIR",
                         r"D:\models\PP-OCRv6\PP-OCRv6_tiny_det_onnx")
REC_DIR = os.environ.get("PPOCR_TEST_REC_DIR",
                         r"D:\models\PP-OCRv6\PP-OCRv6_tiny_rec_onnx")
IMAGE = os.environ.get("PPOCR_TEST_IMAGE", r"D:\code\zmxy\教师节.png")


def _need_det():
    return pytest.mark.skipif(
        not os.path.isfile(os.path.join(DET_DIR, "inference.onnx")),
        reason="PP-OCRv6_tiny_det_onnx not found")


def _need_rec():
    return pytest.mark.skipif(
        not os.path.isfile(os.path.join(REC_DIR, "inference.onnx")),
        reason="PP-OCRv6_tiny_rec_onnx not found")


def _need_image():
    return pytest.mark.skipif(
        not os.path.isfile(IMAGE), reason="test image not found")
