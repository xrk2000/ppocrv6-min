# -*- coding: utf-8 -*-
"""Test configuration.

The tiny det/rec models are bundled in the package, so integration tests run
by default. Override with env vars to point at other models/images:
  PPOCR_TEST_DET_DIR, PPOCR_TEST_REC_DIR, PPOCR_TEST_IMAGE
"""
import os

import pytest

from ppocrv6_min.paths import BUNDLED_DET_DIR, BUNDLED_REC_DIR

DET_DIR = os.environ.get("PPOCR_TEST_DET_DIR", BUNDLED_DET_DIR)
REC_DIR = os.environ.get("PPOCR_TEST_REC_DIR", BUNDLED_REC_DIR)
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
