# -*- coding: utf-8 -*-
"""Model directory resolution.

The two ``tiny`` ONNX models (det ~1.7 MB + rec ~4.4 MB) are bundled inside
the package under ``ppocrv6_min/models/`` so that a plain ``pip install``
works out of the box. Larger tiers (small/medium) are not bundled; point the
API or CLI at them explicitly (or via ``PPOCR_DET_DIR`` / ``PPOCR_REC_DIR``).
"""
from __future__ import annotations

import os

_BUNDLED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

BUNDLED_DET_DIR = os.path.join(_BUNDLED, "PP-OCRv6_tiny_det_onnx")
BUNDLED_REC_DIR = os.path.join(_BUNDLED, "PP-OCRv6_tiny_rec_onnx")

DET_ENV_VAR = "PPOCR_DET_DIR"
REC_ENV_VAR = "PPOCR_REC_DIR"


def default_det_dir() -> str:
    """Resolve the det model dir: $PPOCR_DET_DIR else the bundled tiny model."""
    return os.environ.get(DET_ENV_VAR, BUNDLED_DET_DIR)


def default_rec_dir() -> str:
    """Resolve the rec model dir: $PPOCR_REC_DIR else the bundled tiny model."""
    return os.environ.get(REC_ENV_VAR, BUNDLED_REC_DIR)


def bundled_available() -> bool:
    return (os.path.isfile(os.path.join(BUNDLED_DET_DIR, "inference.onnx"))
            and os.path.isfile(os.path.join(BUNDLED_REC_DIR, "inference.onnx"))
            and os.path.isfile(os.path.join(BUNDLED_REC_DIR, "inference.yml")))
