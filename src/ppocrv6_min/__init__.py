# -*- coding: utf-8 -*-
"""ppocrv6_min: minimal-dependency PP-OCRv6 inference (det + rec).

Only numpy / onnxruntime / opencv-python / pyclipper are required.
"""
from .det import DetModel
from .pipeline import OCR
from .paths import (BUNDLED_DET_DIR, BUNDLED_REC_DIR, bundled_available,
                    default_det_dir, default_rec_dir)
from .rec import RecModel

__version__ = "0.1.0"
__all__ = [
    "OCR", "DetModel", "RecModel",
    "BUNDLED_DET_DIR", "BUNDLED_REC_DIR", "bundled_available",
    "default_det_dir", "default_rec_dir",
    "__version__",
]
