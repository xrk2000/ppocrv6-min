# -*- coding: utf-8 -*-
"""ppocrv6_min: minimal-dependency PP-OCRv6 inference (det + rec).

Only numpy / onnxruntime / opencv-python / pyclipper are required.
"""
from .det import DetModel
from .pipeline import OCR
from .rec import RecModel

__version__ = "0.1.0"
__all__ = ["OCR", "DetModel", "RecModel", "__version__"]
