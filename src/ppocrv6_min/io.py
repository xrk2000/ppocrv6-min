# -*- coding: utf-8 -*-
"""Small shared helpers (unicode-safe image IO)."""
from __future__ import annotations

import os

import cv2
import numpy as np


def imread_unicode(path: str) -> np.ndarray:
    """Read an image as a BGR uint8 array.

    Uses ``np.fromfile`` + ``cv2.imdecode`` because ``cv2.imread`` fails on
    non-ASCII paths (e.g. Chinese file names) on Windows.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    """Write an image to ``path`` (unicode-safe)."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"cannot encode image for {path}")
    buf.tofile(path)
