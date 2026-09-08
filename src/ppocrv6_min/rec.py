# -*- coding: utf-8 -*-
"""CTC text-line recognizer (PP-OCRv6 *_rec ONNX).

Pre/post processing mirrors the PaddleX ``inference.yml`` of the rec model:
  PreProcess : resize to H=48 (W proportional, capped at 320), right-pad to 320,
               (img/255 - 0.5) / 0.5 -> NCHW float32
  Net        : x[1,3,48,Wp] -> probs[1,T,C]   (softmax already inside the graph)
  PostProcess: greedy CTC decode, labels = ['blank'] + character_dict + [' ']

The character dict is parsed from ``inference.yml`` in the model directory
(``character_dict``, one char per ``- `` line). No YAML library is needed.
"""
from __future__ import annotations

import os
import re
import time
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort

# params from inference.yml (RecResizeImg image_shape: [3, 48, 320])
IMG_H = 48
IMG_W_PAD = 320
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _unquote(item: str) -> str:
    """Unquote one YAML scalar as written in PaddleX ``inference.yml``.

    Handles the two quote styles that appear in ``character_dict``:
    * single-quoted: ``''''`` is the YAML spelling of one ``'`` character
      (``''`` escapes a quote inside single-quoted scalars);
    * double-quoted: ``\\"...\\\"`` is treated as a literal (the dict only
      contains single characters, no backslash escapes are used).
    """
    s = item.strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1].replace("''", "'")
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def load_dict(yml_path: str) -> List[str]:
    """Parse ``character_dict`` (one char per ``- `` line) from a PaddleX yml.

    No YAML library is needed: the section is a flat list of single-character
    scalars, so a line-by-line parse plus :func:`_unquote` is exact.
    """
    lines = open(yml_path, encoding="utf-8").read().splitlines()
    try:
        start = next(n for n, l in enumerate(lines) if "character_dict:" in l)
    except StopIteration:
        raise ValueError(f"no character_dict found in {yml_path}")
    items: List[str] = []
    n = start + 1
    while n < len(lines):
        l = lines[n]
        m = re.match(r"\s*-(.*)", l)
        if not m:
            break
        items.append(_unquote(m.group(1)))
        n += 1
    return items


class RecModel:
    """Text-line recognizer.

    Args:
        model_dir: directory holding ``inference.onnx`` and ``inference.yml``.
        charset: optional explicit list of characters; by default it is read
            from ``<model_dir>/inference.yml``.
        providers: onnxruntime execution providers.
    """

    def __init__(self, model_dir: str, charset: Optional[List[str]] = None,
                 providers=None):
        onnx_path = os.path.join(model_dir, "inference.onnx")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(onnx_path)
        if charset is None:
            charset = load_dict(os.path.join(model_dir, "inference.yml"))
        self.dict = list(charset)
        # PaddleOCR v3 CTC label layout: blank + dict + space
        self.labels = ["blank"] + self.dict + [" "]
        self.sess = ort.InferenceSession(
            onnx_path, providers=providers or ["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        n_out = self.sess.get_outputs()[0].shape[-1]
        if n_out != len(self.labels):
            raise ValueError(
                f"model output has {n_out} classes but the dict defines "
                f"{len(self.labels)} (dict={len(self.dict)})")

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        new_w = int(round(w / h * IMG_H))
        new_w = max(1, min(new_w, IMG_W_PAD))
        im = cv2.resize(img_bgr, (new_w, IMG_H), interpolation=cv2.INTER_LINEAR)
        arr = im.astype(np.float32)
        arr = (arr / 255.0 - MEAN) / STD
        if new_w < IMG_W_PAD:  # right pad (RecResizeImg behaviour)
            pad = np.zeros((IMG_H, IMG_W_PAD - new_w, 3), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        return arr.transpose(2, 0, 1)[None]  # NCHW

    def _ctc_decode(self, probs: np.ndarray):
        """Greedy CTC: (T, C) probs -> (text, mean per-step confidence)."""
        idx = probs.argmax(axis=1)
        chars = []
        prev = -1
        for i in idx:
            if i != prev:
                if i != 0:  # skip CTC blank; the last class is a real space
                    chars.append(self.labels[i])
            prev = i
        merged = []
        for c in chars:
            if not merged or merged[-1] != c:
                merged.append(c)
        avg = float(probs.max(axis=1).mean())
        return "".join(merged), avg

    def predict_line(self, img_rgb: np.ndarray):
        """Recognize a single horizontal text-line crop (HxWx3 RGB uint8).

        Returns:
            (text, score, elapsed_seconds, (width, height))
        """
        img_bgr = img_rgb[:, :, ::-1]  # model is trained on BGR
        t0 = time.time()
        x = self.preprocess(img_bgr)
        probs = self.sess.run(None, {self.in_name: x})[0]
        dt = time.time() - t0
        text, score = self._ctc_decode(probs[0].astype(np.float32))
        return text, score, dt, (img_rgb.shape[1], img_rgb.shape[0])
