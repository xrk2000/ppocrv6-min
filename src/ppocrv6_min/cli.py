# -*- coding: utf-8 -*-
"""Command line entry: ``ppocrv6-min``.

Examples:
  ppocrv6-min screenshot.png
  ppocrv6-min --no-det single_line.png
  ppocrv6-min --box-thresh 0.3 --json out.json a.png b.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import __version__
from .pipeline import OCR


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ppocrv6-min",
        description="Minimal PP-OCRv6 OCR (det + rec) on ONNX Runtime.")
    p.add_argument("images", nargs="+", help="image file(s) to OCR")
    p.add_argument("--det-dir",
                   default=os.environ.get(
                       "PPOCR_DET_DIR",
                       os.path.join(os.path.expanduser("~"),
                                    "models", "PP-OCRv6",
                                    "PP-OCRv6_tiny_det_onnx")),
                   help="directory of the *_det_onnx model "
                        "(env PPOCR_DET_DIR)")
    p.add_argument("--rec-dir",
                   default=os.environ.get(
                       "PPOCR_REC_DIR",
                       os.path.join(os.path.expanduser("~"),
                                    "models", "PP-OCRv6",
                                    "PP-OCRv6_tiny_rec_onnx")),
                   help="directory of the *_rec_onnx model "
                        "(env PPOCR_REC_DIR)")
    p.add_argument("--no-det", action="store_true",
                   help="skip detection; feed the whole image to the recognizer")
    p.add_argument("--box-thresh", type=float, default=0.4,
                   help="detection box threshold (default 0.4)")
    p.add_argument("--limit-side", type=int, default=960,
                   help="detection long-side target (default 960)")
    p.add_argument("--json", metavar="PATH",
                   help="also write results to a JSON file")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    ocr = OCR(args.det_dir, args.rec_dir, box_thresh=args.box_thresh,
              limit_side_len=args.limit_side, use_det=not args.no_det)
    all_results = {}
    for path in args.images:
        t_start = time.time()
        items = ocr.predict(path)
        dt = time.time() - t_start
        print(f"=== {path}  ({dt*1000:.0f} ms, {len(items)} lines) ===")
        all_results[path] = [
            {"text": it.text, "score": round(it.score, 4), "box": list(it.box)}
            for it in items]
        for k, it in enumerate(items, 1):
            x0, y0, x1, y1 = it.box
            print(f"  L{k:2d} ({x0:4d},{y0:3d},{x1:4d},{y1:3d})  "
                  f"{it.score:.3f}  {it.text}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.json}")
    print(f"done in {(time.time()-t0)*1000:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
