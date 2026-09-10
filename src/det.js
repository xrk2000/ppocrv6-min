// det.js — DB text-line detector (PP-OCRv6 *_det ONNX), Node port of ppocrv6_min.det.
//
// Post-processing follows PaddleOCR ppocr/postprocess/db_postprocess.py
// exactly: threshold -> contours -> minAreaRect -> polygon-mean score ->
// unclip expansion. The pyclipper offset is replaced by an analytic
// rectangle expansion (see DBPostProcess.unclip for why this is exact).

import fs from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-node";
import {
  resizeBilinear,
  findContours,
  minAreaRect,
  boxPoints,
  fillPolyMask,
  meanMasked,
} from "./cv.js";

// ImageNet stats (PaddleX inference.yml: NormalizeImage)
export const MEAN = [0.485, 0.456, 0.406];
export const STD = [0.229, 0.224, 0.225];

export class DBPostProcess {
  constructor({ thresh = 0.2, boxThresh = 0.4, unclipRatio = 1.4, maxCandidates = 3000, minSize = 3 } = {}) {
    this.thresh = thresh;
    this.boxThresh = boxThresh;
    this.unclipRatio = unclipRatio;
    this.maxCandidates = maxCandidates;
    this.minSize = minSize;
  }

  /**
   * Expand a closed polygon outward by `distance` (pyclipper
   * JT_ROUND + ET_CLOSEDPOLYGON equivalent).
   *
   * In the DB pipeline the input is always the output of minAreaRect, i.e. a
   * rectangle (possibly rotated). Offsetting a rectangle with round joins
   * yields a rounded rectangle whose minimum-area rectangle is exactly the
   * miter join of the edges moved out by `distance` — the rounded corners lie
   * strictly inside. pyclipper additionally rounds the input coordinates to
   * integers, which is reproduced here. The result is a single path or an
   * empty list, matching pyclipper.Execute.
   */
  unclip(box, ratio) {
    const pts = box.map((p) => [Math.round(p[0]), Math.round(p[1])]);
    // Shoelace area + perimeter on the float64 input (same as the Python code).
    let area = 0;
    let length = 0;
    for (let i = 0; i < box.length; i++) {
      const [x1, y1] = box[i];
      const [x2, y2] = box[(i + 1) % box.length];
      area += x1 * y2 - x2 * y1;
      length += Math.hypot(x2 - x1, y2 - y1);
    }
    area = 0.5 * Math.abs(area);
    const distance = (area * ratio) / (length + 1e-9);
    if (pts.length < 3) return [];
    if (distance === 0) return [pts]; // zero offset: pyclipper returns the input
    // Signed area determines orientation; outward normal = rotate(dir, 90°)
    // flipped so that it points away from the polygon centroid.
    const n = pts.length;
    let centroidX = 0;
    let centroidY = 0;
    for (const p of pts) {
      centroidX += p[0];
      centroidY += p[1];
    }
    centroidX /= n;
    centroidY /= n;
    const lines = [];
    for (let i = 0; i < n; i++) {
      const [ax, ay] = pts[i];
      const [bx, by] = pts[(i + 1) % n];
      const ex = bx - ax;
      const ey = by - ay;
      const len = Math.hypot(ex, ey);
      if (len === 0) continue;
      let nx = ey / len;
      let ny = -ex / len;
      const mx = (ax + bx) / 2 - centroidX;
      const my = (ay + by) / 2 - centroidY;
      if (nx * mx + ny * my < 0) {
        nx = -nx;
        ny = -ny;
      }
      lines.push({ px: ax + nx * distance, py: ay + ny * distance, dx: ex, dy: ey });
    }
    if (lines.length < 3) return [pts];
    const out = [];
    for (let i = 0; i < lines.length; i++) {
      const l1 = lines[(i - 1 + lines.length) % lines.length];
      const l2 = lines[i];
      out.push(intersectLines(l1, l2));
    }
    return [out];
  }

  /** cv2.minAreaRect + boxPoints + the PaddleOCR corner reordering. */
  getMiniBoxes(contour) {
    const rect = minAreaRect(contour);
    const points = boxPoints(rect).sort((a, b) => a[0] - b[0]);
    let i1; let i2; let i3; let i4;
    if (points[1][1] > points[0][1]) {
      i1 = 0; i4 = 1;
    } else {
      i1 = 1; i4 = 0;
    }
    if (points[3][1] > points[2][1]) {
      i2 = 2; i3 = 3;
    } else {
      i2 = 3; i3 = 2;
    }
    const box = [points[i1], points[i2], points[i3], points[i4]];
    return { box, sside: Math.min(rect.size[0], rect.size[1]) };
  }

  /** Mean probability inside `box` (float32 polygon) on the (h, w) prob map. */
  boxScoreFast(pred, w, h, box) {
    const pts = box.map((p) => [p[0], p[1]]);
    let minX = Infinity; let maxX = -Infinity; let minY = Infinity; let maxY = -Infinity;
    for (const p of pts) {
      if (p[0] < minX) minX = p[0];
      if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1];
      if (p[1] > maxY) maxY = p[1];
    }
    const xmin = clampInt(Math.floor(minX), 0, w - 1);
    const xmax = clampInt(Math.ceil(maxX), 0, w - 1);
    const ymin = clampInt(Math.floor(minY), 0, h - 1);
    const ymax = clampInt(Math.ceil(maxY), 0, h - 1);
    const winW = xmax - xmin + 1;
    const winH = ymax - ymin + 1;
    const local = pts.map((p) => [Math.trunc(p[0] - xmin), Math.trunc(p[1] - ymin)]);
    const mask = fillPolyMask(winW, winH, local);
    // pred window
    let sum = 0;
    let count = 0;
    for (let yy = 0; yy < winH; yy++) {
      for (let xx = 0; xx < winW; xx++) {
        if (mask[yy * winW + xx]) {
          sum += pred[(ymin + yy) * w + (xmin + xx)];
          count++;
        }
      }
    }
    return count === 0 ? 0 : sum / count;
  }

  /**
   * pred: Float32Array (H*W) probability map at network output resolution.
   * Returns { boxes: number[][][] (int quads), scores: number[] }.
   */
  run(pred, w, h, destW, destH) {
    const mask = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) mask[i] = pred[i] > this.thresh ? 1 : 0;
    const contours = findContours(mask, w, h);
    const boxes = [];
    const scores = [];
    for (const contour of contours.slice(0, this.maxCandidates)) {
      let { box, sside } = this.getMiniBoxes(contour);
      if (sside < this.minSize) continue;
      const score = this.boxScoreFast(pred, w, h, box);
      if (this.boxThresh > score) continue;
      const expanded = this.unclip(box, this.unclipRatio);
      if (expanded.length > 1) continue;
      let box2;
      let sside2;
      if (expanded.length === 0) {
        box2 = [];
        sside2 = 0;
      } else {
        ({ box: box2, sside: sside2 } = this.getMiniBoxes(expanded[0]));
      }
      if (sside2 < this.minSize + 2) continue;
      // int32 truncation + rescale to the original image size, as in Python.
      const quad = box2.map((p) => [
        clampInt(Math.round((Math.trunc(p[0]) / w) * destW), 0, destW),
        clampInt(Math.round((Math.trunc(p[1]) / h) * destH), 0, destH),
      ]);
      boxes.push(quad);
      scores.push(score);
    }
    return { boxes, scores };
  }
}

function clampInt(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

/** Intersection of two (point, direction) lines. */
function intersectLines(l1, l2) {
  const denom = l1.dx * l2.dy - l1.dy * l2.dx; // cross(d1, d2)
  if (denom === 0) {
    // Parallel edges (degenerate): keep the shifted endpoint.
    return [l2.px, l2.py];
  }
  const t =
    ((l2.px - l1.px) * l2.dy - (l2.py - l1.py) * l2.dx) / denom;
  return [l1.px + l1.dx * t, l1.py + l1.dy * t];
}

/** DB text-line detector (PP-OCRv6 det ONNX export). */
export class DetModel {
  /**
   * @param {string} modelDir directory holding inference.onnx
   * @param {object} opts { limitSideLen=960, thresh, boxThresh, unclipRatio, providers }
   */
  constructor(modelDir, { limitSideLen = 960, thresh = 0.2, boxThresh = 0.4, unclipRatio = 1.4, providers } = {}) {
    const onnxPath = path.join(modelDir, "inference.onnx");
    if (!fs.existsSync(onnxPath)) {
      throw new Error(`model not found: ${onnxPath}`);
    }
    this.limit = limitSideLen;
    this.post = new DBPostProcess({ thresh, boxThresh, unclipRatio });
    this.session = null; // created lazily (async)
    this.onnxPath = onnxPath;
    this.providers = providers;
  }

  /** Async initializer — await this before calling detect(). */
  async init() {
    this.session = await ort.InferenceSession.create(this.onnxPath, {
      executionProviders: this.providers || ["cpu"],
    });
    this.inName = this.session.inputNames[0];
    return this;
  }

  /** det resize: long side -> limitSideLen (never upscaled), rounded to %32. */
  targetSize(h, w) {
    const scale = Math.min(this.limit / Math.max(h, w), 1.0);
    const nh = Math.max(32, Math.floor(Math.trunc(h * scale) / 32) * 32);
    const nw = Math.max(32, Math.floor(Math.trunc(w * scale) / 32) * 32);
    return [nh, nw];
  }

  /**
   * Detect text lines in a BGR image object.
   * @returns {Promise<Array<{quad: number[][], score: number}>>} quads are
   *   (4,2) int boxes in original image coordinates (corner order not
   *   guaranteed), score is the mean probability inside the box.
   */
  async detect(imgBgr) {
    const { width: w, height: h, data } = imgBgr;
    const [nh, nw] = this.targetSize(h, w);
    const resized = resizeBilinear({ width: w, height: h, channels: 3, data }, nw, nh);
    // Normalize + NCHW, channel order BGR (as fed by the Python reference).
    const c = 3;
    const x = new Float32Array(c * nh * nw);
    const plane = nh * nw;
    const rd = resized.data;
    for (let i = 0; i < plane; i++) {
      for (let ch = 0; ch < c; ch++) {
        x[ch * plane + i] = (rd[i * c + ch] / 255.0 - MEAN[ch]) / STD[ch];
      }
    }
    const input = new ort.Tensor("float32", x, [1, c, nh, nw]);
    const out = await this.session.run({ [this.inName]: input });
    const t = out[this.session.outputNames[0]];
    // prob = out[0][0, 0] -> (nh, nw)
    const probData = t.data;
    const prob = probData.subarray(0, nh * nw); // [1,1,nh,nw] layout
    const { boxes, scores } = this.post.run(
      prob instanceof Float32Array ? prob : Float32Array.from(prob),
      nw, nh, w, h
    );
    return boxes.map((quad, i) => ({ quad, score: scores[i] }));
  }
}
