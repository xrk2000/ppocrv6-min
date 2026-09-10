// pipeline.js — end-to-end OCR: DB detection -> perspective warp -> CTC
// recognition (Node port of ppocrv6_min.pipeline).

import { DetModel } from "./det.js";
import { RecModel } from "./rec.js";
import { imread, toRgb } from "./io.js";
import { defaultDetDir, defaultRecDir } from "./paths.js";
import { warpPerspective, getPerspectiveTransform } from "./cv.js";

/** A recognized text line. */
export class TextItem {
  constructor(text, score, box, quad) {
    this.text = text;   // recognized string
    this.score = score; // recognition confidence
    this.box = box;     // [x0, y0, x1, y1] axis-aligned bbox of the quad, int
    this.quad = quad;   // (4,2) float quad in original image coordinates
  }
}

/** Order four points to tl, tr, br, bl (sum/diff corner method). */
export function orderQuad(pts) {
  const s = pts.map((p) => p[0] + p[1]);
  // np.diff(pts, axis=1)[:, 0] == y - x (keep the same convention as Python).
  const d = pts.map((p) => p[1] - p[0]);
  const argmin = (a) => a.reduce((bi, v, i, arr) => (v < arr[bi] ? i : bi), 0);
  const argmax = (a) => a.reduce((bi, v, i, arr) => (v > arr[bi] ? i : bi), 0);
  return [pts[argmin(s)], pts[argmin(d)], pts[argmax(s)], pts[argmax(d)]];
}

/**
 * Perspective-warp a 4-point quad into a horizontal strip (same channel
 * layout). The strip width comes from the longer of the two top/bottom
 * edges; a minimum height of 8 px keeps the recognizer stable on very thin
 * lines. Returns null when the quad is degenerate (width < 2).
 */
export function warpQuad(img, quad) {
  const q = orderQuad(quad);
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const w1 = dist(q[0], q[1]);
  const w2 = dist(q[3], q[2]);
  const h1 = dist(q[0], q[3]);
  const h2 = dist(q[1], q[2]);
  const W = Math.trunc(Math.max(w1, w2));
  const H = Math.max(Math.trunc(Math.max(h1, h2)), 8);
  if (W < 2) return null;
  const dst = [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]];
  const M = getPerspectiveTransform(q, dst);
  return warpPerspective(img, M, W, H);
}

/** PP-OCRv6 detect + recognize pipeline. */
export class OCR {
  /**
   * @param {object} opts
   *   detDir / recDir   model directories (default: bundled tiny models)
   *   boxThresh         detection box threshold (higher = fewer, stricter)
   *   limitSideLen      detection long-side target
   *   useDet            false: feed the whole image to the recognizer
   *   providers         onnxruntime execution providers
   */
  constructor({ detDir, recDir, boxThresh = 0.4, limitSideLen = 960, useDet = true, providers } = {}) {
    this.det = useDet
      ? new DetModel(detDir || defaultDetDir(), { limitSideLen, boxThresh, providers })
      : null;
    this.rec = new RecModel(recDir || defaultRecDir(), { providers });
    this.useDet = useDet;
  }

  /** Async initializer — await once before predict(). */
  async init() {
    if (this.det) await this.det.init();
    await this.rec.init();
    return this;
  }

  /** Detect text lines (or the full image as one line when useDet=false). */
  detect(imgBgr) {
    if (this.det === null) {
      const quad = [[0, 0], [imgBgr.width, 0], [imgBgr.width, imgBgr.height], [0, imgBgr.height]];
      return Promise.resolve([{ quad, score: 1.0 }]);
    }
    return this.det.detect(imgBgr);
  }

  /**
   * Run the full pipeline on a path or a BGR image object.
   * @returns {Promise<TextItem[]>} items sorted top-to-bottom, left-to-right
   */
  async predict(image) {
    let imgBgr;
    if (typeof image === "string") {
      imgBgr = imread(image);
    } else {
      imgBgr = image;
    }
    const imgRgb = toRgb(imgBgr);
    const found = await this.detect(imgBgr);
    const items = [];
    for (const { quad } of found) {
      const strip = warpQuad(imgRgb, quad);
      if (strip === null) continue;
      const { text, score } = await this.rec.predictLine(strip);
      const xs = quad.map((p) => p[0]);
      const ys = quad.map((p) => p[1]);
      const box = [
        Math.trunc(Math.min(...xs)),
        Math.trunc(Math.min(...ys)),
        Math.trunc(Math.max(...xs)),
        Math.trunc(Math.max(...ys)),
      ];
      items.push(new TextItem(text, score, box, quad));
    }
    items.sort((a, b) => a.box[1] - b.box[1] || a.box[0] - b.box[0]);
    return items;
  }

  predictFile(p) {
    return this.predict(p);
  }
}
