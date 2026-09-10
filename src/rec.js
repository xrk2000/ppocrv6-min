// rec.js — CTC text-line recognizer (PP-OCRv6 *_rec ONNX), Node port of ppocrv6_min.rec.
//
// Pre/post processing mirrors the PaddleX inference.yml of the rec model:
//   PreProcess : resize to H=48 (W proportional, capped at 320), right-pad to
//                320, (img/255 - 0.5) / 0.5 -> NCHW float32
//   Net        : x[1,3,48,Wp] -> probs[1,T,C] (softmax already inside the graph)
//   PostProcess: greedy CTC decode, labels = ['blank'] + character_dict + [' ']
//
// The character dict is parsed from inference.yml (character_dict, one char
// per "- " line). No YAML library is needed.

import fs from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-node";
import { resizeBilinear } from "./cv.js";

// params from inference.yml (RecResizeImg image_shape: [3, 48, 320])
export const IMG_H = 48;
export const IMG_W_PAD = 320;
export const REC_MEAN = [0.5, 0.5, 0.5];
export const REC_STD = [0.5, 0.5, 0.5];

/** Unquote one YAML scalar as written in PaddleX inference.yml. */
function unquote(item) {
  const s = item.trim();
  if (s.length >= 2 && s[0] === "'" && s[s.length - 1] === "'") {
    return s.slice(1, -1).replace(/''/g, "'");
  }
  if (s.length >= 2 && s[0] === '"' && s[s.length - 1] === '"') {
    return s.slice(1, -1);
  }
  return s;
}

/**
 * Parse character_dict (one char per "- " line) from a PaddleX yml.
 * The section is a flat list of single-character scalars, so a line-by-line
 * parse plus unquote() is exact.
 */
export function loadDict(ymlPath) {
  const lines = fs.readFileSync(ymlPath, "utf-8").split(/\r?\n/);
  const start = lines.findIndex((l) => l.includes("character_dict:"));
  if (start === -1) {
    throw new Error(`no character_dict found in ${ymlPath}`);
  }
  const items = [];
  let n = start + 1;
  while (n < lines.length) {
    const m = lines[n].match(/^\s*-(.*)/);
    if (!m) break;
    items.push(unquote(m[1]));
    n++;
  }
  return items;
}

/** Text-line recognizer. */
export class RecModel {
  /**
   * @param {string} modelDir directory holding inference.onnx + inference.yml
   * @param {object} opts { charset?: string[], providers?: string[] }
   */
  constructor(modelDir, { charset, providers } = {}) {
    const onnxPath = path.join(modelDir, "inference.onnx");
    if (!fs.existsSync(onnxPath)) {
      throw new Error(`model not found: ${onnxPath}`);
    }
    const dict = charset || loadDict(path.join(modelDir, "inference.yml"));
    this.dict = [...dict];
    // PaddleOCR v3 CTC label layout: blank + dict + space
    this.labels = ["blank", ...this.dict, " "];
    this.session = null; // created lazily (async)
    this.onnxPath = onnxPath;
    this.providers = providers;
  }

  /** Async initializer — await this before calling predictLine(). */
  async init() {
    this.session = await ort.InferenceSession.create(this.onnxPath, {
      executionProviders: this.providers || ["cpu"],
    });
    this.inName = this.session.inputNames[0];
    // Validate the class count against the model output (like the Python ctor).
    const outMeta = (this.session.outputMetadata || []).find(
      (m) => m.name === this.session.outputNames[0]
    );
    const nOut = outMeta && outMeta.shape ? outMeta.shape[outMeta.shape.length - 1] : null;
    if (typeof nOut === "number" && nOut !== this.labels.length) {
      throw new Error(
        `model output has ${nOut} classes but the dict defines ` +
        `${this.labels.length} (dict=${this.dict.length})`
      );
    }
    return this;
  }

  /** (img/255 - 0.5)/0.5, resize to 48 rows, right-pad to 320, NCHW float32. */
  preprocess(imgBgr) {
    const { width: w, height: h, data } = imgBgr;
    let newW = Math.round((w / h) * IMG_H);
    newW = Math.max(1, Math.min(newW, IMG_W_PAD));
    const im = resizeBilinear({ width: w, height: h, channels: 3, data }, newW, IMG_H);
    const arr = new Float32Array(3 * IMG_H * IMG_W_PAD); // zero-initialized = pad
    const plane = IMG_H * IMG_W_PAD;
    const id = im.data;
    for (let i = 0; i < IMG_H * newW; i++) {
      for (let ch = 0; ch < 3; ch++) {
        arr[ch * plane + (Math.floor(i / newW) * IMG_W_PAD + (i % newW))] =
          (id[i * 3 + ch] / 255.0 - REC_MEAN[ch]) / REC_STD[ch];
      }
    }
    return arr;
  }

  /** Greedy CTC: (T, C) Float32Array probs -> { text, score }. */
  ctcDecode(probs, T, C) {
    const chars = [];
    let prev = -1;
    let sumMax = 0;
    for (let t = 0; t < T; t++) {
      const off = t * C;
      let best = 0;
      let bestV = -Infinity;
      for (let ci = 0; ci < C; ci++) {
        const v = probs[off + ci];
        if (v > bestV) {
          bestV = v;
          best = ci;
        }
      }
      if (best !== prev && best !== 0) {
        chars.push(this.labels[best]);
      }
      prev = best;
      sumMax += bestV;
    }
    // Merge consecutive duplicates (kept for parity with the Python code;
    // unreachable in practice because prev skips repeats already).
    const merged = [];
    for (const ch of chars) {
      if (merged.length === 0 || merged[merged.length - 1] !== ch) merged.push(ch);
    }
    return { text: merged.join(""), score: sumMax / T };
  }

  /**
   * Recognize a single horizontal text-line crop (RGB uint8 image object).
   * @returns {Promise<{text: string, score: number, elapsedMs: number, size: [w, h]}>}
   */
  async predictLine(imgRgb) {
    // The model is trained on BGR, the pipeline hands over RGB strips.
    const imgBgr = {
      width: imgRgb.width,
      height: imgRgb.height,
      channels: 3,
      data: swapChannels(imgRgb.data),
    };
    const t0 = performance.now();
    const x = this.preprocess(imgBgr);
    const input = new ort.Tensor("float32", x, [1, 3, IMG_H, IMG_W_PAD]);
    const out = await this.session.run({ [this.inName]: input });
    const t = out[this.session.outputNames[0]];
    const [_, T, C] = t.dims;
    const { text, score } = this.ctcDecode(t.data, T, C);
    const elapsedMs = performance.now() - t0;
    return { text, score, elapsedMs, size: [imgRgb.width, imgRgb.height] };
  }
}

function swapChannels(data) {
  const out = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i += 3) {
    out[i] = data[i + 2];
    out[i + 1] = data[i + 1];
    out[i + 2] = data[i];
  }
  return out;
}
