#!/usr/bin/env node
// det_debug.js - compare Node.js detection with Python reference
import fs from "node:fs";
import * as ort from "onnxruntime-node";
import { imread } from "../src/io.js";
import { resizeBilinear, findContours, minAreaRect, boxPoints, fillPolyMask } from "../src/cv.js";

async function main() {
  const path = process.argv[2] || "../教师节.png";
  const imgBgr = imread(path);
  const { width: w, height: h } = imgBgr;

  // Load det model
  const onnxPath = "D:/code/zmxy/ppocrv6-min-node/models/PP-OCRv6_tiny_det_onnx/inference.onnx";
  const sess = await ort.InferenceSession.create(onnxPath, { providers: ["cpu"] });
  const inName = sess.inputNames[0];

  // Det resize
  const limitSideLen = 960;
  const scale = Math.min(limitSideLen / Math.max(h, w), 1.0);
  const nh = Math.max(32, Math.floor(Math.floor(h * scale) / 32) * 32);
  const nw = Math.max(32, Math.floor(Math.floor(w * scale) / 32) * 32);

  const resized = resizeBilinear(imgBgr, nw, nh);

  // Normalize + NCHW, BGR
  const MEAN = [0.485, 0.456, 0.406];
  const STD = [0.229, 0.224, 0.225];
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
  const out = await sess.run({ [inName]: input });
  const t = out[sess.outputNames[0]];
  const prob = Float32Array.from(t.data.subarray(0, nh * nw));

  console.log(`prob shape: [${nh}, ${nw}]`);
  console.log(`prob range: [${Math.min(...prob)}, ${Math.max(...prob)}]`);

  // Threshold
  const thresh = 0.2;
  const mask = new Uint8Array(nw * nh);
  for (let i = 0; i < nw * nh; i++) mask[i] = prob[i] > thresh ? 1 : 0;

  // Find contours
  const contours = findContours(mask, nw, nh);
  console.log(`contours: ${contours.length}`);

  // Process contours
  const boxThresh = 0.4;
  const unclipRatio = 1.4;
  const minSize = 3;

  let finalBoxes = 0;
  const results = [];

  for (const contour of contours.slice(0, 3000)) {
    const rect = minAreaRect(contour);
    const points = boxPoints(rect).sort((a, b) => a[0] - b[0]);

    // Reorder
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
    const sside = Math.min(rect.size[0], rect.size[1]);

    if (sside < minSize) continue;

    // Score
    const pts = box.map(p => [p[0], p[1]]);
    let minX = Infinity; let maxX = -Infinity; let minY = Infinity; let maxY = -Infinity;
    for (const p of pts) {
      if (p[0] < minX) minX = p[0];
      if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1];
      if (p[1] > maxY) maxY = p[1];
    }
    const xmin = clamp(Math.floor(minX), 0, nw - 1);
    const xmax = clamp(Math.ceil(maxX), 0, nw - 1);
    const ymin = clamp(Math.floor(minY), 0, nh - 1);
    const ymax = clamp(Math.ceil(maxY), 0, nh - 1);
    const winW = xmax - xmin + 1;
    const winH = ymax - ymin + 1;
    const local = pts.map(p => [Math.trunc(p[0] - xmin), Math.trunc(p[1] - ymin)]);
    const m = fillPolyMask(winW, winH, local);

    let sum = 0;
    let count = 0;
    for (let yy = 0; yy < winH; yy++) {
      for (let xx = 0; xx < winW; xx++) {
        if (m[yy * winW + xx]) {
          sum += prob[(ymin + yy) * nw + (xmin + xx)];
          count++;
        }
      }
    }
    const score = count === 0 ? 0 : sum / count;

    results.push({ points: pts, sside, score });

    if (boxThresh > score) continue;

    // Unclip
    const ptsRounded = pts.map(p => [Math.round(p[0]), Math.round(p[1])]);

    let area = 0;
    let length = 0;
    for (let i = 0; i < box.length; i++) {
      const [x1, y1] = box[i];
      const [x2, y2] = box[(i + 1) % box.length];
      area += x1 * y2 - x2 * y1;
      length += Math.hypot(x2 - x1, y2 - y1);
    }
    area = 0.5 * Math.abs(area);
    const distance = (area * unclipRatio) / (length + 1e-9);

    if (ptsRounded.length < 3) continue;
    if (distance === 0) {
      results[results.length - 1].unclipResult = "zero_distance";
      continue;
    }

    // Miter offset
    const n = ptsRounded.length;
    let centroidX = 0; let centroidY = 0;
    for (const p of ptsRounded) {
      centroidX += p[0];
      centroidY += p[1];
    }
    centroidX /= n;
    centroidY /= n;
    const lines = [];
    for (let i = 0; i < n; i++) {
      const [ax, ay] = ptsRounded[i];
      const [bx, by] = ptsRounded[(i + 1) % n];
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
    if (lines.length < 3) {
      results[results.length - 1].unclipResult = "insufficient_lines";
      continue;
    }
    const out = [];
    for (let i = 0; i < lines.length; i++) {
      const l1 = lines[(i - 1 + lines.length) % lines.length];
      const l2 = lines[i];
      out.push(intersectLines(l1, l2));
    }
    results[results.length - 1].unclipResult = "success";

    // Get min area rect again
    const rect2 = minAreaRect(out);
    const points2 = boxPoints(rect2).sort((a, b) => a[0] - b[0]);

    // Reorder again
    if (points2[1][1] > points2[0][1]) {
      i1 = 0; i4 = 1;
    } else {
      i1 = 1; i4 = 0;
    }
    if (points2[3][1] > points2[2][1]) {
      i2 = 2; i3 = 3;
    } else {
      i2 = 3; i3 = 2;
    }
    const box2 = [points2[i1], points2[i2], points2[i3], points2[i4]];
    const sside2 = Math.min(rect2.size[0], rect2.size[1]);

    if (sside2 < minSize + 2) continue;

    // Scale to original image
    const quad = box2.map(p => [
      clamp(Math.round((Math.trunc(p[0]) / nw) * w), 0, w),
      clamp(Math.round((Math.trunc(p[1]) / nh) * h), 0, h)
    ]);

    finalBoxes++;
  }

  console.log(`final boxes: ${finalBoxes}`);
  console.log(JSON.stringify(results.slice(0, 10), null, 2));

  // Compare with Python reference
  const pythonRef = JSON.parse(fs.readFileSync("D:/code/zmxy/ppocrv6-min-node/python_det_ref.json", "utf-8"));
  console.log(`\nPython contours: ${pythonRef.contours.length}, final boxes: ${pythonRef.final_boxes.length}`);

  fs.writeFileSync("D:/code/zmxy/ppocrv6-min-node/node_det_ref.json", JSON.stringify(results, null, 2));
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function intersectLines(l1, l2) {
  const denom = l1.dx * l2.dy - l1.dy * l2.dx;
  if (denom === 0) {
    return [l2.px, l2.py];
  }
  const t = ((l2.px - l1.px) * l2.dy - (l2.py - l1.py) * l2.dx) / denom;
  return [l1.px + l1.dx * t, l1.py + l1.dy * t];
}

main().catch(console.error);