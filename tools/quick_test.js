#!/usr/bin/env node
// quick_test.js - quick test
import fs from "node:fs";
import * as ort from "onnxruntime-node";
import { imread } from "../src/io.js";
import { resizeBilinear, findContours } from "../src/cv.js";

async function main() {
  const path = "../教师节.png";
  const imgBgr = imread(path);
  const { width: w, height: h } = imgBgr;

  const onnxPath = "D:/code/zmxy/ppocrv6-min-node/models/PP-OCRv6_tiny_det_onnx/inference.onnx";
  const sess = await ort.InferenceSession.create(onnxPath, { providers: ["cpu"] });
  const inName = sess.inputNames[0];

  const limitSideLen = 960;
  const scale = Math.min(limitSideLen / Math.max(h, w), 1.0);
  const nh = Math.max(32, Math.floor(Math.floor(h * scale) / 32) * 32);
  const nw = Math.max(32, Math.floor(Math.floor(w * scale) / 32) * 32);

  const resized = resizeBilinear(imgBgr, nw, nh);

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

  const thresh = 0.2;
  const mask = new Uint8Array(nw * nh);
  for (let i = 0; i < nw * nh; i++) mask[i] = prob[i] > thresh ? 1 : 0;

  const contours = findContours(mask, nw, nh);
  console.log(`Contours: ${contours.length}`);
  contours.forEach((c, i) => console.log(`  ${i}: ${c.length} points`));
}

main().catch(console.error);