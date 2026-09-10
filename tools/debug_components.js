#!/usr/bin/env node
// debug_components.js - analyze what components are found
import fs from "node:fs";
import * as ort from "onnxruntime-node";
import { imread } from "../src/io.js";
import { resizeBilinear } from "../src/cv.js";

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

  // Count non-zero pixels
  let nonzero = 0;
  for (let i = 0; i < prob.length; i++) {
    if (prob[i] > 0.2) nonzero++;
  }

  console.log(`prob shape: [${nh}, ${nw}]`);
  console.log(`nonzero (>0.2): ${nonzero} out of ${prob.length} (${(nonzero/prob.length*100).toFixed(2)}%)`);
  console.log(`prob range: [${Math.min(...prob)}, ${Math.max(...prob)}]`);
  console.log(`prob mean: ${(prob.reduce((a,b)=>a+b,0)/prob.length).toFixed(3)}`);

  // Visualize: find top 10 probability peaks and their locations
  const peaks = [];
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      peaks.push({x, y, v: prob[y*nw+x]});
    }
  }
  peaks.sort((a,b) => b.v - a.v);
  console.log("\nTop 10 probability peaks:");
  for (let i = 0; i < 10; i++) {
    console.log(`  ${peaks[i].v.toFixed(3)} at (${peaks[i].x}, ${peaks[i].y})`);
  }
}

main().catch(console.error);