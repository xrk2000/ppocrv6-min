// index.js — ppocrv6-min for Node.js: minimal-dependency PP-OCRv6 inference
// (det + rec). Only onnxruntime-node / pngjs / jpeg-js are required — no
// PaddlePaddle, PaddleOCR, OpenCV or Python.
//
//   import { OCR } from "ppocrv6-min";
//   const ocr = await new OCR().init();
//   const items = await ocr.predict("截图.png");
//   for (const it of items) console.log(it.box, it.score.toFixed(3), it.text);

import { DetModel } from "./det.js";
import { RecModel } from "./rec.js";
import { OCR, TextItem, orderQuad, warpQuad } from "./pipeline.js";
import {
  BUNDLED_DET_DIR,
  BUNDLED_REC_DIR,
  bundledAvailable,
  defaultDetDir,
  defaultRecDir,
} from "./paths.js";
import { imread, imwrite, imdecode, toRgb } from "./io.js";

export const __version__ = "0.1.0";

export {
  OCR,
  TextItem,
  DetModel,
  RecModel,
  orderQuad,
  warpQuad,
  BUNDLED_DET_DIR,
  BUNDLED_REC_DIR,
  bundledAvailable,
  defaultDetDir,
  defaultRecDir,
  imread,
  imwrite,
  imdecode,
  toRgb,
};
