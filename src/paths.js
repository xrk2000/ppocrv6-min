// paths.js — model directory resolution (port of ppocrv6_min.paths).
//
// The two tiny ONNX models (det ~1.7 MB + rec ~4.4 MB) are bundled inside the
// package under models/ so a plain `npm install` works out of the box.
// Larger tiers (small/medium) are not bundled; point the API or CLI at them
// via options or the PPOCR_DET_DIR / PPOCR_REC_DIR environment variables.

import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url)); // <pkg>/src
const bundledRoot = path.resolve(here, "..", "models");

export const BUNDLED_DET_DIR = path.join(bundledRoot, "PP-OCRv6_tiny_det_onnx");
export const BUNDLED_REC_DIR = path.join(bundledRoot, "PP-OCRv6_tiny_rec_onnx");

export const DET_ENV_VAR = "PPOCR_DET_DIR";
export const REC_ENV_VAR = "PPOCR_REC_DIR";

/** Resolve the det model dir: $PPOCR_DET_DIR else the bundled tiny model. */
export function defaultDetDir() {
  return process.env[DET_ENV_VAR] || BUNDLED_DET_DIR;
}

/** Resolve the rec model dir: $PPOCR_REC_DIR else the bundled tiny model. */
export function defaultRecDir() {
  return process.env[REC_ENV_VAR] || BUNDLED_REC_DIR;
}

export function bundledAvailable() {
  return (
    fs.existsSync(path.join(BUNDLED_DET_DIR, "inference.onnx")) &&
    fs.existsSync(path.join(BUNDLED_REC_DIR, "inference.onnx")) &&
    fs.existsSync(path.join(BUNDLED_REC_DIR, "inference.yml"))
  );
}
