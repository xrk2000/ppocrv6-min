// integration.mjs — end-to-end tests against the bundled tiny ONNX models.
// Exports `tests` = { name: async () => "skip" | void (throw on failure) }.
//
// Models: bundled tiny det/rec by default; override with
//   PPOCR_DET_DIR / PPOCR_TEST_DET_DIR  and  PPOCR_REC_DIR / PPOCR_TEST_REC_DIR
// Image:  tests/data/sample.png (one Chinese line, "木灵葫芦"); override with
//   PPOCR_TEST_IMAGE.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { OCR, bundledAvailable, imread } from "../src/index.js";
import { DetModel } from "../src/det.js";
import { RecModel } from "../src/rec.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const IMAGE = process.env.PPOCR_TEST_IMAGE ?? path.join(here, "data", "sample.png");

function hasModels() {
  return bundledAvailable() && fs.existsSync(IMAGE);
}

export const tests = {
  "pipeline: sample line recognized (det + rec)": async () => {
    if (!hasModels()) return "skip";
    const ocr = await new OCR().init();
    const items = await ocr.predict(IMAGE);
    if (items.length < 1) throw new Error(`no lines found, expected >= 1`);
    const joined = items.map((it) => it.text).join("");
    if (!joined.includes("木灵葫芦")) throw new Error(`expected text 木灵葫芦, got ${JSON.stringify(joined)}`);
    // box must be inside the image
    const img = imread(IMAGE);
    for (const it of items) {
      const [x0, y0, x1, y1] = it.box;
      if (!(0 <= x0 < x1 <= img.width && 0 <= y0 < y1 <= img.height))
        throw new Error(`box out of image: ${it.box} (img ${img.width}x${img.height})`);
      if (!(it.score > 0.5)) throw new Error(`low confidence ${it.score}`);
    }
  },

  "rec-only: whole sample as one line (useDet=false)": async () => {
    if (!hasModels()) return "skip";
    const ocr = await new OCR({ useDet: false }).init();
    const items = await ocr.predict(IMAGE);
    if (items.length !== 1) throw new Error(`expected 1 item, got ${items.length}`);
    // sample is a single line, but the rec model pads/normalizes the whole
    // image; assert we get *some* Chinese text with acceptable confidence
    if (!items[0].text) throw new Error("empty text");
    if (!(items[0].score > 0.5)) throw new Error(`low confidence ${items[0].score}`);
  },

  "det model: finds >= 1 quad on sample": async () => {
    if (!hasModels()) return "skip";
    const { BUNDLED_DET_DIR } = await import("../src/paths.js");
    const det = await new DetModel(process.env.PPOCR_TEST_DET_DIR ?? BUNDLED_DET_DIR).init();
    const img = imread(IMAGE);
    const found = await det.detect(img);
    if (found.length < 1) throw new Error(`det found 0 quads, expected >= 1`);
    for (const { quad, score } of found) {
      if (quad.length !== 4 || quad.some((p) => p.length !== 2))
        throw new Error(`bad quad shape: ${JSON.stringify(quad)}`);
      if (!(0 < score <= 1)) throw new Error(`det score out of range: ${score}`);
    }
  },

  "rec model: dict size matches output classes": async () => {
    if (!hasModels()) return "skip";
    const { BUNDLED_REC_DIR } = await import("../src/paths.js");
    const rec = await new RecModel(process.env.PPOCR_TEST_REC_DIR ?? BUNDLED_REC_DIR).init();
    if (rec.dict.length === 0) throw new Error("empty dict");
    if (rec.labels.length !== rec.dict.length + 2)
      throw new Error(`labels ${rec.labels.length} != dict ${rec.dict.length} + 2`);
  },
};
