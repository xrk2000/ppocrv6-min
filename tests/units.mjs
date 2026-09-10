// units.mjs — unit tests that do NOT require model files.
// Exports `tests` = { name: async () => "skip" | void (throw on failure) }.
//
// A tiny inline `expect` keeps the runner dependency-free; assertions throw
// on mismatch, the runner counts that as a failure.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { DBPostProcess } from "../src/det.js";
import { loadDict } from "../src/rec.js";
import { orderQuad, warpQuad } from "../src/pipeline.js";
import { imread, imdecode, imwrite } from "../src/io.js";
import {
  resizeBilinear,
  findContours,
  minAreaRect,
  fillPolyMask,
  convexHull,
} from "../src/cv.js";
import { bundledAvailable, BUNDLED_DET_DIR, BUNDLED_REC_DIR } from "../src/paths.js";

function assert(cond, msg) {
  if (!cond) throw new Error(msg ?? "assertion failed");
}
function assertClose(a, b, eps = 1e-6, msg) {
  assert(Math.abs(a - b) <= eps, msg ?? `expected ${a} ~= ${b}`);
}

function tmpFile(name, content) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ppocrv6-"));
  const p = path.join(dir, name);
  fs.writeFileSync(p, content, "utf-8");
  return p;
}

export const tests = {
  "load_dict: strips YAML quotes": () => {
    const yml = tmpFile(
      "inference.yml",
      "PostProcess:\n" +
        "  name: CTCLabelDecode\n" +
        "  character_dict:\n" +
        "  - '!'\n" +
        "  - '\"'\n" +
        "  - ''''\n" +
        "  - $\n" +
        "  - 中\n" +
        "  - 国\n"
    );
    const chars = loadDict(yml);
    assert(JSON.stringify(chars) === JSON.stringify(["!", '"', "'", "$", "中", "国"]),
      `got ${JSON.stringify(chars)}`);
  },

  "load_dict: missing character_dict throws": () => {
    const yml = tmpFile("inference.yml", "Global:\n  model_name: x\n");
    let threw = false;
    try {
      loadDict(yml);
    } catch {
      threw = true;
    }
    assert(threw, "expected loadDict to throw");
  },

  "unclip: expands a rectangle outward": () => {
    const post = new DBPostProcess({ unclipRatio: 2.0 });
    const box = [[0, 0], [10, 0], [10, 10], [0, 10]];
    const expanded = post.unclip(box, 2.0);
    assert(expanded.length === 1, `expected 1 path, got ${expanded.length}`);
    const pts = expanded[0];
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    assert(Math.max(...xs) > 10 && Math.min(...xs) < 0, "x did not expand outward");
    assert(Math.max(...ys) > 10 && Math.min(...ys) < 0, "y did not expand outward");
  },

  "unclip: zero offset returns input": () => {
    const post = new DBPostProcess({ unclipRatio: 0.0 });
    const box = [[5, 5], [15, 5], [15, 12], [5, 12]];
    const expanded = post.unclip(box, 0.0);
    assert(expanded.length === 1 && expanded[0].length === 4, "zero offset should return the box");
  },

  "order_quad: shuffles to tl, tr, br, bl": () => {
    const shuffled = [[100, 100], [0, 0], [100, 0], [0, 100]];
    const q = orderQuad(shuffled);
    assert(q[0][0] === 0 && q[0][1] === 0, `tl wrong: ${q[0]}`);
    assert(q[1][0] === 100 && q[1][1] === 0, `tr wrong: ${q[1]}`);
    assert(q[2][0] === 100 && q[2][1] === 100, `br wrong: ${q[2]}`);
    assert(q[3][0] === 0 && q[3][1] === 100, `bl wrong: ${q[3]}`);
  },

  "warp_quad: horizontal strip keeps size": () => {
    const img = { width: 200, height: 40, channels: 3, data: new Uint8Array(200 * 40 * 3).fill(255) };
    const quad = [[20, 10], [180, 10], [180, 30], [20, 30]];
    const strip = warpQuad(img, quad);
    assert(strip !== null, "expected a strip");
    assert(strip.width >= 160, `width ${strip.width} < 160`);
    assert(strip.height >= 8, `height ${strip.height} < 8`);
  },

  "warp_quad: degenerate quad returns null": () => {
    const img = { width: 100, height: 100, channels: 3, data: new Uint8Array(100 * 100 * 3) };
    assert(warpQuad(img, [[0, 0], [1, 0], [1, 1], [0, 1]]) === null, "width<2 should be null");
  },

  "resize_bilinear: scales down and clamps": () => {
    // black with a white block on the right half
    const src = { width: 100, height: 50, channels: 3, data: new Uint8Array(100 * 50 * 3) };
    for (let y = 0; y < 50; y++)
      for (let x = 50; x < 100; x++) src.data[(y * 100 + x) * 3] = 255;
    const dst = resizeBilinear(src, 10, 5);
    assert(dst.width === 10 && dst.height === 5, "bad size");
    assert(dst.data[0] === 0, "left pixel should stay black");
    assert(dst.data[(0 * 10 + 9) * 3] === 255, "right pixel should stay white");
  },

  "find_contours: single component": () => {
    const w = 20; const h = 10;
    const mask = new Uint8Array(w * h);
    for (let y = 2; y < 8; y++)
      for (let x = 3; x < 17; x++) mask[y * w + x] = 1;
    const contours = findContours(mask, w, h);
    assert(contours.length === 1, `expected 1 contour, got ${contours.length}`);
  },

  "min_area_rect: axis-aligned square": () => {
    const rect = minAreaRect([[5, 5], [15, 5], [15, 15], [5, 15]]);
    assertClose(rect.size[0], 10, 1e-6, "width");
    assertClose(rect.size[1], 10, 1e-6, "height");
  },

  "fill_poly_mask: covers interior": () => {
    const mask = fillPolyMask(10, 10, [[2, 2], [7, 2], [7, 7], [2, 7]]);
    assert(mask[4 * 10 + 4] === 1, "center not filled");
    assert(mask[0] === 0, "corner outside polygon should be 0");
  },

  "convex_hull: drops interior points": () => {
    const hull = convexHull([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]]);
    assert(hull.length === 4, `expected 4 hull points, got ${hull.length}`);
  },

  "imread: missing file throws ENOENT": () => {
    let threw = false;
    try {
      imread(path.join(os.tmpdir(), "definitely-missing-ppocrv6.png"));
    } catch (e) {
      threw = e.code === "ENOENT";
    }
    assert(threw, "expected ENOENT");
  },

  "imdecode/imwrite: png round-trip preserves pixels": () => {
    const img = { width: 4, height: 2, channels: 3, data: new Uint8Array([
      10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
      130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
    ]) };
    const p = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "ppocrv6-")), "rt.png");
    imwrite(p, img);
    const back = imread(p);
    assert(back.width === 4 && back.height === 2, "bad size");
    for (let i = 0; i < img.data.length; i++) assertClose(back.data[i], img.data[i], 0, `pixel ${i}`);
  },

  "bundled models: onnx + yml present": () => {
    if (!bundledAvailable()) return "skip";
    assert(fs.existsSync(path.join(BUNDLED_DET_DIR, "inference.onnx")), "det onnx missing");
    assert(fs.existsSync(path.join(BUNDLED_REC_DIR, "inference.onnx")), "rec onnx missing");
    assert(fs.existsSync(path.join(BUNDLED_REC_DIR, "inference.yml")), "rec yml missing");
  },
};
