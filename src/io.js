// io.js — unicode-safe image IO (Node port of ppocrv6_min.io).
//
// Node's fs already handles Chinese paths on Windows (the reason the Python
// package needs np.fromfile + cv2.imdecode does not apply here), but the
// decode/encode behaviour mirrors cv2.imdecode(..., IMREAD_COLOR):
//   * decode to 3-channel BGR, dropping alpha / converting grayscale+palette
//   * raise on undecodable input

import fs from "node:fs";
import { PNG } from "pngjs";
import jpeg from "jpeg-js";
import { rgbaToBgr, bgrToRgb } from "./cv.js";

/**
 * Read an image file (PNG / JPEG) as { width, height, channels: 3, data } with
 * BGR channel order — the layout every cv2-based Python function sees.
 */
export function imread(path) {
  if (!fs.existsSync(path) || !fs.statSync(path).isFile()) {
    const e = new Error(`ENOENT: no such file: ${path}`);
    e.code = "ENOENT";
    throw e;
  }
  const buf = fs.readFileSync(path);
  return imdecode(buf, path);
}

/** Decode a PNG/JPEG buffer to a BGR image object. */
export function imdecode(buf, label = "<buffer>") {
  let rgba = null;
  let width = 0;
  let height = 0;
  let srcChannels = 4;
  if (isPng(buf)) {
    const png = PNG.sync.read(buf); // -> RGBA (or grey/gray+alpha expanded)
    width = png.width;
    height = png.height;
    srcChannels = png.data.length / (width * height);
    rgba = png.data;
  } else {
    try {
      const raw = jpeg.decode(buf, { useTArray: true, formatAsRGBA: true });
      width = raw.width;
      height = raw.height;
      rgba = raw.data;
    } catch (err) {
      throw new Error(`cannot decode image: ${label} (${err.message})`);
    }
  }
  if (!rgba) {
    throw new Error(`cannot decode image: ${label}`);
  }
  const data = srcChannels === 3 ? rgbToBgr(rgba, width, height) : rgbaToBgr(rgba, width, height, srcChannels);
  return { width, height, channels: 3, data };
}

function isPng(buf) {
  return (
    buf.length > 8 &&
    buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47 &&
    buf[4] === 0x0d && buf[5] === 0x0a && buf[6] === 0x1a && buf[7] === 0x0a
  );
}

function rgbToBgr(rgb, width, height) {
  const out = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    out[i * 3] = rgb[i * 3 + 2];
    out[i * 3 + 1] = rgb[i * 3 + 1];
    out[i * 3 + 2] = rgb[i * 3];
  }
  return out;
}

/** BGR image object -> RGB image object (recognizer-facing channel order). */
export function toRgb(img) {
  return { width: img.width, height: img.height, channels: 3, data: bgrToRgb(img.data, img.width, img.height) };
}

/** Write an image object (BGR) to path as PNG (cv2.imwrite equivalent). */
export function imwrite(path, img) {
  const png = new PNG({ width: img.width, height: img.height });
  for (let i = 0; i < img.width * img.height; i++) {
    png.data[i * 4] = img.data[i * 3 + 2];     // R <- BGR.B
    png.data[i * 4 + 1] = img.data[i * 3 + 1]; // G
    png.data[i * 4 + 2] = img.data[i * 3];     // B <- BGR.R
    png.data[i * 4 + 3] = 255;
  }
  fs.writeFileSync(path, PNG.sync.write(png));
}
