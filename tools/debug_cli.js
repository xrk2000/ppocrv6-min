#!/usr/bin/env node
// debug_cli.js — debug version of cli that prints the quads and skips degenerate ones.
import fs from "node:fs";
import { OCR } from "../src/pipeline.js";
import { imread } from "../src/io.js";

async function main() {
  const path = process.argv[2] || "../教师节.png";
  const imgBgr = imread(path);
  const ocr = await new OCR().init();
  const found = await ocr.detect(imgBgr);
  console.log(`found ${found.length} boxes`);

  const { orderQuad, warpQuad } = await import(("../src/pipeline.js"));
  const imgRgb = (await import(("../src/io.js"))).toRgb(imgBgr);

  for (const { quad } of found) {
    console.log("raw quad:", JSON.stringify(quad));
    try {
      const ordered = orderQuad(quad);
      console.log("ordered:", JSON.stringify(ordered));
    } catch (e) {
      console.error("orderQuad error:", e);
    }
  }

  for (const { quad } of found) {
    console.log("warping quad:", JSON.stringify(quad));
    try {
      const strip = warpQuad(imgRgb, quad);
      console.log("strip:", strip ? `${strip.width}x${strip.height}` : "null");
    } catch (e) {
      console.error("warp error:", e);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});