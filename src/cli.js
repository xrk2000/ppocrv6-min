#!/usr/bin/env node
// cli.js — command line entry: `ppocrv6-min` (port of ppocrv6_min.cli).
//
//   ppocrv6-min screenshot.png
//   ppocrv6-min --no-det single_line.png
//   ppocrv6-min --box-thresh 0.3 --json out.json a.png b.png

import fs from "node:fs";
import { pathToFileURL } from "node:url";
import { __version__ } from "./index.js";
import { OCR } from "./pipeline.js";
import { defaultDetDir, defaultRecDir } from "./paths.js";

const HELP = `usage: ppocrv6-min [-h] [--det-dir DET_DIR] [--rec-dir REC_DIR] [--no-det]
                   [--box-thresh BOX_THRESH] [--limit-side LIMIT_SIDE]
                   [--json PATH] [--version]
                   images [images ...]

Minimal PP-OCRv6 OCR (det + rec) on ONNX Runtime. Bundled tiny models are used by default.

positional arguments:
  images                 image file(s) to OCR

options:
  -h, --help             show this help message and exit
  --det-dir DET_DIR      directory of the *_det_onnx model (default: bundled tiny det, env PPOCR_DET_DIR)
  --rec-dir REC_DIR      directory of the *_rec_onnx model (default: bundled tiny rec, env PPOCR_REC_DIR)
  --no-det               skip detection; feed the whole image to the recognizer
  --box-thresh BOX_THRESH
                         detection box threshold (default 0.4)
  --limit-side LIMIT_SIDE
                         detection long-side target (default 960)
  --json PATH            also write results to a JSON file
  --version              show program's version number and exit`;

function fail(msg) {
  console.error(HELP.split("\n").slice(0, 4).join("\n"));
  console.error(`ppocrv6-min: error: ${msg}`);
  process.exit(2);
}

function parseArgs(argv) {
  const opts = {
    images: [],
    detDir: defaultDetDir(),
    recDir: defaultRecDir(),
    noDet: false,
    boxThresh: 0.4,
    limitSide: 960,
    json: null,
    version: false,
    help: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    let name = a;
    let inlineValue = null;
    const eq = a.indexOf("=");
    if (a.startsWith("--") && eq !== -1) {
      name = a.slice(0, eq);
      inlineValue = a.slice(eq + 1);
    }
    switch (name) {
      case "--no-det":
        opts.noDet = true;
        break;
      case "--version":
        opts.version = true;
        break;
      case "-h":
      case "--help":
        opts.help = true;
        break;
      case "--det-dir":
      case "--rec-dir":
      case "--box-thresh":
      case "--limit-side":
      case "--json": {
        const v = inlineValue !== null ? inlineValue : argv[++i];
        if (v === undefined) fail(`argument ${name}: expected one argument`);
        if (name === "--det-dir") opts.detDir = v;
        else if (name === "--rec-dir") opts.recDir = v;
        else if (name === "--json") opts.json = v;
        else if (name === "--box-thresh") {
          const f = Number(v);
          if (Number.isNaN(f)) fail(`argument --box-thresh: invalid float value: '${v}'`);
          opts.boxThresh = f;
        } else {
          const n = Number(v);
          if (!Number.isInteger(n)) fail(`argument --limit-side: invalid int value: '${v}'`);
          opts.limitSide = n;
        }
        break;
      }
      default:
        if (name.startsWith("-") && name !== "-") {
          fail(`unrecognized arguments: ${name}`);
        }
        opts.images.push(a);
        break;
    }
  }
  return opts;
}

/** CLI main. Returns process exit code. */
export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.help) {
    console.log(HELP);
    return 0;
  }
  if (args.version) {
    console.log(`ppocrv6-min ${__version__}`);
    return 0;
  }
  if (args.images.length === 0) {
    fail("the following arguments are required: images");
  }

  const t0 = performance.now();
  const ocr = await new OCR({
    detDir: args.detDir,
    recDir: args.recDir,
    boxThresh: args.boxThresh,
    limitSideLen: args.limitSide,
    useDet: !args.noDet,
  }).init();
  const allResults = {};
  for (const p of args.images) {
    const tStart = performance.now();
    const items = await ocr.predict(p);
    const dt = performance.now() - tStart;
    console.log(`=== ${p}  (${dt.toFixed(0)} ms, ${items.length} lines) ===`);
    allResults[p] = items.map((it) => ({
      text: it.text,
      score: Math.round(it.score * 10000) / 10000,
      box: it.box,
    }));
    items.forEach((it, idx) => {
      const [x0, y0, x1, y1] = it.box;
      console.log(
        `  L${String(idx + 1).padStart(2)} (${String(x0).padStart(4)},${String(y0).padStart(3)},${String(x1).padStart(4)},${String(y1).padStart(3)})  ` +
        `${it.score.toFixed(3)}  ${it.text}`
      );
    });
  }
  if (args.json) {
    fs.writeFileSync(args.json, JSON.stringify(allResults, null, 2) + "\n", "utf-8");
    console.log(`wrote ${args.json}`);
  }
  console.log(`done in ${(performance.now() - t0).toFixed(0)} ms`);
  return 0;
}

const isMain =
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMain) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
