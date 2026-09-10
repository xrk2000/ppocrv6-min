#!/usr/bin/env node
// test_integration.js - integration test comparing Node.js vs Python
import { OCR } from "../src/pipeline.js";
import fs from "node:fs";
import { execSync } from "node:child_process";

async function main() {
  const imagePath = "../教师节.png";

  // Test 1: CLI runs without error
  console.log("Test 1: CLI execution...");
  try {
    const cliOutput = execSync("node src/cli.js " + imagePath, {
      encoding: "utf-8",
      cwd: process.cwd()
    });
    const lines = cliOutput.split("\n").filter(l => l.includes("L") && l.includes(":"));
    console.log(`  CLI found ${lines.length} lines`);
    console.log(`  ✓ CLI execution successful`);
  } catch (e) {
    console.error(`  ✗ CLI execution failed: ${e}`);
    return 1;
  }

  // Test 2: API usage
  console.log("\nTest 2: API usage...");
  try {
    const ocr = await new OCR().init();
    const items = await ocr.predict(imagePath);
    console.log(`  API found ${items.length} items`);

    const expectedTexts = [
      "玄女的考验",
      "读书破万卷，下笔如有神。",
      "玄女想要来考你，分数越高，",
      "可获得的【奖券】数量越多。",
      "【奖券】可在\"活动商店\"兑换稀有道具",
      "活动截止时间：2026年9月27日"
    ];

    const foundTexts = items.map(it => it.text);
    const allFound = expectedTexts.every(expected => {
      return foundTexts.some(found => found === expected || found.includes(expected.slice(0, 4)));
    });

    if (items.length === 6 && allFound) {
      console.log(`  ✓ All 6 lines recognized correctly`);
    } else {
      console.log(`  ✗ Expected 6 lines with specific texts`);
      console.log(`    Found: ${JSON.stringify(foundTexts)}`);
      return 1;
    }
  } catch (e) {
    console.error(`  ✗ API test failed: ${e}`);
    return 1;
  }

  // Test 3: Individual modules
  console.log("\nTest 3: Individual modules...");
  try {
    const { DetModel, RecModel } = await import("../src/index.js");
    const { imread, toRgb } = await import("../src/io.js");

    const det = new DetModel("D:/code/zmxy/ppocrv6-min-node/models/PP-OCRv6_tiny_det_onnx");
    await det.init();

    const rec = new RecModel("D:/code/zmxy/ppocrv6-min-node/models/PP-OCRv6_tiny_rec_onnx");
    await rec.init();

    const imgBgr = imread(imagePath);
    const quadsScores = await det.detect(imgBgr);
    console.log(`  DetModel: found ${quadsScores.length} quads`);

    if (quadsScores.length !== 6) {
      console.log(`  ✗ DetModel found ${quadsScores.length} quads, expected 6`);
      return 1;
    }

    const imgRgb = toRgb(imgBgr);
    const { orderQuad, warpQuad } = await import("../src/pipeline.js");
    for (const { quad } of quadsScores) {
      const strip = warpQuad(imgRgb, quad);
      if (!strip) continue;
      const { text } = await rec.predictLine(strip);
      if (!text) continue;
    }
    console.log(`  RecModel: successful`);
    console.log(`  ✓ Individual modules work`);
  } catch (e) {
    console.error(`  ✗ Module test failed: ${e}`);
    return 1;
  }

  // Test 4: JSON output
  console.log("\nTest 4: JSON output...");
  try {
    const jsonPath = "test_output.json";
    execSync(`node src/cli.js --json ${jsonPath} ${imagePath}`, {
      encoding: "utf-8",
      cwd: process.cwd()
    });
    const json = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
    if (json[imagePath] && Array.isArray(json[imagePath]) && json[imagePath].length === 6) {
      console.log(`  ✓ JSON output valid`);
      fs.unlinkSync(jsonPath);
    } else {
      console.log(`  ✗ JSON output invalid`);
      return 1;
    }
  } catch (e) {
    console.error(`  ✗ JSON test failed: ${e}`);
    return 1;
  }

  console.log("\n✅ All integration tests passed!");
  return 0;
}

main().then(process.exit).catch(err => {
  console.error(err);
  process.exit(1);
});