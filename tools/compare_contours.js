#!/usr/bin/env node
// compare_contours.js - detailed contour comparison
import fs from "node:fs";

const pythonRef = JSON.parse(fs.readFileSync("D:/code/zmxy/ppocrv6-min-node/python_det_ref.json", "utf-8"));
const nodeRef = JSON.parse(fs.readFileSync("D:/code/zmxy/ppocrv6-min-node/node_det_ref.json", "utf-8"));

console.log(`Python contours: ${pythonRef.contours.length}`);
console.log(`Node contours: ${nodeRef.length}\n`);

console.log("Python contour details (first 3):");
for (let i = 0; i < Math.min(3, pythonRef.contours.length); i++) {
  const c = pythonRef.contours[i];
  console.log(`  contour ${i}: ${c.points.length} points, sside=${c.sside.toFixed(2)}, score=${c.score?.toFixed(3)}`);
}

console.log("\nNode contour details (first 3):");
for (let i = 0; i < Math.min(3, nodeRef.length); i++) {
  const c = nodeRef[i];
  console.log(`  contour ${i}: ${c.points.length} points, sside=${c.sside.toFixed(2)}, score=${c.score.toFixed(3)}, unclip=${c.unclipResult}`);
}