#!/usr/bin/env node
// run_tests.mjs — zero-dependency test runner (no test framework needed).
//
//   node tests/run_tests.mjs            # run all
//   node tests/run_tests.mjs unit       # units only (no model files needed)
//   node tests/run_tests.mjs integ      # integration only (bundled models)
//
// Exit code: 0 = all passed (skips don't count as failures).

import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const only = process.argv[2]; // "unit" | "integ" | undefined

const suites = [
  { name: "unit", file: path.join(here, "units.mjs") },
  { name: "integ", file: path.join(here, "integration.mjs") },
].filter((s) => !only || s.name === only);

let totalPass = 0;
let totalFail = 0;
let totalSkip = 0;
const failures = [];

for (const suite of suites) {
  const mod = await import(pathToFileURL(suite.file).href);
  console.log(`\n== ${suite.name} ==`);
  for (const [name, fn] of Object.entries(mod.tests ?? {})) {
    let outcome;
    try {
      const r = await fn();
      outcome = r === "skip" ? "skip" : "pass";
    } catch (err) {
      outcome = "fail";
      failures.push({ suite: suite.name, name, err });
    }
    if (outcome === "pass") totalPass++;
    else if (outcome === "skip") totalSkip++;
    else totalFail++;
    const mark = outcome === "pass" ? "  ok  " : outcome === "skip" ? " skip " : " FAIL ";
    console.log(`${mark} ${name}${outcome === "fail" ? ` — ${failures.at(-1).err.message}` : ""}`);
  }
}

console.log(
  `\n${totalPass} passed, ${totalFail} failed, ${totalSkip} skipped (${suites.length} suite(s))`
);
if (failures.length > 0) {
  for (const f of failures) {
    console.error(`\n--- ${f.suite} :: ${f.name} ---`);
    console.error(f.err.stack ?? f.err.message);
  }
  process.exit(1);
}
