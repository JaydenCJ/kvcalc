/**
 * cli.test.mjs — integration tests against the real compiled CLI.
 *
 * Each test spawns `node dist/cli.js` exactly as a user would run it, and
 * asserts on stdout/stderr/exit codes. The exit-code contract (0 fits, 1 a
 * --vram check failed, 2 usage error) is what makes kvcalc scriptable, so it
 * gets the densest coverage.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { runCli, tinyConfig, ROOT } from "./helpers.mjs";

const GQA8B = "examples/gqa-8b.json";

test("--version prints the package.json version, nothing else", () => {
  const pkg = JSON.parse(readFileSync(path.join(ROOT, "package.json"), "utf8"));
  const res = runCli(["--version"]);
  assert.equal(res.status, 0);
  assert.equal(res.stdout.trim(), pkg.version);
});

test("--help documents commands, key flags, exit codes; bare invocation exits 2", () => {
  const res = runCli(["--help"]);
  assert.equal(res.status, 0);
  for (const word of ["report", "table", "fit", "dtypes", "--vram", "--kv", "--json", "Exit codes"]) {
    assert.ok(res.stdout.includes(word), `help missing ${word}`);
  }
  assert.equal(runCli([]).status, 2); // help shown, but it was not a successful run
});

test("unknown commands, unknown flags and misplaced flags exit 2 with stderr messages", () => {
  const cmd = runCli(["frobnicate", GQA8B]);
  assert.equal(cmd.status, 2);
  assert.match(cmd.stderr, /unknown command/);
  const flag = runCli(["report", GQA8B, "--kv-dytpe", "fp16"]);
  assert.equal(flag.status, 2);
  assert.match(flag.stderr, /unknown flag --kv-dytpe/);
  const misplaced = runCli(["report", GQA8B, "--ctx-list", "4k,8k"]);
  assert.equal(misplaced.status, 2);
  assert.match(misplaced.stderr, /--ctx-list only applies to table/);
});

test("missing operands and missing files exit 2", () => {
  assert.equal(runCli(["report"]).status, 2);
  const res = runCli(["report", "does-not-exist.json"]);
  assert.equal(res.status, 2);
  assert.match(res.stderr, /cannot read/);
});

test("bad dtype, bad ctx and bad batch values exit 2", () => {
  assert.equal(runCli(["report", GQA8B, "--weights", "q9_Z"]).status, 2);
  assert.equal(runCli(["report", GQA8B, "--ctx", "many"]).status, 2);
  assert.equal(runCli(["report", GQA8B, "--batch", "0"]).status, 2);
  assert.equal(runCli(["report", GQA8B, "--kv", "q4_K_M"]).status, 2); // not a kv dtype
});

test("report: the flagship question — 128k ctx in 24GiB at q4_K_M — fits, exit 0", () => {
  const res = runCli(["report", GQA8B, "--ctx", "128k", "--weights", "q4_K_M", "--vram", "24GiB"]);
  assert.equal(res.status, 0);
  assert.match(res.stdout, /16\.00 GiB/);
  assert.match(res.stdout, /FITS/);
});

test("report: the same question at bf16 weights does not fit, exit 1", () => {
  const res = runCli(["report", GQA8B, "--ctx", "128k", "--vram", "24GiB"]);
  assert.equal(res.status, 1);
  assert.match(res.stdout, /DOES NOT FIT/);
});

test("report --json: machine-readable, correct headline numbers, ctx defaults to model max", () => {
  const res = runCli(["report", GQA8B, "--weights", "q4_K_M", "--json"]);
  assert.equal(res.status, 0);
  const r = JSON.parse(res.stdout);
  assert.equal(r.tool, "kvcalc");
  assert.equal(r.command, "report");
  assert.equal(r.params.total, 8030261248);
  assert.equal(r.kv.ctx, 131072); // --ctx omitted → model max
  assert.equal(r.kv.bytes, 16 * 1024 ** 3);
  assert.equal(r.model.attention, "gqa");
});

test("config warnings go to stderr, never into the report on stdout", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "kvcalc-cli-"));
  try {
    const p = path.join(dir, "config.json");
    writeFileSync(p, JSON.stringify(tinyConfig({ intermediate_size: undefined })));
    const res = runCli(["report", p, "--ctx", "1k"]);
    assert.equal(res.status, 0);
    assert.match(res.stderr, /note: no intermediate_size/);
    assert.ok(!res.stdout.includes("note:"));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("table: default grid ends at the model max; exits 1 when a cell overflows the budget", () => {
  const res = runCli(["table", GQA8B, "--vram", "8GiB"]);
  assert.equal(res.status, 1); // 128k fp16 cell does not fit in 8 GiB
  assert.match(res.stdout, /128k/);
  assert.match(res.stdout, /✗/);
  assert.match(res.stdout, /✓/);
  // Without --vram there are no marks and the exit code is 0.
  const plain = runCli(["table", GQA8B, "--ctx-list", "4k,8k"]);
  assert.equal(plain.status, 0);
  assert.ok(!plain.stdout.includes("✗"));
});

test("table --json: grid dimensions follow --ctx-list and --kv-list", () => {
  const res = runCli(["table", GQA8B, "--ctx-list", "4k,64k", "--kv-list", "fp16,q8_0", "--json"]);
  const t = JSON.parse(res.stdout);
  assert.deepEqual(t.rows.map((r) => r.ctx), [4096, 65536]);
  assert.deepEqual(t.kvDtypes, ["fp16", "q8_0"]);
});

test("fit: reports the max context and that the full model context fits", () => {
  const res = runCli(["fit", GQA8B, "--vram", "24GiB", "--weights", "q4_K_M", "--json"]);
  assert.equal(res.status, 0);
  const f = JSON.parse(res.stdout);
  assert.ok(f.maxCtx > 131072, `maxCtx ${f.maxCtx}`);
  assert.equal(f.fullContextFits, true);
});

test("fit: exits 1 when weights alone exceed the budget; missing --vram exits 2", () => {
  const res = runCli(["fit", GQA8B, "--vram", "4GiB"]);
  assert.equal(res.status, 1);
  assert.match(res.stdout, /weights alone exceed the budget/);
  const usage = runCli(["fit", GQA8B]);
  assert.equal(usage.status, 2);
  assert.match(usage.stderr, /fit requires --vram/);
});

test("fit: --overhead shrinks the achievable context by exactly overhead / bytes-per-token", () => {
  const base = JSON.parse(
    runCli(["fit", GQA8B, "--vram", "24GiB", "--weights", "q4_0", "--json"]).stdout,
  );
  const less = JSON.parse(
    runCli(["fit", GQA8B, "--vram", "24GiB", "--weights", "q4_0", "--overhead", "2GiB", "--json"]).stdout,
  );
  assert.equal(base.maxCtx - less.maxCtx, (2 * 1024 ** 3) / (128 * 1024)); // 2GiB / 128KiB per token
});

test("dtypes: lists the reference table; --json round-trips", () => {
  const res = runCli(["dtypes"]);
  assert.equal(res.status, 0);
  assert.match(res.stdout, /q4_K_M/);
  assert.match(res.stdout, /~4\.85/);
  const j = JSON.parse(runCli(["dtypes", "--json"]).stdout);
  assert.ok(j.dtypes.length >= 15);
});

test("dtypes: model flags are rejected, not silently ignored", () => {
  // dtypes takes no model, so accepting --vram would imply a check that never ran.
  const res = runCli(["dtypes", "--vram", "24GiB"]);
  assert.equal(res.status, 2);
  assert.match(res.stderr, /--vram does not apply to dtypes/);
});

test("identical invocations produce byte-identical output (determinism is API)", () => {
  const args = ["table", GQA8B, "--vram", "24GiB", "--weights", "q4_K_M", "--json"];
  const a = runCli(args);
  const b = runCli(args);
  assert.equal(a.stdout, b.stdout);
  assert.equal(a.status, b.status);
});

test("batch scales the report linearly: kv at batch 4 is 4× batch 1", () => {
  const one = JSON.parse(runCli(["report", GQA8B, "--ctx", "8k", "--json"]).stdout);
  const four = JSON.parse(runCli(["report", GQA8B, "--ctx", "8k", "--batch", "4", "--json"]).stdout);
  assert.equal(four.kv.bytes, 4 * one.kv.bytes);
});
