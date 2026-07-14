/**
 * dtype.test.mjs — the bits-per-weight registry.
 *
 * Block-quant figures must equal bytes-per-block × 8 / weights-per-block from
 * the published block layouts; these tests pin them so a "helpful" rounding
 * never sneaks in. A wrong bpw here corrupts every downstream number.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { resolveDtype, dtypesFor, DTYPES, DtypeError } from "../dist/index.js";

test("floats are exact powers of two bits", () => {
  assert.equal(resolveDtype("fp32", "weights").bitsPerWeight, 32);
  assert.equal(resolveDtype("fp16", "weights").bitsPerWeight, 16);
  assert.equal(resolveDtype("bf16", "weights").bitsPerWeight, 16);
  assert.equal(resolveDtype("fp8", "weights").bitsPerWeight, 8);
});

test("32-wide block quants: q8_0 34B, q5_1 24B, q5_0 22B, q4_1 20B, q4_0 18B per 32", () => {
  assert.equal(resolveDtype("q8_0", "weights").bitsPerWeight, (34 * 8) / 32);
  assert.equal(resolveDtype("q5_1", "weights").bitsPerWeight, (24 * 8) / 32);
  assert.equal(resolveDtype("q5_0", "weights").bitsPerWeight, (22 * 8) / 32);
  assert.equal(resolveDtype("q4_1", "weights").bitsPerWeight, (20 * 8) / 32);
  assert.equal(resolveDtype("q4_0", "weights").bitsPerWeight, (18 * 8) / 32);
});

test("k-quant super-blocks: q6_K 210B, q5_K 176B, q4_K 144B, q3_K 110B, q2_K 82B per 256", () => {
  assert.equal(resolveDtype("q6_K", "weights").bitsPerWeight, (210 * 8) / 256);
  assert.equal(resolveDtype("q5_K", "weights").bitsPerWeight, (176 * 8) / 256);
  assert.equal(resolveDtype("q4_K", "weights").bitsPerWeight, (144 * 8) / 256);
  assert.equal(resolveDtype("q3_K", "weights").bitsPerWeight, (110 * 8) / 256);
  assert.equal(resolveDtype("q2_K", "weights").bitsPerWeight, (82 * 8) / 256);
});

test("mixed _M presets carry the approx flag; pure block types do not", () => {
  assert.equal(resolveDtype("q4_K_M", "weights").approx, true);
  assert.equal(resolveDtype("q5_K_M", "weights").approx, true);
  assert.equal(resolveDtype("q3_K_M", "weights").approx, true);
  assert.equal(resolveDtype("q4_K", "weights").approx, false);
  assert.equal(resolveDtype("q8_0", "weights").approx, false);
  assert.equal(resolveDtype("fp16", "weights").approx, false);
});

test("lookup is case- and separator-insensitive, and aliases resolve to canonical dtypes", () => {
  assert.equal(resolveDtype("Q4_K_M", "weights").name, "q4_K_M");
  assert.equal(resolveDtype("q4-k-m", "weights").name, "q4_K_M");
  assert.equal(resolveDtype("q4km", "weights").name, "q4_K_M");
  assert.equal(resolveDtype("BF16", "weights").name, "bf16");
  assert.equal(resolveDtype("f16", "kv").name, "fp16");
  assert.equal(resolveDtype("half", "weights").name, "fp16");
  assert.equal(resolveDtype("float32", "weights").name, "fp32");
  assert.equal(resolveDtype("bfloat16", "kv").name, "bf16");
  assert.equal(resolveDtype("fp8_e4m3", "kv").name, "fp8");
});

test("unknown dtype names throw a DtypeError that points at `kvcalc dtypes`", () => {
  assert.throws(() => resolveDtype("q4_X_L", "weights"), DtypeError);
  assert.throws(() => resolveDtype("", "weights"), DtypeError);
  try {
    resolveDtype("q9_Z", "weights");
    assert.fail("should have thrown");
  } catch (err) {
    assert.match(err.message, /kvcalc dtypes/);
  }
});

test("256-wide super-block quants are rejected for the KV cache role", () => {
  assert.throws(() => resolveDtype("q4_K_M", "kv"), DtypeError);
  assert.throws(() => resolveDtype("q6_K", "kv"), DtypeError);
  // ...but 32-wide block quants and floats are fine.
  assert.equal(resolveDtype("q8_0", "kv").name, "q8_0");
  assert.equal(resolveDtype("q4_0", "kv").name, "q4_0");
  assert.equal(resolveDtype("q5_1", "kv").name, "q5_1");
});

test("dtypesFor filters by role, preserves order; every entry documents its provenance", () => {
  const kv = dtypesFor("kv");
  assert.ok(kv.every((d) => d.kv));
  assert.ok(kv.length >= 8);
  const weights = dtypesFor("weights");
  assert.ok(weights.every((d) => d.weights));
  const names = DTYPES.map((d) => d.name);
  assert.deepEqual(
    weights.map((d) => d.name),
    names.filter((n) => weights.some((d) => d.name === n)),
  );
  for (const d of DTYPES) {
    assert.ok(d.note.length > 0, `${d.name} has no note`);
    assert.ok(d.bitsPerWeight > 0 && d.bitsPerWeight <= 32);
  }
});
