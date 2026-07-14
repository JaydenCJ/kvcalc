/**
 * units.test.mjs — size/context parsing and formatting.
 *
 * The GB-means-GiB convention is load-bearing: a "24GB" GPU has 24 GiB of
 * VRAM, and a calculator that silently used decimal gigabytes would be ~7%
 * optimistic on every verdict.
 */

import test from "node:test";
import assert from "node:assert/strict";
import {
  parseSize,
  parseCtx,
  parseCtxList,
  formatBytes,
  formatCount,
  formatCtx,
  UnitError,
  GIB,
  MIB,
} from "../dist/index.js";

test("parseSize: byte suffixes are binary; GB and bare G read as GiB (GPU spec-sheet convention)", () => {
  assert.equal(parseSize("24GiB"), 24 * GIB);
  assert.equal(parseSize("24GB"), 24 * GIB);
  assert.equal(parseSize("24g"), 24 * GIB);
  assert.equal(parseSize("512MiB"), 512 * MIB);
  assert.equal(parseSize("1.5g"), 1.5 * GIB);
  assert.equal(parseSize("1024"), 1024);
  assert.equal(parseSize(" 8 kib "), 8 * 1024);
});

test("parseSize: rejects garbage, negatives and unknown units", () => {
  assert.throws(() => parseSize("many"), UnitError);
  assert.throws(() => parseSize("24 parsecs"), UnitError);
  assert.throws(() => parseSize(""), UnitError);
  assert.throws(() => parseSize("-4GiB"), UnitError);
});

test("parseCtx: k and m suffixes are binary, matching context-window naming", () => {
  assert.equal(parseCtx("128k"), 131072);
  assert.equal(parseCtx("32K"), 32768);
  assert.equal(parseCtx("1m"), 1048576);
  assert.equal(parseCtx("4096"), 4096);
});

test("parseCtx: rejects zero, negatives and non-integer token counts", () => {
  assert.throws(() => parseCtx("0"), UnitError);
  assert.throws(() => parseCtx("1.5"), UnitError); // 1.5 tokens is not a thing
  assert.equal(parseCtx("0.5k"), 512); // but 0.5k = 512 whole tokens is fine
  assert.throws(() => parseCtx("-8k"), UnitError);
  assert.throws(() => parseCtx("8kb"), UnitError); // bytes units make no sense for tokens
});

test("parseCtxList: splits, trims and rejects empties", () => {
  assert.deepEqual(parseCtxList("4k, 8k,16384"), [4096, 8192, 16384]);
  assert.throws(() => parseCtxList(" , "), UnitError);
});

test("formatBytes: picks the right binary unit with two decimals", () => {
  assert.equal(formatBytes(24 * GIB), "24.00 GiB");
  assert.equal(formatBytes(16 * GIB + GIB / 2), "16.50 GiB");
  assert.equal(formatBytes(128 * 1024), "128.00 KiB");
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(1536 * GIB), "1.50 TiB");
});

test("formatCount and formatCtx: colloquial forms, numeric fallbacks", () => {
  assert.equal(formatCount(8030261248), "8.03 B");
  assert.equal(formatCount(525336576), "525.34 M");
  assert.equal(formatCount(4096), "4.10 K");
  assert.equal(formatCount(320), "320");
  assert.equal(formatCtx(131072), "128k");
  assert.equal(formatCtx(1048576), "1m");
  assert.equal(formatCtx(159465), "159465");
});
