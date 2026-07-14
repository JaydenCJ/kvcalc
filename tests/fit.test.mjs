/**
 * fit.test.mjs — solving for the largest context that fits a budget.
 *
 * The tiny config makes the boundary exactly computable: weights at fp16 are
 * 173696 bytes and each token of KV at fp16 costs 256 bytes, so a budget of
 * weights + N·256 must land on maxCtx == N, and one byte less on N-1.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { normalizeConfig, fitContext, estimateKv, CTX_SEARCH_MAX } from "../dist/index.js";
import { tinyConfig } from "./helpers.mjs";

const OPTS = { weightsDtype: "fp16", kvDtype: "fp16", batch: 1, overheadBytes: 0 };
const WEIGHTS_FP16 = 86848 * 2; // params (see weights.test.mjs) × 2 bytes
const KV_PER_TOKEN = 256; // 2 layers · 64 elements · 2 bytes

test("the boundary is exact: budget = weights + N tokens → maxCtx == N, one byte less → N−1", () => {
  const cfg = normalizeConfig(tinyConfig());
  const exact = fitContext(cfg, { ...OPTS, budgetBytes: WEIGHTS_FP16 + 1000 * KV_PER_TOKEN });
  assert.equal(exact.maxCtx, 1000);
  assert.equal(exact.kvBytesAtMax, 1000 * KV_PER_TOKEN);
  const under = fitContext(cfg, { ...OPTS, budgetBytes: WEIGHTS_FP16 + 1000 * KV_PER_TOKEN - 1 });
  assert.equal(under.maxCtx, 999);
});

test("weights overflowing the budget → maxCtx 0; fitting exactly → maxCtx 0 but kv budget 0", () => {
  const cfg = normalizeConfig(tinyConfig());
  const over = fitContext(cfg, { ...OPTS, budgetBytes: WEIGHTS_FP16 - 1 });
  assert.equal(over.maxCtx, 0);
  assert.ok(over.kvBudgetBytes < 0);
  const exact = fitContext(cfg, { ...OPTS, budgetBytes: WEIGHTS_FP16 });
  assert.equal(exact.maxCtx, 0);
  assert.equal(exact.kvBudgetBytes, 0);
});

test("overhead is subtracted before solving", () => {
  const cfg = normalizeConfig(tinyConfig());
  const base = fitContext(cfg, { ...OPTS, budgetBytes: WEIGHTS_FP16 + 1000 * KV_PER_TOKEN });
  const withOverhead = fitContext(cfg, {
    ...OPTS,
    budgetBytes: WEIGHTS_FP16 + 1000 * KV_PER_TOKEN,
    overheadBytes: 100 * KV_PER_TOKEN,
  });
  assert.equal(base.maxCtx - withOverhead.maxCtx, 100);
});

test("batch divides the achievable context", () => {
  const cfg = normalizeConfig(tinyConfig());
  const b1 = fitContext(cfg, { ...OPTS, budgetBytes: WEIGHTS_FP16 + 8000 * KV_PER_TOKEN });
  const b4 = fitContext(cfg, { ...OPTS, batch: 4, budgetBytes: WEIGHTS_FP16 + 8000 * KV_PER_TOKEN });
  assert.equal(b1.maxCtx, 8000);
  assert.equal(b4.maxCtx, 2000);
});

test("fullContextFits compares maxCtx against max_position_embeddings", () => {
  const cfg = normalizeConfig(tinyConfig()); // model max 4096
  const yes = fitContext(cfg, { ...OPTS, budgetBytes: WEIGHTS_FP16 + 4096 * KV_PER_TOKEN });
  const no = fitContext(cfg, { ...OPTS, budgetBytes: WEIGHTS_FP16 + 4095 * KV_PER_TOKEN });
  assert.equal(yes.fullContextFits, true);
  assert.equal(no.fullContextFits, false);
});

test("fullContextFits is null when the config has no max_position_embeddings", () => {
  const cfg = normalizeConfig(tinyConfig({ max_position_embeddings: undefined }));
  const fit = fitContext(cfg, { ...OPTS, budgetBytes: WEIGHTS_FP16 + KV_PER_TOKEN });
  assert.equal(fit.fullContextFits, null);
});

test("an all-sliding model flattens out: any post-window budget hits the search ceiling", () => {
  const cfg = normalizeConfig(tinyConfig({ sliding_window: 128 }));
  const kvAtWindow = estimateKv(cfg, "fp16", 128, 1).bytes;
  const fit = fitContext(cfg, { ...OPTS, budgetBytes: WEIGHTS_FP16 + kvAtWindow });
  // KV never grows past the window, so "infinite" context fits (capped at the ceiling).
  assert.equal(fit.maxCtx, CTX_SEARCH_MAX);
});

test("cheaper kv dtypes buy more context in exact proportion", () => {
  const cfg = normalizeConfig(tinyConfig({ max_position_embeddings: undefined }));
  const budgetBytes = WEIGHTS_FP16 + 9000 * KV_PER_TOKEN;
  const fp16 = fitContext(cfg, { ...OPTS, budgetBytes });
  const q4 = fitContext(cfg, { ...OPTS, kvDtype: "q4_0", budgetBytes });
  assert.equal(fp16.maxCtx, 9000);
  assert.equal(q4.maxCtx, 32000); // 16 / 4.5 × 9000
});
