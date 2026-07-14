/**
 * kv.test.mjs — the KV-cache formulas.
 *
 * The GQA baseline (2 · kv_heads · head_dim per layer per token) and the MLA
 * baseline (kv_lora_rank + qk_rope_head_dim) are pinned with hand-computed
 * bytes; sliding-window behavior is checked at, below and above the window.
 */

import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import {
  normalizeConfig,
  loadConfig,
  estimateKv,
  kvBytesPerToken,
  kvElementsPerLayer,
} from "../dist/index.js";
import { tinyConfig, EXAMPLES } from "./helpers.mjs";

test("GQA elements per layer = 2 · kv_heads · head_dim", () => {
  const cfg = normalizeConfig(tinyConfig());
  assert.equal(kvElementsPerLayer(cfg), 2 * 2 * 16);
});

test("MLA caches the latent + rope key, not per-head K/V", () => {
  const cfg = normalizeConfig(
    tinyConfig({
      kv_lora_rank: 32,
      q_lora_rank: 24,
      qk_rope_head_dim: 8,
      qk_nope_head_dim: 16,
      v_head_dim: 16,
    }),
  );
  // The same geometry as plain GQA would cache 2·2·16=64 elements; MLA caches 40.
  assert.equal(kvElementsPerLayer(cfg), 32 + 8);
});

test("the 8B GQA example costs exactly 128 KiB per token at fp16", () => {
  const cfg = loadConfig(path.join(EXAMPLES, "gqa-8b.json"));
  // 32 layers · 2 · 8 kv heads · 128 dim · 2 bytes = 131072 bytes
  assert.equal(kvBytesPerToken(cfg, "fp16"), 128 * 1024);
});

test("the 8B GQA example at 128k ctx costs exactly 16 GiB at fp16", () => {
  const cfg = loadConfig(path.join(EXAMPLES, "gqa-8b.json"));
  const kv = estimateKv(cfg, "fp16", 131072, 1);
  assert.equal(kv.bytes, 16 * 1024 ** 3);
});

test("the MLA example costs exactly 67.5 KiB per token at fp16", () => {
  const cfg = loadConfig(path.join(EXAMPLES, "mla-moe-236b.json"));
  // 60 layers · (512 + 64) · 2 bytes = 69120 bytes
  assert.equal(kvBytesPerToken(cfg, "fp16"), 69120);
});

test("cache size is linear in ctx (no sliding) and in batch", () => {
  const cfg = normalizeConfig(tinyConfig());
  const a = estimateKv(cfg, "fp16", 1000, 1);
  assert.equal(estimateKv(cfg, "fp16", 3000, 1).bytes, 3 * a.bytes);
  assert.equal(estimateKv(cfg, "fp16", 1000, 8).bytes, 8 * a.bytes);
});

test("fractional-bpw kv dtypes stay exact: q8_0 = 8.5 bits, q4_0 = 4.5 bits", () => {
  const cfg = normalizeConfig(tinyConfig());
  // 2 layers · 64 elements · 8.5 bits · 100 tokens / 8 = 13600 bytes
  assert.equal(estimateKv(cfg, "q8_0", 100, 1).bytes, 13600);
  const fp16 = estimateKv(cfg, "fp16", 4096, 1).bytes;
  assert.equal(estimateKv(cfg, "q4_0", 4096, 1).bytes / fp16, 4.5 / 16);
});

test("sliding windows cap per-layer tokens at min(ctx, window)", () => {
  const cfg = normalizeConfig(tinyConfig({ sliding_window: 512 }));
  const below = estimateKv(cfg, "fp16", 256, 1);
  const at = estimateKv(cfg, "fp16", 512, 1);
  const above = estimateKv(cfg, "fp16", 4096, 1);
  assert.equal(below.windowCapped, false);
  assert.equal(at.windowCapped, false);
  assert.equal(above.windowCapped, true);
  assert.equal(above.bytes, at.bytes); // flat past the window
  assert.equal(at.bytes, 2 * below.bytes); // linear below it
  // bytesPerToken still reports the uncapped all-full-layers figure.
  assert.equal(above.bytesPerToken, 2 * 64 * 2);
});

test("hybrid layouts: full layers keep growing, sliding layers stay capped", () => {
  const cfg = normalizeConfig(
    tinyConfig({ sliding_window: 512, layer_types: ["full_attention", "sliding_attention"] }),
  );
  const kv = estimateKv(cfg, "fp16", 4096, 1);
  assert.equal(kv.fullLayers, 1);
  assert.equal(kv.slidingLayers, 1);
  // full layer: 4096 tokens · 64 elems · 2B = 524288; sliding layer: 512 · 64 · 2 = 65536
  assert.equal(kv.bytes, 524288 + 65536);
});

test("the bundled hybrid example: 8 full + 40 sliding layers", () => {
  const cfg = loadConfig(path.join(EXAMPLES, "swa-hybrid-12b.json"));
  const kv = estimateKv(cfg, "fp16", 131072, 1);
  assert.equal(kv.fullLayers, 8);
  assert.equal(kv.slidingLayers, 40);
  // 8 full · 131072 + 40 sliding · 1024 tokens, × (2·8·256 elems · 2 B)
  assert.equal(kv.bytes, (8 * 131072 + 40 * 1024) * 2 * 8 * 256 * 2);
});

test("ctx 0 costs 0 bytes; invalid ctx/batch are rejected", () => {
  const cfg = normalizeConfig(tinyConfig());
  assert.equal(estimateKv(cfg, "fp16", 0, 1).bytes, 0);
  assert.throws(() => estimateKv(cfg, "fp16", -1, 1), RangeError);
  assert.throws(() => estimateKv(cfg, "fp16", 10.5, 1), RangeError);
  assert.throws(() => estimateKv(cfg, "fp16", 1024, 0), RangeError);
});
