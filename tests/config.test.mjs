/**
 * config.test.mjs — normalization of HuggingFace-style config.json objects.
 *
 * The normalizer is key-driven: MLA is whatever declares kv_lora_rank, GQA is
 * whatever has fewer kv heads than query heads. These tests cover the alias
 * keys, defaulting rules, nesting and every hard error path.
 */

import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { normalizeConfig, loadConfig, ConfigError } from "../dist/index.js";
import { tinyConfig } from "./helpers.mjs";

test("normalizes a plain GQA config with no warnings", () => {
  const cfg = normalizeConfig(tinyConfig());
  assert.equal(cfg.attentionKind, "gqa");
  assert.equal(cfg.numKvHeads, 2);
  assert.equal(cfg.headDim, 16);
  assert.equal(cfg.maxPositionEmbeddings, 4096);
  assert.deepEqual(cfg.warnings, []);
});

test("kv heads default to query heads (MHA); num_kv_heads is accepted as an alias", () => {
  const mha = normalizeConfig(tinyConfig({ num_key_value_heads: undefined }));
  assert.equal(mha.attentionKind, "mha");
  assert.equal(mha.numKvHeads, 4);
  const alias = normalizeConfig(tinyConfig({ num_key_value_heads: undefined, num_kv_heads: 2 }));
  assert.equal(alias.numKvHeads, 2);
  assert.equal(alias.attentionKind, "gqa");
});

test("head_dim derives from hidden/heads when absent; an explicit value wins", () => {
  assert.equal(normalizeConfig(tinyConfig({ head_dim: undefined })).headDim, 64 / 4);
  assert.equal(normalizeConfig(tinyConfig({ head_dim: 256 })).headDim, 256);
});

test("non-integral derived head_dim is a hard error, not a silent truncation", () => {
  assert.throws(
    () => normalizeConfig(tinyConfig({ head_dim: undefined, hidden_size: 65 })),
    ConfigError,
  );
});

test("missing required keys are reported by name, all at once", () => {
  try {
    normalizeConfig({ hidden_size: 64 });
    assert.fail("should have thrown");
  } catch (err) {
    assert.ok(err instanceof ConfigError);
    assert.match(err.message, /num_hidden_layers/);
    assert.match(err.message, /num_attention_heads/);
    assert.match(err.message, /vocab_size/);
  }
});

test("missing intermediate_size defaults to 4 × hidden with a warning", () => {
  const cfg = normalizeConfig(tinyConfig({ intermediate_size: undefined }));
  assert.equal(cfg.intermediateSize, 256);
  assert.equal(cfg.warnings.length, 1);
  assert.match(cfg.warnings[0], /intermediate_size/);
});

test("non-object inputs and wrongly-typed keys are rejected with the key name", () => {
  assert.throws(() => normalizeConfig(null), ConfigError);
  assert.throws(() => normalizeConfig([1, 2]), ConfigError);
  assert.throws(() => normalizeConfig("config"), ConfigError);
  assert.throws(() => normalizeConfig(tinyConfig({ hidden_size: "64" })), /hidden_size/);
  assert.throws(() => normalizeConfig(tinyConfig({ tie_word_embeddings: 1 })), /tie_word_embeddings/);
  assert.throws(() => normalizeConfig(tinyConfig({ num_hidden_layers: 2.5 })), ConfigError);
});

test("multimodal configs: the text_config nesting is flattened", () => {
  const cfg = normalizeConfig({
    model_type: "vlm",
    vision_config: { hidden_size: 1152 },
    text_config: tinyConfig({ hidden_size: 128, num_attention_heads: 8, head_dim: 16 }),
  });
  assert.equal(cfg.hiddenSize, 128);
  assert.equal(cfg.modelType, "vlm"); // outer model_type is kept
});

test("MLA: kv_lora_rank switches the attention kind and captures geometry", () => {
  const cfg = normalizeConfig(
    tinyConfig({
      head_dim: undefined,
      kv_lora_rank: 32,
      q_lora_rank: 24,
      qk_rope_head_dim: 8,
      qk_nope_head_dim: 16,
      v_head_dim: 16,
    }),
  );
  assert.equal(cfg.attentionKind, "mla");
  assert.deepEqual(cfg.mla, {
    kvLoraRank: 32,
    qLoraRank: 24,
    qkRopeHeadDim: 8,
    qkNopeHeadDim: 16,
    vHeadDim: 16,
  });
  // MLA head_dim = nope + rope, not hidden/heads.
  assert.equal(cfg.headDim, 24);
});

test("MLA: missing geometry keys are a hard error; missing q_lora_rank only warns", () => {
  assert.throws(
    () => normalizeConfig(tinyConfig({ kv_lora_rank: 32, qk_rope_head_dim: 8 })),
    /qk_nope_head_dim/,
  );
  const dense = normalizeConfig(
    tinyConfig({ kv_lora_rank: 32, qk_rope_head_dim: 8, qk_nope_head_dim: 16, v_head_dim: 16 }),
  );
  assert.equal(dense.mla.qLoraRank, null);
  assert.match(dense.warnings.join(" "), /q_lora_rank/);
});

test("MoE: all three expert-count keys are accepted; a single expert is just dense", () => {
  for (const key of ["n_routed_experts", "num_local_experts", "num_experts"]) {
    const cfg = normalizeConfig(tinyConfig({ [key]: 8, num_experts_per_tok: 2 }));
    assert.equal(cfg.moe.numExperts, 8, key);
  }
  assert.equal(normalizeConfig(tinyConfig({ num_experts: 1 })).moe, null);
});

test("MoE defaults: experts_per_tok warns at 2; moe_intermediate_size falls back", () => {
  const cfg = normalizeConfig(tinyConfig({ num_experts: 8 }));
  assert.equal(cfg.moe.expertsPerTok, 2);
  assert.equal(cfg.moe.numSharedExperts, 0);
  assert.equal(cfg.moe.firstKDense, 0);
  assert.equal(cfg.moe.moeIntermediateSize, 128); // falls back to intermediate_size
  assert.match(cfg.warnings.join(" "), /num_experts_per_tok/);
  const explicit = normalizeConfig(
    tinyConfig({ num_experts: 8, num_experts_per_tok: 2, moe_intermediate_size: 32 }),
  );
  assert.equal(explicit.moe.moeIntermediateSize, 32);
});

test("sliding: no window keys → all full; a plain global window → all sliding", () => {
  const dense = normalizeConfig(tinyConfig());
  assert.equal(dense.sliding.window, null);
  assert.deepEqual(dense.sliding.layerTypes, ["full", "full"]);
  const swa = normalizeConfig(tinyConfig({ sliding_window: 512 }));
  assert.equal(swa.sliding.window, 512);
  assert.deepEqual(swa.sliding.layerTypes, ["sliding", "sliding"]);
});

test("sliding: use_sliding_window=false disables the window with a warning", () => {
  const cfg = normalizeConfig(tinyConfig({ sliding_window: 512, use_sliding_window: false }));
  assert.equal(cfg.sliding.window, null);
  assert.deepEqual(cfg.sliding.layerTypes, ["full", "full"]);
  assert.match(cfg.warnings.join(" "), /use_sliding_window/);
});

test("sliding: sliding_window_pattern N makes every Nth layer full", () => {
  const cfg = normalizeConfig(
    tinyConfig({ num_hidden_layers: 6, sliding_window: 512, sliding_window_pattern: 3 }),
  );
  assert.deepEqual(cfg.sliding.layerTypes, ["sliding", "sliding", "full", "sliding", "sliding", "full"]);
});

test("sliding: explicit layer_types wins, must match the layer count, entries validated", () => {
  const cfg = normalizeConfig(
    tinyConfig({ sliding_window: 512, layer_types: ["full_attention", "sliding_attention"] }),
  );
  assert.deepEqual(cfg.sliding.layerTypes, ["full", "sliding"]);
  assert.throws(
    () => normalizeConfig(tinyConfig({ sliding_window: 512, layer_types: ["full_attention"] })),
    /layer_types has 1 entries/,
  );
  assert.throws(
    () => normalizeConfig(tinyConfig({ layer_types: ["full_attention", "chunked_attention"] })),
    /chunked_attention/,
  );
});

test("sliding: layer_types with sliding entries but no window is a hard error", () => {
  assert.throws(
    () => normalizeConfig(tinyConfig({ layer_types: ["full_attention", "sliding_attention"] })),
    /sliding_window is missing/,
  );
});

test("loadConfig: reads a file, rejects missing files and invalid JSON", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "kvcalc-test-"));
  try {
    const good = path.join(dir, "config.json");
    writeFileSync(good, JSON.stringify(tinyConfig()));
    assert.equal(loadConfig(good).hiddenSize, 64);

    assert.throws(() => loadConfig(path.join(dir, "nope.json")), /cannot read/);

    const bad = path.join(dir, "bad.json");
    writeFileSync(bad, "{not json");
    assert.throws(() => loadConfig(bad), /not valid JSON/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
