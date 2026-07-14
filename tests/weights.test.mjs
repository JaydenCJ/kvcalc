/**
 * weights.test.mjs — parameter counting against paper arithmetic.
 *
 * Every expected number here was worked out by hand from tensor shapes and is
 * shown in the comments, so a regression can be diagnosed by re-doing the
 * arithmetic — no "expected = actual at time of writing" snapshots.
 */

import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { normalizeConfig, loadConfig, countParams, estimateWeights } from "../dist/index.js";
import { tinyConfig, EXAMPLES } from "./helpers.mjs";

test("tiny GQA config: every component matches paper arithmetic", () => {
  const p = countParams(normalizeConfig(tinyConfig()));
  // embed = vocab·h = 100·64
  assert.equal(p.embeddings, 6400);
  // per layer: q 64·4·16=4096, k+v 2·64·2·16=4096, o 4·16·64=4096 → 12288; ×2 layers
  assert.equal(p.attention, 24576);
  // per layer: 3·64·128 = 24576; ×2 layers
  assert.equal(p.mlp, 49152);
  // 2 norms ×2 layers ×64 + final 64
  assert.equal(p.norms, 320);
  assert.equal(p.lmHead, 6400);
  assert.equal(p.router, 0);
  assert.equal(p.total, 6400 + 24576 + 49152 + 320 + 6400);
  assert.equal(p.active, p.total); // dense model: active == total
});

test("tie_word_embeddings drops the LM head", () => {
  const tied = countParams(normalizeConfig(tinyConfig({ tie_word_embeddings: true })));
  const untied = countParams(normalizeConfig(tinyConfig()));
  assert.equal(tied.lmHead, 0);
  assert.equal(untied.total - tied.total, 6400);
});

test("attention_bias adds exactly the q/k/v output dims per layer", () => {
  const plain = countParams(normalizeConfig(tinyConfig()));
  const biased = countParams(normalizeConfig(tinyConfig({ attention_bias: true })));
  // per layer: q 4·16 + k 2·16 + v 2·16 = 128; ×2 layers = 256
  assert.equal(biased.attention - plain.attention, 256);
});

test("MLA attention: low-rank q and joint kv projections, counted exactly", () => {
  const cfg = normalizeConfig(
    tinyConfig({
      num_hidden_layers: 1,
      kv_lora_rank: 32,
      q_lora_rank: 24,
      qk_rope_head_dim: 8,
      qk_nope_head_dim: 16,
      v_head_dim: 16,
    }),
  );
  const p = countParams(cfg);
  // q: 64·24 + 24 (latent norm) + 24·4·(16+8) = 1536 + 24 + 2304 = 3864
  // kv: 64·(32+8) + 32 (latent norm) + 32·4·(16+16) = 2560 + 32 + 4096 = 6688
  // o: 4·16·64 = 4096
  assert.equal(p.attention, 3864 + 6688 + 4096);
});

test("MLA without q_lora_rank uses a dense q projection", () => {
  const cfg = normalizeConfig(
    tinyConfig({
      num_hidden_layers: 1,
      kv_lora_rank: 32,
      qk_rope_head_dim: 8,
      qk_nope_head_dim: 16,
      v_head_dim: 16,
    }),
  );
  const p = countParams(cfg);
  // q dense: 64·4·(16+8) = 6144; kv and o as above
  assert.equal(p.attention, 6144 + 6688 + 4096);
});

test("MoE: routed + shared experts, router gates, first_k_dense_replace, active gap", () => {
  const cfg = normalizeConfig(
    tinyConfig({
      num_hidden_layers: 4,
      n_routed_experts: 8,
      num_experts_per_tok: 2,
      moe_intermediate_size: 32,
      n_shared_experts: 1,
      first_k_dense_replace: 1,
    }),
  );
  const p = countParams(cfg);
  // dense layer: 3·64·128 = 24576 (layer 0)
  // expert: 3·64·32 = 6144; 3 MoE layers: routed 3·8·6144 = 147456, shared 3·1·6144 = 18432
  assert.equal(p.mlp, 24576 + 147456 + 18432);
  // router: 3 layers · 64·8
  assert.equal(p.router, 1536);
  // active mlp swaps routed 8 experts for the top-2: gap = 3·(8−2)·6144
  assert.equal(p.total - p.active, 147456 - 36864);
  // all-experts-active edge: no gap
  const all = countParams(
    normalizeConfig(tinyConfig({ num_experts: 4, num_experts_per_tok: 4, moe_intermediate_size: 32 })),
  );
  assert.equal(all.active, all.total);
});

test("the bundled 8B GQA example lands on the published parameter count", () => {
  const cfg = loadConfig(path.join(EXAMPLES, "gqa-8b.json"));
  const p = countParams(cfg);
  // embed 128256·4096 = 525336576 (×2, untied) + attn 32·41943040 + mlp 32·176160768 + norms 266240
  assert.equal(p.total, 8030261248);
});

test("the bundled MLA-MoE example lands in the 236B class with ~21B active", () => {
  const cfg = loadConfig(path.join(EXAMPLES, "mla-moe-236b.json"));
  const p = countParams(cfg);
  assert.ok(Math.abs(p.total - 235.7e9) < 1e9, `total ${p.total}`);
  assert.ok(Math.abs(p.active - 21.4e9) < 1e9, `active ${p.active}`);
});

test("the bundled A3B MoE example: ~30.5B total, ~3.3B active", () => {
  const cfg = loadConfig(path.join(EXAMPLES, "moe-a3b.json"));
  const p = countParams(cfg);
  assert.ok(Math.abs(p.total - 30.5e9) < 0.2e9, `total ${p.total}`);
  assert.ok(Math.abs(p.active - 3.35e9) < 0.1e9, `active ${p.active}`);
});

test("estimateWeights: bytes = params × bpw / 8 exactly; canonical name and approx flag surface", () => {
  const cfg = normalizeConfig(tinyConfig());
  const total = countParams(cfg).total;
  assert.equal(estimateWeights(cfg, "fp16").bytes, total * 2);
  assert.equal(estimateWeights(cfg, "q4_0").bytes, (total * 4.5) / 8);
  assert.equal(estimateWeights(cfg, "q8_0").bytes, (total * 8.5) / 8);
  const w = estimateWeights(cfg, "q4km");
  assert.equal(w.dtype, "q4_K_M");
  assert.equal(w.approx, true);
  assert.equal(w.bitsPerWeight, 4.85);
});
