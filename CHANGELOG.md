# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-07-13

### Added

- The `report` command: weight memory + KV-cache memory for any
  (context, batch) point, with a component-level parameter breakdown
  (embeddings, attention, MLP, router, LM head) and an optional `--vram`
  verdict with signed headroom.
- Key-driven config.json normalization: GQA/MHA head grouping,
  `head_dim` derivation, alias keys (`num_kv_heads`, `num_experts`,
  `n_routed_experts`, `num_local_experts`), `text_config` nesting for
  multimodal configs, typed errors for every malformed input.
- MLA (Multi-head Latent Attention) support: caches
  `kv_lora_rank + qk_rope_head_dim` elements per layer per token instead
  of per-head K/V, and counts the low-rank q/kv projection weights
  exactly — the compressed-cache math that generic calculators miss.
- Sliding-window attention support: plain global windows,
  `sliding_window_pattern` interleaves, explicit `layer_types` lists and
  `use_sliding_window: false` opt-outs, with per-layer token capping at
  `min(ctx, window)`.
- Mixture-of-Experts support: routed + shared experts, router gates,
  `first_k_dense_replace` dense tails, and an "active params" figure
  alongside the total.
- A bits-per-weight registry with receipts: exact block-layout math for
  q8_0/q5_1/q5_0/q4_1/q4_0 and the k-quant super-blocks, flagged
  empirical averages for the mixed q4_K_M/q5_K_M/q3_K_M presets, plus
  float and integer dtypes; `kvcalc dtypes` prints the whole table.
- The `table` command: a context × kv-dtype grid of totals with ✓/✗
  marks against a `--vram` budget, defaulting to a 2k→model-max sweep.
- The `fit` command: exact integer binary search for the largest context
  that fits a budget at a given batch, weight dtype, kv dtype and
  `--overhead`, compared against the model's own maximum.
- Script-friendly contract: `--json` on every command, byte-identical
  output for identical inputs, exit codes 0 (fits) / 1 (a `--vram` check
  failed) / 2 (usage or config error).
- Four committed example configs spanning the architecture space: dense
  GQA 8B, MLA + MoE 236B, hybrid sliding-window 12B (nested
  `text_config`), and a 30B-A3B MoE.
- Test suite: 87 node:test tests (unit + CLI integration, every expected
  number hand-derived in comments) and an end-to-end `scripts/smoke.sh`
  against the bundled examples.

[0.1.0]: https://github.com/JaydenCJ/kvcalc/releases/tag/v0.1.0
