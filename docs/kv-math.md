# The math kvcalc runs

Everything kvcalc prints reduces to the formulas on this page. All byte
figures are binary (1 GiB = 2³⁰ B); `GB` on the command line is read as GiB
because that is what GPU spec sheets mean.

## KV cache

Per **token**, per **layer**, the cache stores:

| Attention | Cached elements per layer | Config keys |
|---|---|---|
| MHA | `2 · num_heads · head_dim` | `num_attention_heads`, `head_dim` |
| GQA | `2 · num_key_value_heads · head_dim` | `num_key_value_heads` |
| MLA | `kv_lora_rank + qk_rope_head_dim` | `kv_lora_rank`, `qk_rope_head_dim` |

MLA caches the *compressed latent* plus the rotary key; per-head keys and
values are reconstructed at attention time and never stored. That is why a
128-head MLA model can cache fewer bytes per token than an 8-kv-head GQA
model — and why calculators that apply the GQA formula to MLA configs
overestimate by an order of magnitude.

Total cache:

```text
kv_bytes(ctx, batch) = batch · Σ_layers  tokens(layer) · elements(layer) · bits(kv_dtype) / 8
tokens(layer)        = min(ctx, sliding_window)   if the layer slides
                     = ctx                        otherwise
```

Sliding layers are resolved per layer: an explicit `layer_types` list wins;
else `sliding_window_pattern: N` marks every Nth layer full; else a plain
`sliding_window` puts every layer on the window; `use_sliding_window: false`
disables it. Bits are summed before the final division by 8, so fractional
bits-per-element cache dtypes (q8_0 = 8.5) stay exact.

## Weights

Parameters are counted from tensor shapes, per component:

```text
embeddings = vocab_size · hidden
lm_head    = vocab_size · hidden        (0 when tie_word_embeddings)
norms      = layers · 2 · hidden + hidden
attention  (GQA/MHA, per layer)
           = hidden·heads·head_dim            (q)
           + 2·hidden·kv_heads·head_dim       (k, v)
           + heads·head_dim·hidden            (o)
           + [heads·head_dim + 2·kv_heads·head_dim  if attention_bias]
attention  (MLA, per layer)
           = hidden·q_lora + q_lora + q_lora·heads·(nope+rope)     (q path)
           + hidden·(kv_lora+rope) + kv_lora
           + kv_lora·heads·(nope+v_dim)                            (kv path)
           + heads·v_dim·hidden                                    (o)
mlp        (dense, per layer) = 3 · hidden · intermediate_size
mlp        (MoE, per layer)   = experts · 3·hidden·moe_intermediate
                              + shared_experts · 3·hidden·moe_intermediate
router     (per MoE layer)    = hidden · experts
```

The first `first_k_dense_replace` layers of an MoE model use the dense MLP.
"Active" parameters replace `experts` with `num_experts_per_tok` (shared
experts and the router always count). Weight bytes are
`total_params · bits(weight_dtype) / 8`.

## Bits per weight

Block quants store groups of weights plus scales; the effective bits per
weight follow from the block layout and are exact:

| dtype | block | bytes/block | bpw |
|---|---|---|---|
| q8_0 | 32 | 34 | 8.5 |
| q5_1 | 32 | 24 | 6.0 |
| q5_0 | 32 | 22 | 5.5 |
| q4_1 | 32 | 20 | 5.0 |
| q4_0 | 32 | 18 | 4.5 |
| q6_K | 256 | 210 | 6.5625 |
| q5_K | 256 | 176 | 5.5 |
| q4_K | 256 | 144 | 4.5 |
| q3_K | 256 | 110 | 3.4375 |
| q2_K | 256 | 82 | 2.5625 |

The mixed presets (`q4_K_M` ≈ 4.85, `q5_K_M` ≈ 5.69, `q3_K_M` ≈ 3.91) blend
two block types across tensor roles, so their effective bpw varies slightly
with model geometry; kvcalc uses measured averages and marks them `~` in
output. 256-wide super-block quants are rejected as *cache* dtypes — cache
lines are written per token and the runtimes only offer 32-wide blocks there.

## Honest limits

- kvcalc models **weights + KV cache**, the two terms that dominate and
  scale. Runtime overhead (CUDA/Metal context, activation workspace,
  fragmentation, graph buffers) is real but runtime- and version-specific;
  budget for it explicitly with `--overhead` (1–2 GiB is a common allowance).
- Quantized *file* sizes can deviate a couple of percent from
  `params × bpw`: embeddings and output tensors are often kept at higher
  precision, and mixed presets vary by geometry.
- Parameter counts assume gated MLPs and RMS norms (the universal choice in
  current decoder-only models); exotic blocks will be counted approximately.
- `sliding_window` describes the *model*. Some runtimes still allocate the
  full rectangle; treat sliding-window savings as an upper bound unless your
  runtime documents ring buffers.
