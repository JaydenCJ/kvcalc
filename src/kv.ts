/**
 * KV-cache math — the heart of kvcalc.
 *
 * Per token, per layer, the cache holds:
 *   - MHA/GQA:  2 · num_kv_heads · head_dim          elements (K and V)
 *   - MLA:      kv_lora_rank + qk_rope_head_dim      elements (the compressed
 *               latent plus the rotary key — values are reconstructed, not cached)
 *
 * Sliding-window layers cap their cached tokens at min(ctx, window); full
 * layers cache the whole context. Everything scales linearly with batch.
 * Sizes are computed in bits and divided by 8 once at the end, so fractional
 * bits-per-weight quants (q8_0 = 8.5) stay exact.
 */

import { resolveDtype } from "./dtype.js";
import type { KvEstimate, NormalizedConfig } from "./types.js";

/** Cached elements per token, per layer. */
export function kvElementsPerLayer(cfg: NormalizedConfig): number {
  if (cfg.mla) return cfg.mla.kvLoraRank + cfg.mla.qkRopeHeadDim;
  return 2 * cfg.numKvHeads * cfg.headDim;
}

/** Cache bytes per token, per sequence, if every layer attended fully. */
export function kvBytesPerToken(cfg: NormalizedConfig, kvDtypeName: string): number {
  const dtype = resolveDtype(kvDtypeName, "kv");
  return (cfg.numLayers * kvElementsPerLayer(cfg) * dtype.bitsPerWeight) / 8;
}

/** Full KV-cache size for a (ctx, batch) point, honoring per-layer sliding windows. */
export function estimateKv(
  cfg: NormalizedConfig,
  kvDtypeName: string,
  ctx: number,
  batch: number,
): KvEstimate {
  if (!Number.isInteger(ctx) || ctx < 0) throw new RangeError(`ctx must be a non-negative integer, got ${ctx}`);
  if (!Number.isInteger(batch) || batch < 1) throw new RangeError(`batch must be a positive integer, got ${batch}`);
  const dtype = resolveDtype(kvDtypeName, "kv");
  const elems = kvElementsPerLayer(cfg);
  const { window, layerTypes } = cfg.sliding;

  let fullLayers = 0;
  let slidingLayers = 0;
  let tokenLayerProduct = 0; // Σ over layers of tokens cached in that layer.
  for (const t of layerTypes) {
    if (t === "sliding" && window !== null) {
      slidingLayers += 1;
      tokenLayerProduct += Math.min(ctx, window);
    } else {
      fullLayers += 1;
      tokenLayerProduct += ctx;
    }
  }

  const bits = tokenLayerProduct * elems * dtype.bitsPerWeight * batch;
  return {
    dtype: dtype.name,
    ctx,
    batch,
    bytesPerToken: (cfg.numLayers * elems * dtype.bitsPerWeight) / 8,
    bytes: bits / 8,
    fullLayers,
    slidingLayers,
    window,
    windowCapped: slidingLayers > 0 && window !== null && ctx > window,
  };
}
