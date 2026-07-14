/**
 * Parameter counting and weight-memory estimation.
 *
 * Counts are derived from tensor shapes, component by component, so the
 * breakdown is auditable: embeddings, attention projections (GQA or MLA
 * shapes), MLPs (dense or expert), norms, router gates and the LM head.
 * Rotary embeddings contribute no parameters; layer norms are RMS-style
 * (one weight vector, no bias), which matches every architecture in scope.
 */

import { resolveDtype } from "./dtype.js";
import type { NormalizedConfig, ParamBreakdown, WeightEstimate } from "./types.js";

/** Attention parameters for one layer. */
function attentionParamsPerLayer(cfg: NormalizedConfig): number {
  const h = cfg.hiddenSize;
  if (cfg.mla) {
    const { kvLoraRank, qLoraRank, qkRopeHeadDim, qkNopeHeadDim, vHeadDim } = cfg.mla;
    const qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;
    // Query path: either dense (h → heads·qkHeadDim) or low-rank with an RMS norm on the latent.
    const q =
      qLoraRank === null
        ? h * cfg.numHeads * qkHeadDim
        : h * qLoraRank + qLoraRank + qLoraRank * cfg.numHeads * qkHeadDim;
    // KV path: joint down-projection to (latent + rope key), RMS norm on the latent,
    // then up-projection from the latent to per-head nope-keys and values.
    const kv = h * (kvLoraRank + qkRopeHeadDim) + kvLoraRank + kvLoraRank * cfg.numHeads * (qkNopeHeadDim + vHeadDim);
    const o = cfg.numHeads * vHeadDim * h;
    return q + kv + o;
  }
  const qkvOut = cfg.numHeads * cfg.headDim + 2 * cfg.numKvHeads * cfg.headDim;
  const q = h * cfg.numHeads * cfg.headDim;
  const kv = 2 * h * cfg.numKvHeads * cfg.headDim;
  const o = cfg.numHeads * cfg.headDim * h;
  const bias = cfg.attentionBias ? qkvOut : 0;
  return q + kv + o + bias;
}

/** A gated MLP (gate + up + down) with the given intermediate size. */
function gatedMlpParams(hidden: number, intermediate: number): number {
  return 3 * hidden * intermediate;
}

/** Count parameters, broken down by component. */
export function countParams(cfg: NormalizedConfig): ParamBreakdown {
  const h = cfg.hiddenSize;
  const embeddings = cfg.vocabSize * h;
  const lmHead = cfg.tieWordEmbeddings ? 0 : cfg.vocabSize * h;
  const attention = cfg.numLayers * attentionParamsPerLayer(cfg);
  // Two RMS norms per layer (pre-attention, pre-MLP) plus the final norm.
  const norms = cfg.numLayers * 2 * h + h;

  let mlp = 0;
  let router = 0;
  let activeMlp = 0;
  if (cfg.moe) {
    const { numExperts, expertsPerTok, moeIntermediateSize, numSharedExperts, firstKDense } = cfg.moe;
    const denseLayers = Math.min(firstKDense, cfg.numLayers);
    const moeLayers = cfg.numLayers - denseLayers;
    const expertParams = gatedMlpParams(h, moeIntermediateSize);
    const densePart = denseLayers * gatedMlpParams(h, cfg.intermediateSize);
    const sharedPart = moeLayers * numSharedExperts * expertParams;
    mlp = densePart + moeLayers * numExperts * expertParams + sharedPart;
    router = moeLayers * h * numExperts;
    activeMlp = densePart + moeLayers * expertsPerTok * expertParams + sharedPart;
  } else {
    mlp = cfg.numLayers * gatedMlpParams(h, cfg.intermediateSize);
    activeMlp = mlp;
  }

  const total = embeddings + attention + mlp + norms + router + lmHead;
  const active = embeddings + attention + activeMlp + norms + router + lmHead;
  return { embeddings, attention, mlp, norms, lmHead, router, total, active };
}

/** Weight memory at a given storage dtype. */
export function estimateWeights(cfg: NormalizedConfig, dtypeName: string): WeightEstimate {
  const dtype = resolveDtype(dtypeName, "weights");
  const params = countParams(cfg);
  return {
    params,
    dtype: dtype.name,
    bitsPerWeight: dtype.bitsPerWeight,
    bytes: (params.total * dtype.bitsPerWeight) / 8,
    approx: dtype.approx,
  };
}
