/**
 * Shared types for kvcalc.
 *
 * Everything here is a plain-data description: the config normalizer produces a
 * `NormalizedConfig`, and the pure math modules (weights.ts, kv.ts, fit.ts)
 * consume it without ever touching the filesystem or the process.
 */

/** How the model attends — decides both the KV-cache formula and the attention weight shapes. */
export type AttentionKind = "mha" | "gqa" | "mla";

/** Multi-head Latent Attention (MLA) geometry, as found in DeepSeek-style configs. */
export interface MlaGeometry {
  /** Rank of the compressed KV latent; this (plus the rope dim) is what gets cached. */
  kvLoraRank: number;
  /** Rank of the optional low-rank query projection; null means a dense q projection. */
  qLoraRank: number | null;
  /** Per-head rotary (positional) key dimension — cached alongside the latent. */
  qkRopeHeadDim: number;
  /** Per-head non-rotary query/key dimension — reconstructed from the latent, not cached. */
  qkNopeHeadDim: number;
  /** Per-head value dimension — reconstructed from the latent, not cached. */
  vHeadDim: number;
}

/** Mixture-of-Experts geometry. */
export interface MoeGeometry {
  /** Number of routed experts per MoE layer. */
  numExperts: number;
  /** Experts activated per token (drives the "active params" figure). */
  expertsPerTok: number;
  /** Intermediate size of each expert MLP. */
  moeIntermediateSize: number;
  /** Always-active shared experts per MoE layer (DeepSeek-style); 0 if none. */
  numSharedExperts: number;
  /** The first K layers use a dense MLP instead of experts (DeepSeek-style); 0 if none. */
  firstKDense: number;
}

/** Per-layer attention span: "full" attends to the whole context, "sliding" to a window. */
export type LayerAttentionType = "full" | "sliding";

/** Sliding-window attention layout, resolved to one entry per layer. */
export interface SlidingLayout {
  /** Window size in tokens, or null when no layer slides. */
  window: number | null;
  /** One entry per hidden layer. All "full" when the model has no SWA. */
  layerTypes: LayerAttentionType[];
}

/** A HuggingFace-style config.json, normalized: defaults resolved, nesting flattened. */
export interface NormalizedConfig {
  /** Raw `model_type` from the config, or "unknown". Informational only — math is key-driven. */
  modelType: string;
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  vocabSize: number;
  intermediateSize: number;
  tieWordEmbeddings: boolean;
  /** Whether q/k/v projections carry bias terms (Qwen2-style). */
  attentionBias: boolean;
  maxPositionEmbeddings: number | null;
  attentionKind: AttentionKind;
  /** Present iff attentionKind === "mla". */
  mla: MlaGeometry | null;
  sliding: SlidingLayout;
  /** Present for Mixture-of-Experts models. */
  moe: MoeGeometry | null;
  /** Non-fatal oddities found while normalizing (e.g. a derived head_dim that isn't integral). */
  warnings: string[];
}

/** Parameter count, broken down by component. All values are element counts, not bytes. */
export interface ParamBreakdown {
  embeddings: number;
  attention: number;
  mlp: number;
  norms: number;
  lmHead: number;
  /** MoE router gates; 0 for dense models. */
  router: number;
  total: number;
  /** Parameters touched per token (differs from total only for MoE models). */
  active: number;
}

/** Weight memory for a parameter count at a given storage dtype. */
export interface WeightEstimate {
  params: ParamBreakdown;
  dtype: string;
  bitsPerWeight: number;
  bytes: number;
  /** True when the dtype is a mixed quantization whose bpw is an empirical average. */
  approx: boolean;
}

/** KV-cache cost for one (ctx, batch) point. */
export interface KvEstimate {
  dtype: string;
  ctx: number;
  batch: number;
  /** Bytes per token per sequence if every layer attended fully. */
  bytesPerToken: number;
  /** Total cache bytes for this ctx and batch (sliding windows applied per layer). */
  bytes: number;
  fullLayers: number;
  slidingLayers: number;
  window: number | null;
  /** True when at least one sliding layer is capped below ctx. */
  windowCapped: boolean;
}

/** Result of solving for the largest context that fits a VRAM budget. */
export interface FitResult {
  budgetBytes: number;
  overheadBytes: number;
  weightBytes: number;
  /** Bytes left for KV cache after weights and overhead; negative when weights alone overflow. */
  kvBudgetBytes: number;
  batch: number;
  /** Largest context length (tokens) whose KV cache fits, 0 if none. */
  maxCtx: number;
  /** KV bytes actually used at maxCtx. */
  kvBytesAtMax: number;
  /** Whether the model's own max_position_embeddings fits within the budget. */
  fullContextFits: boolean | null;
}
