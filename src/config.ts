/**
 * Loading and normalizing HuggingFace-style config.json files.
 *
 * kvcalc is deliberately key-driven, not name-driven: it never dispatches on
 * `model_type`. If a config carries `kv_lora_rank` it gets the MLA cache
 * formula, if it carries `num_key_value_heads < num_attention_heads` it gets
 * GQA, and so on. New architectures that reuse these keys work unmodified.
 */

import { readFileSync } from "node:fs";
import type {
  LayerAttentionType,
  MlaGeometry,
  MoeGeometry,
  NormalizedConfig,
  SlidingLayout,
} from "./types.js";

export class ConfigError extends Error {}

type Raw = Record<string, unknown>;

function num(raw: Raw, key: string): number | undefined {
  const v = raw[key];
  if (v === undefined || v === null) return undefined;
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new ConfigError(`config key "${key}" must be a number, got ${JSON.stringify(v)}`);
  }
  return v;
}

function bool(raw: Raw, key: string): boolean | undefined {
  const v = raw[key];
  if (v === undefined || v === null) return undefined;
  if (typeof v !== "boolean") {
    throw new ConfigError(`config key "${key}" must be a boolean, got ${JSON.stringify(v)}`);
  }
  return v;
}

/** First defined numeric value among several alias keys. */
function firstNum(raw: Raw, keys: string[]): number | undefined {
  for (const k of keys) {
    const v = num(raw, k);
    if (v !== undefined) return v;
  }
  return undefined;
}

function detectMla(raw: Raw, warnings: string[]): MlaGeometry | null {
  const kvLoraRank = num(raw, "kv_lora_rank");
  if (kvLoraRank === undefined || kvLoraRank <= 0) return null;
  const qkRopeHeadDim = num(raw, "qk_rope_head_dim");
  const qkNopeHeadDim = num(raw, "qk_nope_head_dim");
  const vHeadDim = num(raw, "v_head_dim");
  if (qkRopeHeadDim === undefined || qkNopeHeadDim === undefined || vHeadDim === undefined) {
    throw new ConfigError(
      "config declares kv_lora_rank (MLA) but is missing qk_rope_head_dim, qk_nope_head_dim or v_head_dim",
    );
  }
  const qLoraRank = num(raw, "q_lora_rank") ?? null;
  if (qLoraRank === null) {
    warnings.push("MLA config has no q_lora_rank; assuming a dense query projection");
  }
  return { kvLoraRank, qLoraRank, qkRopeHeadDim, qkNopeHeadDim, vHeadDim };
}

function detectMoe(raw: Raw, intermediateSize: number, warnings: string[]): MoeGeometry | null {
  const numExperts = firstNum(raw, ["n_routed_experts", "num_local_experts", "num_experts"]);
  if (numExperts === undefined || numExperts <= 1) return null;
  let expertsPerTok = firstNum(raw, ["num_experts_per_tok", "top_k"]);
  if (expertsPerTok === undefined) {
    expertsPerTok = 2;
    warnings.push("MoE config has no num_experts_per_tok; assuming 2 active experts per token");
  }
  const moeIntermediateSize = num(raw, "moe_intermediate_size") ?? intermediateSize;
  const numSharedExperts = num(raw, "n_shared_experts") ?? 0;
  const firstKDense = num(raw, "first_k_dense_replace") ?? 0;
  return { numExperts, expertsPerTok, moeIntermediateSize, numSharedExperts, firstKDense };
}

function resolveSliding(raw: Raw, numLayers: number, warnings: string[]): SlidingLayout {
  const allFull: SlidingLayout = {
    window: null,
    layerTypes: new Array<LayerAttentionType>(numLayers).fill("full"),
  };

  // Explicit per-layer list wins over everything else.
  const layerTypesRaw = raw["layer_types"];
  if (Array.isArray(layerTypesRaw)) {
    if (layerTypesRaw.length !== numLayers) {
      throw new ConfigError(
        `layer_types has ${layerTypesRaw.length} entries but num_hidden_layers is ${numLayers}`,
      );
    }
    const layerTypes = layerTypesRaw.map((t): LayerAttentionType => {
      if (t === "full_attention") return "full";
      if (t === "sliding_attention") return "sliding";
      throw new ConfigError(`unknown layer_types entry ${JSON.stringify(t)}`);
    });
    const anySliding = layerTypes.includes("sliding");
    const window = num(raw, "sliding_window");
    if (anySliding && (window === undefined || window <= 0)) {
      throw new ConfigError("layer_types declares sliding_attention but sliding_window is missing");
    }
    return { window: anySliding ? window! : null, layerTypes };
  }

  const window = num(raw, "sliding_window");
  if (window === undefined || window <= 0) return allFull;
  if (bool(raw, "use_sliding_window") === false) {
    warnings.push("sliding_window is set but use_sliding_window is false; treating all layers as full attention");
    return allFull;
  }

  // Gemma-style "1 full layer every N" interleave.
  const pattern = num(raw, "sliding_window_pattern");
  if (pattern !== undefined && pattern > 1) {
    const layerTypes = Array.from(
      { length: numLayers },
      (_, i): LayerAttentionType => ((i + 1) % pattern === 0 ? "full" : "sliding"),
    );
    return { window, layerTypes };
  }

  // Plain global window: every layer slides.
  return {
    window,
    layerTypes: new Array<LayerAttentionType>(numLayers).fill("sliding"),
  };
}

/** Normalize a parsed config.json object. Throws ConfigError on missing/invalid keys. */
export function normalizeConfig(rawInput: unknown): NormalizedConfig {
  if (typeof rawInput !== "object" || rawInput === null || Array.isArray(rawInput)) {
    throw new ConfigError("config.json must be a JSON object");
  }
  let raw = rawInput as Raw;
  const modelType = typeof raw["model_type"] === "string" ? (raw["model_type"] as string) : "unknown";

  // Multimodal configs nest the language model under text_config.
  const textConfig = raw["text_config"];
  if (
    typeof textConfig === "object" &&
    textConfig !== null &&
    !Array.isArray(textConfig) &&
    (textConfig as Raw)["hidden_size"] !== undefined
  ) {
    raw = textConfig as Raw;
  }

  const warnings: string[] = [];
  const missing = ["hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size"].filter(
    (k) => num(raw, k) === undefined,
  );
  if (missing.length > 0) {
    throw new ConfigError(
      `config.json is missing required ${missing.length === 1 ? "key" : "keys"}: ${missing.join(", ")} — is this a text-model config?`,
    );
  }

  const hiddenSize = num(raw, "hidden_size")!;
  const numLayers = num(raw, "num_hidden_layers")!;
  const numHeads = num(raw, "num_attention_heads")!;
  const vocabSize = num(raw, "vocab_size")!;
  const intermediateSize = num(raw, "intermediate_size") ?? 4 * hiddenSize;
  if (num(raw, "intermediate_size") === undefined) {
    warnings.push(`no intermediate_size; assuming 4 × hidden_size = ${intermediateSize}`);
  }

  const mla = detectMla(raw, warnings);

  let headDim = firstNum(raw, ["head_dim", "attention_head_dim"]);
  if (headDim === undefined) {
    if (mla) {
      headDim = mla.qkNopeHeadDim + mla.qkRopeHeadDim;
    } else {
      headDim = hiddenSize / numHeads;
      if (!Number.isInteger(headDim)) {
        throw new ConfigError(
          `cannot derive head_dim: hidden_size ${hiddenSize} is not divisible by num_attention_heads ${numHeads}`,
        );
      }
    }
  }

  const numKvHeads = firstNum(raw, ["num_key_value_heads", "num_kv_heads"]) ?? numHeads;
  if (numKvHeads > numHeads || (mla === null && numHeads % numKvHeads !== 0)) {
    warnings.push(
      `unusual head grouping: ${numHeads} query heads over ${numKvHeads} kv heads`,
    );
  }

  const attentionKind = mla ? "mla" : numKvHeads < numHeads ? "gqa" : "mha";

  for (const key of ["hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size"]) {
    const v = num(raw, key)!;
    if (!Number.isInteger(v) || v <= 0) {
      throw new ConfigError(`config key "${key}" must be a positive integer, got ${v}`);
    }
  }

  return {
    modelType,
    hiddenSize,
    numLayers,
    numHeads,
    numKvHeads,
    headDim,
    vocabSize,
    intermediateSize,
    tieWordEmbeddings: bool(raw, "tie_word_embeddings") ?? false,
    attentionBias: bool(raw, "attention_bias") ?? bool(raw, "qkv_bias") ?? false,
    maxPositionEmbeddings: num(raw, "max_position_embeddings") ?? null,
    attentionKind,
    mla,
    sliding: resolveSliding(raw, numLayers, warnings),
    moe: detectMoe(raw, intermediateSize, warnings),
    warnings,
  };
}

/** Read and normalize a config.json from disk. */
export function loadConfig(path: string): NormalizedConfig {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (err) {
    throw new ConfigError(`cannot read ${path}: ${(err as Error).message}`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new ConfigError(`${path} is not valid JSON: ${(err as Error).message}`);
  }
  return normalizeConfig(parsed);
}
