/**
 * Storage dtype registry: exact bits-per-weight for floats and block quants,
 * documented empirical averages for the mixed "_M" presets.
 *
 * Block-quant figures are derived from the block layouts themselves
 * (bytes-per-block × 8 / weights-per-block), so they are exact, not folklore:
 * e.g. q4_0 stores 32 weights in 18 bytes → 4.5 bpw; q6_K stores 256 weights
 * in 210 bytes → 6.5625 bpw. Mixed presets (q4_K_M, q5_K_M, q3_K_M) blend two
 * block types across tensor roles, so their effective bpw depends slightly on
 * the model geometry; the registry carries a measured average and flags it.
 */

export interface DtypeInfo {
  /** Canonical name, as printed. */
  name: string;
  /** Bits per stored element. Fractional for block quants (exact binary fractions). */
  bitsPerWeight: number;
  /** True when bitsPerWeight is an empirical average rather than a block-layout constant. */
  approx: boolean;
  /** Usable for model weights. */
  weights: boolean;
  /** Usable for the KV cache. */
  kv: boolean;
  /** One-line provenance of the number. */
  note: string;
}

const D = (
  name: string,
  bitsPerWeight: number,
  weights: boolean,
  kv: boolean,
  note: string,
  approx = false,
): DtypeInfo => ({ name, bitsPerWeight, approx, weights, kv, note });

/** Every dtype kvcalc knows, in display order. */
export const DTYPES: readonly DtypeInfo[] = [
  D("fp32", 32, true, true, "IEEE 754 single precision"),
  D("bf16", 16, true, true, "bfloat16"),
  D("fp16", 16, true, true, "IEEE 754 half precision"),
  D("fp8", 8, true, true, "8-bit float (e4m3/e5m2)"),
  D("int8", 8, true, true, "plain 8-bit integer"),
  D("int4", 4, true, false, "plain 4-bit integer (AWQ/GPTQ-style, scales not counted)"),
  D("q8_0", 8.5, true, true, "block quant: 32 weights in 34 bytes"),
  D("q6_K", 6.5625, true, false, "block quant: 256 weights in 210 bytes"),
  D("q5_K_M", 5.69, true, false, "mixed q5_K/q6_K preset, measured average", true),
  D("q5_K", 5.5, true, false, "block quant: 256 weights in 176 bytes"),
  D("q5_1", 6, true, true, "block quant: 32 weights in 24 bytes"),
  D("q5_0", 5.5, true, true, "block quant: 32 weights in 22 bytes"),
  D("q4_K_M", 4.85, true, false, "mixed q4_K/q6_K preset, measured average", true),
  D("q4_K", 4.5, true, false, "block quant: 256 weights in 144 bytes"),
  D("q4_1", 5, true, true, "block quant: 32 weights in 20 bytes"),
  D("q4_0", 4.5, true, true, "block quant: 32 weights in 18 bytes"),
  D("q3_K_M", 3.91, true, false, "mixed q3_K/q5_K preset, measured average", true),
  D("q3_K", 3.4375, true, false, "block quant: 256 weights in 110 bytes"),
  D("q2_K", 2.5625, true, false, "block quant: 256 weights in 82 bytes"),
];

/** Aliases accepted on the command line, normalized → canonical name. */
const ALIASES: Record<string, string> = {
  f32: "fp32",
  float32: "fp32",
  f16: "fp16",
  half: "fp16",
  float16: "fp16",
  bfloat16: "bf16",
  f8: "fp8",
  fp8e4m3: "fp8",
  fp8e5m2: "fp8",
  i8: "int8",
  i4: "int4",
};

/** Lowercase and strip separators so "Q4_K_M", "q4-k-m" and "q4km" all match. */
function normalizeKey(s: string): string {
  return s.toLowerCase().replace(/[\s_.-]/g, "");
}

const BY_KEY = new Map<string, DtypeInfo>();
for (const d of DTYPES) BY_KEY.set(normalizeKey(d.name), d);
for (const [alias, canonical] of Object.entries(ALIASES)) {
  const info = BY_KEY.get(normalizeKey(canonical));
  if (info) BY_KEY.set(normalizeKey(alias), info);
}

export class DtypeError extends Error {}

/** Look up a dtype by (fuzzy) name; `role` narrows to weight- or kv-capable dtypes. */
export function resolveDtype(name: string, role: "weights" | "kv"): DtypeInfo {
  const info = BY_KEY.get(normalizeKey(name));
  if (!info) {
    throw new DtypeError(
      `unknown dtype "${name}" — run \`kvcalc dtypes\` for the full list`,
    );
  }
  if (role === "weights" && !info.weights) {
    throw new DtypeError(`dtype "${info.name}" is not usable for weights`);
  }
  if (role === "kv" && !info.kv) {
    throw new DtypeError(
      `dtype "${info.name}" is not usable for the KV cache (256-wide super-blocks don't fit per-token cache lines) — try fp16, q8_0 or q4_0`,
    );
  }
  return info;
}

/** All dtypes usable in the given role, in display order. */
export function dtypesFor(role: "weights" | "kv"): DtypeInfo[] {
  return DTYPES.filter((d) => (role === "weights" ? d.weights : d.kv));
}
