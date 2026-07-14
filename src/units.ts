/**
 * Parsing and formatting of sizes, context lengths and parameter counts.
 *
 * Convention: byte suffixes are binary. "24GB" on a GPU spec sheet means
 * 24 GiB of VRAM, so kvcalc treats G/GB/GiB identically (2^30 bytes) rather
 * than surprising users with a 7% decimal/binary gap. Context suffixes are
 * binary too: "128k" is 131072 tokens, matching how context windows are named.
 */

export const KIB = 1024;
export const MIB = 1024 * KIB;
export const GIB = 1024 * MIB;
export const TIB = 1024 * GIB;

export class UnitError extends Error {}

const SIZE_UNITS: Record<string, number> = {
  "": 1,
  b: 1,
  k: KIB,
  kb: KIB,
  kib: KIB,
  m: MIB,
  mb: MIB,
  mib: MIB,
  g: GIB,
  gb: GIB,
  gib: GIB,
  t: TIB,
  tb: TIB,
  tib: TIB,
};

/** Parse a human byte size: "24GiB", "24GB", "512MiB", "1.5g", "1073741824". */
export function parseSize(input: string): number {
  const m = /^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$/.exec(input);
  if (!m) throw new UnitError(`cannot parse size "${input}" (try e.g. 24GiB or 512MiB)`);
  const value = Number(m[1]);
  const unit = SIZE_UNITS[(m[2] ?? "").toLowerCase()];
  if (unit === undefined) {
    throw new UnitError(`unknown size unit "${m[2]}" in "${input}" (use B/KiB/MiB/GiB/TiB)`);
  }
  const bytes = value * unit;
  if (!Number.isFinite(bytes) || bytes < 0) throw new UnitError(`size out of range: "${input}"`);
  return bytes;
}

/** Parse a context length: "128k" → 131072, "1m" → 1048576, "4096" → 4096. */
export function parseCtx(input: string): number {
  const m = /^\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*$/.exec(input);
  if (!m) throw new UnitError(`cannot parse context length "${input}" (try e.g. 8192 or 128k)`);
  const mult = m[2] === "" ? 1 : m[2]!.toLowerCase() === "k" ? 1024 : 1024 * 1024;
  const ctx = Number(m[1]) * mult;
  if (!Number.isInteger(ctx) || ctx <= 0) {
    throw new UnitError(`context length must be a positive whole number of tokens: "${input}"`);
  }
  return ctx;
}

/** Parse a comma-separated list of context lengths, e.g. "4k,8k,32k". */
export function parseCtxList(input: string): number[] {
  const parts = input.split(",").map((s) => s.trim()).filter((s) => s.length > 0);
  if (parts.length === 0) throw new UnitError("empty context list");
  return parts.map(parseCtx);
}

/** Format bytes with a binary unit, two decimals from MiB up: "20.53 GiB". */
export function formatBytes(bytes: number): string {
  if (bytes >= TIB) return `${(bytes / TIB).toFixed(2)} TiB`;
  if (bytes >= GIB) return `${(bytes / GIB).toFixed(2)} GiB`;
  if (bytes >= MIB) return `${(bytes / MIB).toFixed(2)} MiB`;
  if (bytes >= KIB) return `${(bytes / KIB).toFixed(2)} KiB`;
  return `${Math.round(bytes)} B`;
}

/** Format a parameter count: 8030261248 → "8.03 B", 524288 → "524.29 K". */
export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(2)} K`;
  return String(n);
}

/** Format a context length the way people say it: 131072 → "128k", 4096 → "4k", 5000 → "5000". */
export function formatCtx(ctx: number): string {
  if (ctx >= 1024 * 1024 && ctx % (1024 * 1024) === 0) return `${ctx / (1024 * 1024)}m`;
  if (ctx >= 1024 && ctx % 1024 === 0) return `${ctx / 1024}k`;
  return String(ctx);
}
