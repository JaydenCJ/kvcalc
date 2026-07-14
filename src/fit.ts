/**
 * Budget solving: the largest context that fits a VRAM budget.
 *
 * KV size is monotonically non-decreasing and piecewise linear in ctx
 * (sliding-window layers flatten out past their window), so a binary search
 * over token counts finds the exact integer boundary.
 */

import { estimateKv } from "./kv.js";
import { estimateWeights } from "./weights.js";
import type { FitResult, NormalizedConfig } from "./types.js";

export interface FitOptions {
  budgetBytes: number;
  weightsDtype: string;
  kvDtype: string;
  batch: number;
  overheadBytes: number;
}

/** Hard search ceiling: 1 Gi tokens. Far beyond any real deployment; keeps the search bounded. */
export const CTX_SEARCH_MAX = 1 << 30;

/** Solve for the largest context length whose weights + KV cache fit the budget. */
export function fitContext(cfg: NormalizedConfig, opts: FitOptions): FitResult {
  const weights = estimateWeights(cfg, opts.weightsDtype);
  const kvBudget = opts.budgetBytes - opts.overheadBytes - weights.bytes;

  const kvAt = (ctx: number): number => estimateKv(cfg, opts.kvDtype, ctx, opts.batch).bytes;

  let maxCtx = 0;
  if (kvBudget >= 0) {
    // Invariant: kvAt(lo) <= kvBudget < kvAt(hi)  (hi clamped to the search ceiling).
    let lo = 0;
    let hi = CTX_SEARCH_MAX;
    if (kvAt(hi) <= kvBudget) {
      maxCtx = hi;
    } else {
      while (lo < hi - 1) {
        const mid = Math.floor((lo + hi) / 2);
        if (kvAt(mid) <= kvBudget) lo = mid;
        else hi = mid;
      }
      maxCtx = lo;
    }
  }

  const modelMax = cfg.maxPositionEmbeddings;
  return {
    budgetBytes: opts.budgetBytes,
    overheadBytes: opts.overheadBytes,
    weightBytes: weights.bytes,
    kvBudgetBytes: kvBudget,
    batch: opts.batch,
    maxCtx,
    kvBytesAtMax: kvAt(maxCtx),
    fullContextFits: modelMax === null ? null : maxCtx >= modelMax,
  };
}
