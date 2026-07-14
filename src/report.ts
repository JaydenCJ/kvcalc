/**
 * Computation + rendering for the CLI commands. Each command has a `compute*`
 * step that produces a plain JSON-able object (also what `--json` prints) and
 * a `render*` step that turns it into aligned terminal text. Keeping the two
 * apart makes every number testable without scraping formatted output.
 */

import { dtypesFor, resolveDtype } from "./dtype.js";
import { estimateKv, kvBytesPerToken } from "./kv.js";
import { estimateWeights } from "./weights.js";
import { fitContext } from "./fit.js";
import { formatBytes, formatCount, formatCtx } from "./units.js";
import { VERSION } from "./version.js";
import type { NormalizedConfig } from "./types.js";

export interface CommonOptions {
  path: string;
  weightsDtype: string;
  kvDtype: string;
  batch: number;
  overheadBytes: number;
}

/* ------------------------------------------------------------------ model */

function modelSummary(cfg: NormalizedConfig, path: string) {
  return {
    path,
    modelType: cfg.modelType,
    layers: cfg.numLayers,
    hiddenSize: cfg.hiddenSize,
    heads: cfg.numHeads,
    kvHeads: cfg.numKvHeads,
    headDim: cfg.headDim,
    vocabSize: cfg.vocabSize,
    maxPositionEmbeddings: cfg.maxPositionEmbeddings,
    attention: cfg.attentionKind,
    mla: cfg.mla,
    moe: cfg.moe
      ? {
          numExperts: cfg.moe.numExperts,
          expertsPerTok: cfg.moe.expertsPerTok,
          numSharedExperts: cfg.moe.numSharedExperts,
        }
      : null,
    slidingWindow: cfg.sliding.window,
    slidingLayers: cfg.sliding.layerTypes.filter((t) => t === "sliding").length,
  };
}

/** One-line architecture description: "32 layers · GQA 32q/8kv · head_dim 128 · …". */
export function describeArch(cfg: NormalizedConfig): string {
  const parts: string[] = [`${cfg.numLayers} layers`];
  if (cfg.attentionKind === "mla") {
    parts.push(`MLA latent ${cfg.mla!.kvLoraRank} + rope ${cfg.mla!.qkRopeHeadDim}`);
  } else if (cfg.attentionKind === "gqa") {
    parts.push(`GQA ${cfg.numHeads}q/${cfg.numKvHeads}kv · head_dim ${cfg.headDim}`);
  } else {
    parts.push(`MHA ${cfg.numHeads} heads · head_dim ${cfg.headDim}`);
  }
  if (cfg.moe) {
    const shared = cfg.moe.numSharedExperts > 0 ? `+${cfg.moe.numSharedExperts} shared` : "";
    parts.push(`MoE ${cfg.moe.numExperts}${shared} experts, top-${cfg.moe.expertsPerTok}`);
  }
  const slidingCount = cfg.sliding.layerTypes.filter((t) => t === "sliding").length;
  if (slidingCount > 0) {
    parts.push(`sliding ${slidingCount}/${cfg.numLayers} layers @ ${cfg.sliding.window}`);
  }
  if (cfg.maxPositionEmbeddings !== null) parts.push(`max ctx ${cfg.maxPositionEmbeddings}`);
  return parts.join(" · ");
}

/* ----------------------------------------------------------------- report */

export interface ReportOptions extends CommonOptions {
  ctx: number;
  budgetBytes: number | null;
}

export function computeReport(cfg: NormalizedConfig, opts: ReportOptions) {
  const weights = estimateWeights(cfg, opts.weightsDtype);
  const kv = estimateKv(cfg, opts.kvDtype, opts.ctx, opts.batch);
  const totalBytes = weights.bytes + kv.bytes + opts.overheadBytes;
  const budget =
    opts.budgetBytes === null
      ? null
      : {
          bytes: opts.budgetBytes,
          fits: totalBytes <= opts.budgetBytes,
          headroomBytes: opts.budgetBytes - totalBytes,
        };
  return {
    tool: "kvcalc",
    version: VERSION,
    command: "report" as const,
    model: modelSummary(cfg, opts.path),
    params: weights.params,
    weights: {
      dtype: weights.dtype,
      bitsPerWeight: weights.bitsPerWeight,
      approx: weights.approx,
      bytes: weights.bytes,
    },
    kv,
    overheadBytes: opts.overheadBytes,
    totalBytes,
    budget,
  };
}

export function renderReport(r: ReturnType<typeof computeReport>, cfg: NormalizedConfig): string {
  const lines: string[] = [];
  lines.push(`kvcalc ${VERSION} — memory report`);
  lines.push("");
  lines.push(`model     ${r.model.path}`);
  lines.push(`arch      ${describeArch(cfg)}`);
  const active =
    r.params.active !== r.params.total ? `  (${formatCount(r.params.active)} active)` : "";
  lines.push(`params    ${formatCount(r.params.total)}${active}`);
  const bd = r.params;
  const routerPart = bd.router > 0 ? ` · router ${formatCount(bd.router)}` : "";
  const headPart = bd.lmHead > 0 ? ` · head ${formatCount(bd.lmHead)}` : " · head tied";
  lines.push(
    `          embed ${formatCount(bd.embeddings)} · attn ${formatCount(bd.attention)} · mlp ${formatCount(bd.mlp)}${routerPart}${headPart}`,
  );
  lines.push("");
  const approxMark = r.weights.approx ? " (~)" : "";
  lines.push(
    `weights   ${r.weights.dtype.padEnd(7)} ${formatBytes(r.weights.bytes).padStart(11)}   ${formatCount(bd.total)} × ${r.weights.bitsPerWeight} bpw${approxMark}`,
  );
  const capped = r.kv.windowCapped ? `, ${r.kv.slidingLayers} layers capped @ ${r.kv.window}` : "";
  lines.push(
    `kv cache  ${r.kv.dtype.padEnd(7)} ${formatBytes(r.kv.bytes).padStart(11)}   ctx ${r.kv.ctx} × batch ${r.kv.batch} × ${formatBytes(r.kv.bytesPerToken)}/token${capped}`,
  );
  if (r.overheadBytes > 0) {
    lines.push(`overhead          ${formatBytes(r.overheadBytes).padStart(11)}`);
  }
  lines.push(`total             ${formatBytes(r.totalBytes).padStart(11)}`);
  if (r.budget) {
    lines.push("");
    const verdict = r.budget.fits
      ? `FITS   (headroom ${formatBytes(r.budget.headroomBytes)})`
      : `DOES NOT FIT   (over by ${formatBytes(-r.budget.headroomBytes)})`;
    lines.push(`budget    ${formatBytes(r.budget.bytes)} → ${verdict}`);
  }
  return lines.join("\n");
}

/* ------------------------------------------------------------------ table */

export interface TableOptions extends CommonOptions {
  ctxList: number[];
  kvDtypes: string[];
  budgetBytes: number | null;
}

export function computeTable(cfg: NormalizedConfig, opts: TableOptions) {
  const weights = estimateWeights(cfg, opts.weightsDtype);
  const kvDtypes = opts.kvDtypes.map((d) => resolveDtype(d, "kv").name);
  const rows = opts.ctxList.map((ctx) => ({
    ctx,
    cells: kvDtypes.map((dtype) => {
      const kv = estimateKv(cfg, dtype, ctx, opts.batch);
      const totalBytes = weights.bytes + kv.bytes + opts.overheadBytes;
      return {
        kvDtype: dtype,
        kvBytes: kv.bytes,
        totalBytes,
        fits: opts.budgetBytes === null ? null : totalBytes <= opts.budgetBytes,
      };
    }),
  }));
  return {
    tool: "kvcalc",
    version: VERSION,
    command: "table" as const,
    model: modelSummary(cfg, opts.path),
    weights: { dtype: weights.dtype, bytes: weights.bytes },
    batch: opts.batch,
    overheadBytes: opts.overheadBytes,
    budgetBytes: opts.budgetBytes,
    kvDtypes,
    rows,
  };
}

export function renderTable(t: ReturnType<typeof computeTable>): string {
  const lines: string[] = [];
  lines.push(`kvcalc ${VERSION} — memory table`);
  lines.push("");
  const budget = t.budgetBytes === null ? "" : ` · budget ${formatBytes(t.budgetBytes)}`;
  lines.push(
    `model     ${t.model.path} · weights ${t.weights.dtype} = ${formatBytes(t.weights.bytes)} · batch ${t.batch}${budget}`,
  );
  lines.push("");
  const CELL = 14;
  const header = ["ctx".padStart(8), ...t.kvDtypes.map((d) => `kv ${d}`.padStart(CELL))].join("  ");
  lines.push(header);
  for (const row of t.rows) {
    const cells = row.cells.map((c) => {
      const mark = c.fits === null ? "" : c.fits ? " ✓" : " ✗";
      return `${formatBytes(c.totalBytes)}${mark}`.padStart(CELL);
    });
    lines.push([formatCtx(row.ctx).padStart(8), ...cells].join("  "));
  }
  lines.push("");
  lines.push(
    t.budgetBytes === null
      ? "cells are weights + kv (+ overhead)"
      : "cells are weights + kv (+ overhead); ✓/✗ compare against --vram",
  );
  return lines.join("\n");
}

/* -------------------------------------------------------------------- fit */

export interface FitCmdOptions extends CommonOptions {
  budgetBytes: number;
}

export function computeFit(cfg: NormalizedConfig, opts: FitCmdOptions) {
  const result = fitContext(cfg, {
    budgetBytes: opts.budgetBytes,
    weightsDtype: opts.weightsDtype,
    kvDtype: opts.kvDtype,
    batch: opts.batch,
    overheadBytes: opts.overheadBytes,
  });
  return {
    tool: "kvcalc",
    version: VERSION,
    command: "fit" as const,
    model: modelSummary(cfg, opts.path),
    weightsDtype: resolveDtype(opts.weightsDtype, "weights").name,
    kvDtype: resolveDtype(opts.kvDtype, "kv").name,
    kvBytesPerToken: kvBytesPerToken(cfg, opts.kvDtype),
    ...result,
  };
}

export function renderFit(f: ReturnType<typeof computeFit>): string {
  const lines: string[] = [];
  lines.push(`kvcalc ${VERSION} — fit ${formatBytes(f.budgetBytes)}`);
  lines.push("");
  lines.push(`model     ${f.model.path}`);
  lines.push(`weights   ${f.weightsDtype.padEnd(7)} ${formatBytes(f.weightBytes).padStart(11)}`);
  if (f.overheadBytes > 0) {
    lines.push(`overhead          ${formatBytes(f.overheadBytes).padStart(11)}`);
  }
  lines.push(
    `kv        ${f.kvDtype.padEnd(7)} ${formatBytes(f.kvBytesPerToken).padStart(11)}/token × batch ${f.batch}`,
  );
  if (f.kvBudgetBytes < 0) {
    lines.push("");
    lines.push(
      `verdict   weights alone exceed the budget by ${formatBytes(-f.kvBudgetBytes)} — nothing fits`,
    );
    return lines.join("\n");
  }
  lines.push(`kv budget         ${formatBytes(f.kvBudgetBytes).padStart(11)}`);
  lines.push("");
  const pretty = formatCtx(f.maxCtx) !== String(f.maxCtx) ? ` (${formatCtx(f.maxCtx)})` : "";
  lines.push(`max ctx   ${f.maxCtx} tokens${pretty} at batch ${f.batch}, kv ${formatBytes(f.kvBytesAtMax)}`);
  const modelMax = f.model.maxPositionEmbeddings;
  if (modelMax !== null) {
    const verdict = f.fullContextFits
      ? `full model context FITS`
      : `full model context DOES NOT FIT`;
    lines.push(`model max ${modelMax} (${formatCtx(modelMax)}) → ${verdict}`);
  }
  return lines.join("\n");
}

/* ----------------------------------------------------------------- dtypes */

export function computeDtypes() {
  return {
    tool: "kvcalc",
    version: VERSION,
    command: "dtypes" as const,
    dtypes: dtypesFor("weights").concat(dtypesFor("kv").filter((d) => !d.weights)),
  };
}

export function renderDtypes(d: ReturnType<typeof computeDtypes>): string {
  const lines: string[] = [];
  lines.push(`kvcalc ${VERSION} — dtype reference`);
  lines.push("");
  lines.push(`${"name".padEnd(8)}${"bpw".padStart(8)}  ${"weights".padEnd(8)}${"kv".padEnd(4)}note`);
  for (const info of d.dtypes) {
    const bpw = info.approx ? `~${info.bitsPerWeight}` : String(info.bitsPerWeight);
    lines.push(
      `${info.name.padEnd(8)}${bpw.padStart(8)}  ${(info.weights ? "yes" : "-").padEnd(8)}${(info.kv ? "yes" : "-").padEnd(4)}${info.note}`,
    );
  }
  return lines.join("\n");
}
