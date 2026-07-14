#!/usr/bin/env node
/**
 * kvcalc CLI — the only module that touches the filesystem or the process.
 *
 * Exit codes (script-friendly, documented in --help):
 *   0  success; when --vram is given, everything checked fits
 *   1  a --vram budget check failed (report/table: some/all shown points
 *      don't fit; fit: not even ctx 0 fits)
 *   2  usage, config or dtype error
 */

import { loadConfig, ConfigError } from "./config.js";
import { DtypeError } from "./dtype.js";
import { parseArgs, positiveInt, UsageError, type FlagSpec } from "./args.js";
import {
  computeDtypes,
  computeFit,
  computeReport,
  computeTable,
  renderDtypes,
  renderFit,
  renderReport,
  renderTable,
} from "./report.js";
import { parseCtx, parseCtxList, parseSize, UnitError } from "./units.js";
import { VERSION } from "./version.js";
import type { NormalizedConfig } from "./types.js";

const HELP = `kvcalc ${VERSION} — KV-cache and weight memory from config.json

Usage:
  kvcalc report <config.json> [options]   memory at one (ctx, batch) point
  kvcalc table  <config.json> [options]   totals across ctx × kv-dtype grid
  kvcalc fit    <config.json> --vram <size> [options]
                                          largest ctx that fits a budget
  kvcalc dtypes                           dtype reference (bits per weight)

Options:
  --ctx <n|Nk>          context length (report: default = model max, else 8k)
  --ctx-list <a,b,...>  table rows (default 2k,4k,8k,16k,32k,64k,128k,256k,
                        capped at the model max)
  --batch <n>           concurrent sequences (default 1)
  --weights <dtype>     weight storage dtype (default bf16)
  --kv <dtype>          KV-cache dtype (default fp16)
  --kv-list <a,b,...>   table columns (default fp16,q8_0,q4_0)
  --vram <size>         budget, e.g. 24GiB (GB is read as GiB — that is what
                        GPU spec sheets mean)
  --overhead <size>     extra reserved bytes, e.g. 1.5GiB (default 0)
  --json                machine-readable output
  --help, --version

Exit codes: 0 ok / fits · 1 a --vram check failed · 2 usage or config error`;

const COMMON_FLAGS: FlagSpec[] = [
  { name: "ctx", takesValue: true },
  { name: "ctx-list", takesValue: true },
  { name: "batch", takesValue: true },
  { name: "weights", takesValue: true },
  { name: "kv", takesValue: true },
  { name: "kv-list", takesValue: true },
  { name: "vram", takesValue: true },
  { name: "overhead", takesValue: true },
  { name: "json", takesValue: false },
  { name: "help", takesValue: false },
  { name: "version", takesValue: false },
];

interface Io {
  out: (s: string) => void;
  err: (s: string) => void;
}

function loadModel(positionals: string[], io: Io): NormalizedConfig {
  const path = positionals[1];
  if (path === undefined) throw new UsageError("missing <config.json> operand");
  if (positionals.length > 2) throw new UsageError(`unexpected operand "${positionals[2]}"`);
  const cfg = loadConfig(path);
  for (const w of cfg.warnings) io.err(`note: ${w}`);
  return cfg;
}

const DEFAULT_TABLE_CTX = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];

function defaultCtxList(cfg: NormalizedConfig): number[] {
  const max = cfg.maxPositionEmbeddings;
  if (max === null) return DEFAULT_TABLE_CTX;
  const list = DEFAULT_TABLE_CTX.filter((c) => c <= max);
  if (list.length === 0) return [max];
  if (list[list.length - 1]! < max) list.push(max);
  return list;
}

/** Run the CLI. Pure with respect to `io`; returns the process exit code. */
export function run(argv: string[], io: Io): number {
  const parsed = parseArgs(argv, COMMON_FLAGS);
  const { values, flags, positionals } = parsed;

  if (flags.has("version")) {
    io.out(VERSION);
    return 0;
  }
  if (flags.has("help") || positionals.length === 0) {
    io.out(HELP);
    return flags.has("help") ? 0 : 2;
  }

  const command = positionals[0]!;
  const json = flags.has("json");
  const batch = values.has("batch") ? positiveInt("batch", values.get("batch")!) : 1;
  const weightsDtype = values.get("weights") ?? "bf16";
  const kvDtype = values.get("kv") ?? "fp16";
  const overheadBytes = values.has("overhead") ? parseSize(values.get("overhead")!) : 0;
  const budgetBytes = values.has("vram") ? parseSize(values.get("vram")!) : null;

  const reject = (flag: string, cmds: string) => {
    if (values.has(flag) || flags.has(flag)) {
      throw new UsageError(`--${flag} only applies to ${cmds}`);
    }
  };

  switch (command) {
    case "report": {
      reject("ctx-list", "table");
      reject("kv-list", "table");
      const cfg = loadModel(positionals, io);
      const ctx = values.has("ctx")
        ? parseCtx(values.get("ctx")!)
        : (cfg.maxPositionEmbeddings ?? 8192);
      const r = computeReport(cfg, {
        path: positionals[1]!,
        ctx,
        batch,
        weightsDtype,
        kvDtype,
        overheadBytes,
        budgetBytes,
      });
      io.out(json ? JSON.stringify(r, null, 2) : renderReport(r, cfg));
      return r.budget !== null && !r.budget.fits ? 1 : 0;
    }
    case "table": {
      reject("ctx", "report and fit");
      reject("kv", "report and fit");
      const cfg = loadModel(positionals, io);
      const ctxList = values.has("ctx-list")
        ? parseCtxList(values.get("ctx-list")!)
        : defaultCtxList(cfg);
      const kvDtypes = values.has("kv-list")
        ? values.get("kv-list")!.split(",").map((s) => s.trim()).filter((s) => s.length > 0)
        : ["fp16", "q8_0", "q4_0"];
      if (kvDtypes.length === 0) throw new UsageError("--kv-list is empty");
      const t = computeTable(cfg, {
        path: positionals[1]!,
        ctxList,
        kvDtypes,
        batch,
        weightsDtype,
        kvDtype,
        overheadBytes,
        budgetBytes,
      });
      io.out(json ? JSON.stringify(t, null, 2) : renderTable(t));
      const anyChecked = t.rows.some((r) => r.cells.some((c) => c.fits !== null));
      const allFit = t.rows.every((r) => r.cells.every((c) => c.fits !== false));
      return anyChecked && !allFit ? 1 : 0;
    }
    case "fit": {
      reject("ctx", "report");
      reject("ctx-list", "table");
      reject("kv-list", "table");
      if (budgetBytes === null) throw new UsageError("fit requires --vram <size>");
      const cfg = loadModel(positionals, io);
      const f = computeFit(cfg, {
        path: positionals[1]!,
        budgetBytes,
        batch,
        weightsDtype,
        kvDtype,
        overheadBytes,
      });
      io.out(json ? JSON.stringify(f, null, 2) : renderFit(f));
      return f.maxCtx > 0 ? 0 : 1;
    }
    case "dtypes": {
      if (positionals.length > 1) throw new UsageError(`unexpected operand "${positionals[1]}"`);
      for (const spec of COMMON_FLAGS) {
        if (spec.takesValue && values.has(spec.name)) {
          throw new UsageError(`--${spec.name} does not apply to dtypes`);
        }
      }
      const d = computeDtypes();
      io.out(json ? JSON.stringify(d, null, 2) : renderDtypes(d));
      return 0;
    }
    default:
      throw new UsageError(`unknown command "${command}" — see kvcalc --help`);
  }
}

/** Entry point used by the bin shim. */
export function main(argv: string[]): number {
  const io: Io = {
    out: (s) => process.stdout.write(s + "\n"),
    err: (s) => process.stderr.write(s + "\n"),
  };
  try {
    return run(argv, io);
  } catch (err) {
    if (
      err instanceof UsageError ||
      err instanceof ConfigError ||
      err instanceof DtypeError ||
      err instanceof UnitError ||
      err instanceof RangeError
    ) {
      io.err(`kvcalc: ${err.message}`);
      return 2;
    }
    throw err;
  }
}

process.exitCode = main(process.argv.slice(2));
