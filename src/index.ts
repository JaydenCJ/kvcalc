/**
 * kvcalc public API.
 *
 * The library surface is pure: load/normalize a config, then compute weights,
 * KV-cache and fit numbers as plain data. Only the CLI touches the process.
 */

export { loadConfig, normalizeConfig, ConfigError } from "./config.js";
export { countParams, estimateWeights } from "./weights.js";
export { estimateKv, kvBytesPerToken, kvElementsPerLayer } from "./kv.js";
export { fitContext, CTX_SEARCH_MAX, type FitOptions } from "./fit.js";
export { DTYPES, dtypesFor, resolveDtype, DtypeError, type DtypeInfo } from "./dtype.js";
export {
  formatBytes,
  formatCount,
  formatCtx,
  parseCtx,
  parseCtxList,
  parseSize,
  UnitError,
  GIB,
  MIB,
  KIB,
  TIB,
} from "./units.js";
export {
  computeReport,
  computeTable,
  computeFit,
  computeDtypes,
  describeArch,
} from "./report.js";
export { VERSION } from "./version.js";
export type {
  AttentionKind,
  FitResult,
  KvEstimate,
  LayerAttentionType,
  MlaGeometry,
  MoeGeometry,
  NormalizedConfig,
  ParamBreakdown,
  SlidingLayout,
  WeightEstimate,
} from "./types.js";
