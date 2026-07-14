/**
 * Shared test helpers: a raw-config factory with sane tiny defaults, and a
 * CLI runner that spawns the real compiled binary. Everything is offline and
 * deterministic — configs are literal objects, the CLI reads only local files.
 */

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const CLI = path.join(ROOT, "dist", "cli.js");
export const EXAMPLES = path.join(ROOT, "examples");

/**
 * A tiny, hand-checkable GQA config. Totals (worked out on paper, see
 * weights.test.mjs): 86848 params, 64 KV elements per layer per token.
 */
export function tinyConfig(overrides = {}) {
  return {
    model_type: "test",
    hidden_size: 64,
    num_hidden_layers: 2,
    num_attention_heads: 4,
    num_key_value_heads: 2,
    head_dim: 16,
    intermediate_size: 128,
    vocab_size: 100,
    max_position_embeddings: 4096,
    tie_word_embeddings: false,
    ...overrides,
  };
}

/** Run the compiled CLI with the given argv; returns { status, stdout, stderr }. */
export function runCli(args, opts = {}) {
  const res = spawnSync(process.execPath, [CLI, ...args], {
    cwd: ROOT,
    encoding: "utf8",
    ...opts,
  });
  if (res.error) throw res.error;
  return { status: res.status, stdout: res.stdout, stderr: res.stderr };
}
