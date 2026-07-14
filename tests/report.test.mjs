/**
 * report.test.mjs — the compute layer behind each CLI command.
 *
 * These test the JSON-able result objects (what --json prints) so numbers and
 * verdicts are asserted as data, not scraped out of formatted text.
 */

import test from "node:test";
import assert from "node:assert/strict";
import {
  normalizeConfig,
  computeReport,
  computeTable,
  computeFit,
  computeDtypes,
  describeArch,
  GIB,
} from "../dist/index.js";
import { tinyConfig } from "./helpers.mjs";

const BASE = {
  path: "test.json",
  weightsDtype: "fp16",
  kvDtype: "fp16",
  batch: 1,
  overheadBytes: 0,
};

test("computeReport: total = weights + kv + overhead, exactly", () => {
  const cfg = normalizeConfig(tinyConfig());
  const r = computeReport(cfg, { ...BASE, ctx: 1000, budgetBytes: null, overheadBytes: 512 });
  assert.equal(r.weights.bytes, 86848 * 2);
  assert.equal(r.kv.bytes, 1000 * 256);
  assert.equal(r.totalBytes, 86848 * 2 + 256000 + 512);
  assert.equal(r.budget, null);
});

test("computeReport: budget verdict and signed headroom", () => {
  const cfg = normalizeConfig(tinyConfig());
  const fits = computeReport(cfg, { ...BASE, ctx: 1000, budgetBytes: GIB });
  assert.equal(fits.budget.fits, true);
  assert.equal(fits.budget.headroomBytes, GIB - fits.totalBytes);
  const tight = computeReport(cfg, { ...BASE, ctx: 1000, budgetBytes: fits.totalBytes - 1 });
  assert.equal(tight.budget.fits, false);
  assert.equal(tight.budget.headroomBytes, -1);
});

test("computeTable: one row per ctx, one cell per kv dtype, fits flags against budget", () => {
  const cfg = normalizeConfig(tinyConfig());
  const t = computeTable(cfg, {
    ...BASE,
    ctxList: [1000, 2000],
    kvDtypes: ["fp16", "q4_0"],
    budgetBytes: 86848 * 2 + 1500 * 256,
  });
  assert.equal(t.rows.length, 2);
  assert.deepEqual(t.kvDtypes, ["fp16", "q4_0"]);
  const [row1k, row2k] = t.rows;
  assert.equal(row1k.cells[0].fits, true); // 1000 tokens fp16 fits
  assert.equal(row2k.cells[0].fits, false); // 2000 tokens fp16 does not
  assert.equal(row2k.cells[1].fits, true); // …but q4_0 kv does (2000·72 bytes)
  assert.equal(row2k.cells[1].kvBytes, (2000 * (256 * 4.5)) / 16);
});

test("computeTable: without a budget every fits flag is null", () => {
  const cfg = normalizeConfig(tinyConfig());
  const t = computeTable(cfg, { ...BASE, ctxList: [1000], kvDtypes: ["fp16"], budgetBytes: null });
  assert.equal(t.rows[0].cells[0].fits, null);
});

test("computeFit: carries the solved context and canonical dtype names", () => {
  const cfg = normalizeConfig(tinyConfig());
  const f = computeFit(cfg, { ...BASE, weightsDtype: "half", budgetBytes: 86848 * 2 + 500 * 256 });
  assert.equal(f.weightsDtype, "fp16");
  assert.equal(f.maxCtx, 500);
  assert.equal(f.kvBytesPerToken, 256);
  assert.equal(f.fullContextFits, false); // model max 4096 > 500
});

test("describeArch: GQA, MLA, MoE and sliding facts all surface", () => {
  const gqa = describeArch(normalizeConfig(tinyConfig()));
  assert.match(gqa, /GQA 4q\/2kv/);
  assert.match(gqa, /2 layers/);

  const mla = describeArch(
    normalizeConfig(
      tinyConfig({
        kv_lora_rank: 32,
        q_lora_rank: 24,
        qk_rope_head_dim: 8,
        qk_nope_head_dim: 16,
        v_head_dim: 16,
      }),
    ),
  );
  assert.match(mla, /MLA latent 32 \+ rope 8/);

  const moe = describeArch(
    normalizeConfig(tinyConfig({ num_experts: 8, num_experts_per_tok: 2, n_shared_experts: 1 })),
  );
  assert.match(moe, /MoE 8\+1 shared experts, top-2/);

  const swa = describeArch(normalizeConfig(tinyConfig({ sliding_window: 512 })));
  assert.match(swa, /sliding 2\/2 layers @ 512/);
});

test("every compute result is JSON-round-trippable plain data; dtypes listed exactly once", () => {
  const cfg = normalizeConfig(tinyConfig());
  const r = computeReport(cfg, { ...BASE, ctx: 100, budgetBytes: GIB });
  const t = computeTable(cfg, { ...BASE, ctxList: [100], kvDtypes: ["fp16"], budgetBytes: null });
  const f = computeFit(cfg, { ...BASE, budgetBytes: GIB });
  const d = computeDtypes();
  for (const obj of [r, t, f, d]) {
    assert.deepEqual(JSON.parse(JSON.stringify(obj)), obj);
    assert.equal(obj.tool, "kvcalc");
  }
  const names = d.dtypes.map((x) => x.name);
  assert.equal(new Set(names).size, names.length);
  assert.ok(names.includes("q4_K_M") && names.includes("fp16") && names.includes("q8_0"));
});
