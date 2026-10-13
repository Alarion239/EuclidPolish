/* Fixtures of the Universal tab's payload (GET /ensemble/universal.json),
 * imported by tests only: universal.test.ts, notes.test.ts, tabs.test.tsx.
 * Six models — 211 (depth 16, 1M steps, the best), 210, 208 (depth 32),
 * 215 (a MIG slice, a stale score made at 1.2 h / 120k steps), 196 (an
 * ensemble member, unscored) and 212 (no checkpoint yet) — two scored gates
 * on the scores' own fields (`comparable`; `comb` alone is a legacy row
 * without the flag) and two FASRC rows (one pullable, one training). */
import type { UniversalCombiner, UniversalPayload, UniversalRow } from "../api";

export const score = (mean: number, patch: Partial<NonNullable<UniversalRow["score"]>> = {}) => ({
  integrated: [mean - 6, mean + 3, mean + 1, mean + 2], mean, n_fields: 100, stale: false, scored_at: "2026-10-13T10:00:00Z", ...patch,
});

export const urow = (n: number, patch: Partial<UniversalRow> = {}): UniversalRow => ({
  name: `member_${n}`, label: `${n}·psnr`, spec: `universal:member_${n}`, class: "universal",
  num_res_blocks: 16, steps: 200_000, served_step: 198_000, target_steps: 200_000, batch_size: 4, crops_per_field: 8,
  knee: "learned", output_knee: null, learned_knees: [8.1, 12, 15, 20], jit_compile: false, mixed_precision: false,
  bootstrap: null, seed: 2898473870, train_hours: 2.5, train_rows: 200,
  hardware: { partition: "gpu", label: "A100", jobs: ["51865478"] },
  fingerprint: `fp${n}`, score: score(61), vs_production_db: null, created_at: "2026-10-01T00:00:00Z", commit: "abc1234",
  ...patch,
});

/** A row as the server sends it: Δ vs production (60.5 dB) on the band mean
 *  when it has a score (the server gives it only when comparable). */
const scored = (n: number, patch: Partial<UniversalRow> = {}): UniversalRow => {
  const r = urow(n, patch);
  return { ...r, vs_production_db: r.score?.mean != null ? +(r.score.mean - 60.5).toFixed(4) : null };
};

export const comb = (id: string, patch: Partial<UniversalCombiner> = {}): UniversalCombiner => ({
  id, name: id === "production" ? "spatial_gate_combiner" : `spatial_gate_${id.replace(/^gate:/, "")}`, label: id,
  spec: id, production: id === "production", reads: ["170·psnr", "171·psnr"], n_reads: 2,
  integrated: [55, 64, 62, 61], mean: 60.5, score_source: "knee-psnr", stale: false,
  weights_source: "production", weights_reason: null,
  members: [
    { label: "170·psnr", name: "member_170", hours: 2, max_weight: 0.5, mean_weight: 0.25 },
    { label: "171·psnr", name: "member_171", hours: 4, max_weight: 0.25, mean_weight: 0.1 },
  ],
  hours: { max: 2, mean: 0.9, sum: 6 }, missing_hours: [], fit_hours: 0.29,
  ...patch,
});

export const payload = (patch: Partial<UniversalPayload> = {}): UniversalPayload => ({
  available: true, knees: [0.1, 1, 10, 100, 1000, 10000], bands: ["VIS", "Y_E", "J_E", "H_E"],
  integration: { from_e: 0.1, to_e: 10000 }, identity: { subset: "test", indices: Array.from({ length: 100 }, (_, i) => i) },
  production: { integrated: [55, 64, 62, 61], mean: 60.5, stale: false },
  mean_model: { integrated: [54, 63, 61, 60], mean: 59.5, stale: false },
  models: [
    scored(211, { num_res_blocks: 16, steps: 1_000_000, train_hours: 12.7, score: score(61.6) }),
    scored(210, { num_res_blocks: 16, steps: 200_000, train_hours: 2.5, score: score(61.0) }),
    scored(208, { num_res_blocks: 32, steps: 200_000, train_hours: 4.1, score: score(60.9) }),
    scored(215, { num_res_blocks: 16, steps: 200_000, train_hours: 1.9, hardware: { partition: "gpu_test", label: "A100 MIG slice", jobs: ["52306327"] },
      jit_compile: true, mixed_precision: true, score: score(60.8, { stale: true }), scored_hours: 1.2, scored_steps: 120_000 }),
    scored(196, { class: "member", spec: "member:member_196", knee: "fixed", output_knee: 10, learned_knees: null, num_res_blocks: 32,
      steps: 96_000, train_hours: null, score: null }),
    scored(212, { fingerprint: null, score: null }),
  ],
  combiners: [comb("production", { comparable: true }), comb("gate:p20s1", { mean: 60.2, integrated: [54.6, 63.8, 61.6, 60.8],
    hours: { max: 0.6, mean: 0.3, sum: 3 }, weights_source: "variant", score_source: "compare", comparable: true })],
  fasrc: [
    { name: "member_218", jobids: ["52727140"], state: "RUNNING", num_res_blocks: 16, steps: 2_000_000, knee: "learned", partition: "gpu",
      finished: false, pullable: false, registered: false },
    { name: "member_213", jobids: ["52123135"], state: "COMPLETED", num_res_blocks: 16, steps: 250_000, knee: "learned", partition: "gpu",
      finished: true, pullable: true, registered: false },
  ],
  weights_pending: [],
  notes: [],
  ...patch,
});
