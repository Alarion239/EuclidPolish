/* Fixtures of the Universal tab's payload (GET /ensemble/universal.json),
 * imported by tests only: universal.test.ts, notes.test.ts, tabs.test.tsx.
 * Five models — 211 (depth 16, 1M steps, the best), 210, 208 (depth 32),
 * 215 (a MIG slice, a stale score made at 1.2 h / 120k steps) and 212 (no
 * checkpoint yet) — production's own score (60.5 dB, on the scores' own
 * fields) and two FASRC rows (one pullable, one training).
 * `universalProduction` is the same payload while one of the listed models
 * is production (210 by default, 61.0 dB): its own score is the reference. */
import type { UniversalPayload, UniversalRow } from "../api";

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

export const payload = (patch: Partial<UniversalPayload> = {}): UniversalPayload => ({
  available: true, knees: [0.1, 1, 10, 100, 1000, 10000], bands: ["VIS", "Y_E", "J_E", "H_E"],
  integration: { from_e: 0.1, to_e: 10000 }, identity: { subset: "test", indices: Array.from({ length: 100 }, (_, i) => i) },
  production: { integrated: [55, 64, 62, 61], mean: 60.5, stale: false, model: "member_257" },
  production_model: "member_257", production_comparable: true,
  models: [
    scored(211, { num_res_blocks: 16, steps: 1_000_000, train_hours: 12.7, score: score(61.6) }),
    scored(210, { num_res_blocks: 16, steps: 200_000, train_hours: 2.5, score: score(61.0) }),
    scored(208, { num_res_blocks: 32, steps: 200_000, train_hours: 4.1, score: score(60.9) }),
    scored(215, { num_res_blocks: 16, steps: 200_000, train_hours: 1.9, hardware: { partition: "gpu_test", label: "A100 MIG slice", jobs: ["52306327"] },
      jit_compile: true, mixed_precision: true, score: score(60.8, { stale: true }), scored_hours: 1.2, scored_steps: 120_000 }),
    scored(212, { fingerprint: null, score: null }),
  ],
  fasrc: [
    { name: "member_218", jobids: ["52727140"], state: "RUNNING", num_res_blocks: 16, steps: 2_000_000, knee: "learned", partition: "gpu",
      finished: false, pullable: false, registered: false },
    { name: "member_213", jobids: ["52123135"], state: "COMPLETED", num_res_blocks: 16, steps: 250_000, knee: "learned", partition: "gpu",
      finished: true, pullable: true, registered: false },
  ],
  notes: [],
  ...patch,
});

/** The payload while universal model `name` is production, as the server
 *  sends it: `production` is its own score (`model` = its name; null while it
 *  is unscored), `production_comparable` once it is scored, each row's Δ vs
 *  production against it. */
export const universalProduction = (name = "member_210", patch: Partial<UniversalPayload> = {}): UniversalPayload => {
  const base = payload();
  const s = base.models.find((r) => r.name === name)?.score ?? null;
  const ref = s ? { integrated: s.integrated, mean: s.mean, stale: s.stale, model: name } : null;
  return {
    ...base, production_model: name, production: ref, production_comparable: ref != null,
    models: base.models.map((r) => ({ ...r,
      vs_production_db: ref?.mean != null && r.score?.mean != null ? +(r.score.mean - ref.mean).toFixed(4) : null })),
    ...patch,
  };
};
