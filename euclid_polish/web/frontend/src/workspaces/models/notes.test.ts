import { describe, expect, it } from "vitest";
import type { CompareReport, KneeModel, Overview, Variant } from "./api";
import { benchmarkExperiment, kneeLeaderboard } from "./model";
import { compareNote, compareRows, evaluationNote, kneeNote, promoteNote, utcText, variantNote } from "./notes";

const OVERVIEW: Overview = {
  active_members: ["196·psnr", "195·psnr"], n_members: 30, test_present: true,
  evaluated_at: "2026-09-25T23:32:26+00:00", summary: {}, eval_subset: "test",
  headline: {
    metric: "vis_asinh", knee_e: 100, n_scored: 100,
    production: { psnr: 44.123, vs_mean_db: 0.404, vs_best_member_db: 0.231 },
    mean: { psnr: 43.72, vs_mean_member_db: 1.2 },
    best_member: { psnr: 43.89, label: "196·psnr", mean_member_psnr: 42.5 },
    knee: {
      available: true, stale: false, n_fields: 100, integration: { from_e: 0.1, to_e: 10000 },
      production: 38.21, production_bands: [39.1, 38, 37.9, 37.84], mean: 38.07, best_member: 37.91, best_member_label: "196·psnr",
    },
  },
  checks: [
    { id: "eval", ok: true, tone: "good", title: "Evaluation current", detail: "" },
    { id: "knee", ok: false, tone: "warn", title: "Knee curves stale", detail: "members changed" },
  ],
  production_gate: { available: true, n_members: 30, mix_space: "linear", fitted_at: "2026-09-25T22:30:13+00:00" },
};

describe("utcText", () => {
  it("prints an ISO time as a UTC minute", () => {
    expect(utcText("2026-09-25T23:32:26+00:00")).toBe("2026-09-25 23:32 UTC");
    expect(utcText("2026-09-25T19:32:26-04:00")).toBe("2026-09-25 23:32 UTC");
    expect(utcText(null)).toBe("—");
    expect(utcText("not a date")).toBe("not a date");
  });
});

describe("evaluation note", () => {
  it("states the Evaluate summary's facts with their definitions", () => {
    const md = evaluationNote(OVERVIEW);
    expect(md).toContain("**Ensemble evaluation** — 2026-09-25 23:32 UTC · 30 members · 100 test fields");
    expect(md).toContain("- Production gate: 44.12 dB (+0.23 dB vs best member, +0.40 dB vs plain mean)");
    expect(md).toContain("- Plain mean: 43.72 dB (+1.20 dB vs mean member)");
    expect(md).toContain("- Best member: #196 43.89 dB");
    expect(md).toContain("- ∫PSNR over 0.1–10k e⁻ (100 fields): production gate 38.21 dB (+0.14 dB vs plain mean, +0.30 dB vs #196) · VIS 39.10 · Y 38.00 · J 37.90 · H 37.84");
    expect(md).toContain("- Production gate fitted for 30 members, linear mix, 2026-09-25 22:30 UTC");
    expect(md).toContain("- Open: Knee curves stale — members changed");
    expect(md).toContain("VIS asinh PSNR at knee 100 e⁻");
  });
  it("says what is missing instead of printing blanks", () => {
    const md = evaluationNote({ ...OVERVIEW, evaluated_at: null, checks: [],
      headline: { ...OVERVIEW.headline, production: {}, knee: { available: false, stale: false } } });
    expect(md).toContain("**Ensemble evaluation** — not evaluated yet");
    expect(md).toContain("- Production gate: no score");
    expect(md).not.toContain("∫PSNR");
    expect(md).not.toContain("Open:");
  });
});

const model = (id: string, kind: KneeModel["kind"], label: string, v: number): KneeModel => ({
  id, kind, label, psnr: [[v, v + 1, v + 2, v + 3], [v, v + 1, v + 2, v + 3]], integrated: [v, v + 1, v + 2, v + 3],
});

describe("knee leaderboard note", () => {
  const models = [
    model("spatial_gate", "combiner", "Spatial gate", 40), model("mean", "mean", "mean", 39),
    model("member_196", "member", "196·psnr", 38), model("member_195", "member", "195·psnr", 37), model("member_170", "member", "170·psnr", 36),
  ];
  const knees = [0.1, 10000];
  const board = kneeLeaderboard(models, knees, [0.1, 10000]);
  it("is a markdown table with the integration range, band and fields", () => {
    const md = kneeNote(board, { range: [0.1, 10000], full: [0.1, 10000], band: "all", bands: ["VIS", "Y_E", "J_E", "H_E"], nFields: 100 });
    expect(md).toContain("**Knee-integrated PSNR** — ∫ over 0.1–10k e⁻ (the full range, uniform in log knee) · all bands · 100 test fields");
    expect(md).toContain("| # | Model | Trained | ∫VIS | ∫Y | ∫J | ∫H | ∫ mean | vs mean |");
    expect(md).toContain("| 1 | production gate | combiner | 40.00 | 41.00 | 42.00 | 43.00 | 41.50 | +1.00 |");
    expect(md).toContain("| 3 | #196 | 100 e⁻ | 38.00 |");
    expect(md.split("\n").filter((l) => l.startsWith("| ") && !l.startsWith("| #") && !l.startsWith("| ---")).length).toBe(5);
  });
  it("keeps the combiners and the mean, and only the top members", () => {
    const md = kneeNote(board, { range: [1, 100], full: [0.1, 10000], band: "VIS", bands: ["VIS", "Y_E", "J_E", "H_E"], nFields: 100, top: 1 });
    expect(md).toContain("∫ over 1–100 e⁻ (a sub-range of 0.1–10k e⁻, uniform in log knee) · VIS");
    expect(md).toContain("#196");
    expect(md).not.toContain("#195");
    expect(md).toContain("Top 1 of 3 members, plus the plain mean and the combiners.");
  });
});

const VARIANT: Variant = {
  name: "spatial_gate_pruned", kind: "gate", spec: "gate:pruned", production: false, backup: false,
  member_labels: [], reads: [], n_members: 20, n_reads: 6, pruned: true, mix_space: "linear", use_lr: false, width: 32,
  fitted_at: "2026-09-25T22:30:13+00:00", membership: { current: false, missing: ["171·psnr"], extra: [] },
  applies_to_test_cubes: true, fit: { steps: 2000, loss_knees_e: "all" },
  selected: { step: 1750, loss: 0.91234, vis_psnr: 44.1 }, history: [],
};

describe("gate variant note", () => {
  it("states the fit, its scores against production and the real-data holes", () => {
    const bench = benchmarkExperiment([{ id: "20260927-x", created: "2026-09-27", label: "cached tile ra0273", summary: {
      "gate:pruned": { n_tiles: 1, per_band: { VIS: { hole_pct: 19 }, Y_E: { hole_pct: 13 }, J_E: { hole_pct: 19 }, H_E: { hole_pct: 30 } } } } }]);
    const md = variantNote({ ...VARIANT, testVis: 44.2, kneeMean: 38.3, bench: bench?.bySpec.get("gate:pruned") ?? null },
      { prod: { testVis: 44.1, kneeMean: 38.21 }, benchmark: bench });
    expect(md).toContain("**Gate variant `pruned`** — fitted 2026-09-25 22:30 UTC");
    expect(md).toContain("- Reads 6 of 20 members (not the active set: missing #171)");
    expect(md).toContain("- Fit: linear mix · width 32 · 2000 steps · loss knees all · no LR input");
    expect(md).toContain("- Held-out (step 1750): loss 0.9123 · VIS 44.10 dB");
    expect(md).toContain("- Test: VIS 44.20 dB (+0.10 vs production) · ∫PSNR 38.30 dB (+0.09 vs production)");
    expect(md).toContain("- Real holes on cached tile ra0273 (experiment `20260927-x`): VIS 19 · Y 13 · J 19 · H 30 % (worst: H)");
  });
});

const REPORT: CompareReport = {
  id: "cmp-1", created: "2026-09-25T22:40:00+00:00", members: ["a", "b"], bands: ["VIS", "Y_E"], brightness_names: [],
  methods: ["mean", "gate:spatial_gate_combiner", "gate:spatial_gate_linear"],
  n_fields: { natural: 100, blackout: 40 },
  groups: { natural: {
    mean: { band_psnr: [43.7, 42], bin_mse: [], halo_mse: [], hole_mse: [] },
    "member:196·psnr": { band_psnr: [43.9, 41.5], bin_mse: [], halo_mse: [], hole_mse: [] },
    "member:195·psnr": { band_psnr: [43.1, 41.9], bin_mse: [], halo_mse: [], hole_mse: [] },
    "gate:spatial_gate_combiner": { band_psnr: [44.1, 42.4], bin_mse: [], halo_mse: [], hole_mse: [] },
    "gate:spatial_gate_linear": { band_psnr: [44.2, 42.3], bin_mse: [], halo_mse: [], hole_mse: [] },
  } },
  knee: { knees: [1], n_fields: 100, methods: { "gate:spatial_gate_linear": { psnr: [], integrated: [38, 39] } } },
};

describe("compare report", () => {
  it("orders the rows: mean, the best member, then the methods", () => {
    expect(compareRows(REPORT, "natural")).toEqual(["mean", "member:196·psnr", "gate:spatial_gate_combiner", "gate:spatial_gate_linear"]);
    expect(compareRows(REPORT, "blackout")).toEqual([]);
  });
  it("logs the natural-field table", () => {
    const md = compareNote(REPORT);
    expect(md).toContain("**Gate compare** — report `cmp-1`, 2026-09-25 22:40 UTC · 100 natural + 40 blackout test fields · 2 cube members");
    expect(md).toContain("| Method | VIS | Y | ∫ mean |");
    expect(md).toContain("| best member #196 | 43.900 | 41.500 | — |");
    expect(md).toContain("| linear | 44.200 | 42.300 | 38.500 |");
    expect(md).toContain("| production | 44.100 | 42.400 | — |");
  });
  it("copes with a report scored without blackout fields (an empty group of null PSNRs)", () => {
    const none = { band_psnr: [null, null], bin_mse: [0], halo_mse: [0], hole_mse: [0] };
    const empty: CompareReport = { ...REPORT, n_fields: { natural: 100, blackout: 0 },
      groups: { ...REPORT.groups, blackout: { mean: none, "member:196·psnr": none, "member:195·psnr": none } } };
    expect(compareRows(empty, "blackout")).toEqual(["mean", "member:196·psnr"]);
    expect(compareNote(empty)).toContain("· 100 natural test fields ·");
  });
});

describe("promote note", () => {
  it("records the swap and the rollback backup", () => {
    const md = promoteNote({ promoted: "spatial_gate_linear", backup: "spatial_gate_backup_20260927T120000Z", test_rescored: true },
      { testVis: 44.2, kneeMean: 38.3 });
    expect(md).toContain("**Promoted `linear` to production**");
    expect(md).toContain("- The previous production gate is backed up as `spatial_gate_backup_20260927T120000Z` (promote it to roll back)");
    expect(md).toContain("- Test cubes re-scored: production now VIS 44.20 dB · ∫PSNR 38.30 dB");
    const plain = promoteNote({ promoted: "spatial_gate_linear", test_rescored: false }, null);
    expect(plain).toContain("- Test cubes not re-scored (the variant does not read the cached members)");
  });
});
