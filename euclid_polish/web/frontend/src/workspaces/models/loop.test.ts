/* The Models loop's pure logic added by the console regrouping (plan
 * "Console phases 2–6", Team M): the Leaderboard status line and its real
 * benchmark, the members waiting on FASRC, the gate share bar, the combiner
 * variant scope, the synthetic stamps and SR→HR recovery, Δm vs LR, the
 * running training batch and the Train submit label. */
import { describe, expect, it } from "vitest";
import type { Check, ExperimentSummary, MemberRow, TrainingJob, Variant } from "./api";
import {
  deltaMagText, gateShare, leaderboardBenchmark, lossFacets, runningBatches, stampSets, statusChecks, submitLabel,
  variantScope, waitingOnFasrc, compareRunPath, benchmarkChoices, coherenceOrder, forwardModelFacts, lrScheduleText,
} from "./model";

const check = (id: string, ok: boolean, action: string | null = null, tone: Check["tone"] = "warn"): Check =>
  ({ id, ok, tone: ok ? "good" : tone, title: id, detail: `${id} detail`, action });

const row = (n: number, patch: Partial<MemberRow> = {}): MemberRow => ({
  name: `member_${n}`, label: `${n}·psnr`, starless: false, regime: "starfull", origin: null, loss: "l2",
  status: "complete", timeout: false, job: null, psnr: 61, ...patch,
} as MemberRow);

describe("leaderboard status line", () => {
  it("lists only the failing checks, each fix offered once, and adds unscored members", () => {
    const s = statusChecks([check("eval-members", false, "evaluate"), check("eval-gate", false, "evaluate"),
      check("knee", true, "knee"), check("gate-members", false, "combiners", "info")],
    [row(196), row(178, { psnr: null }), row(179, { psnr: null })]);
    expect(s.map((c) => [c.id, c.fix])).toEqual([
      ["eval-members", "evaluate"], ["eval-gate", null], ["gate-members", "combiners"], ["member-psnr", "member-psnr"],
    ]);
    expect(s[3].detail).toBe("2 members have no test PSNR yet: 178, 179.");
  });

  it("is empty when everything is current (the page says 'All current')", () => {
    expect(statusChecks([check("knee", true, "knee")], [row(196)])).toEqual([]);
    expect(statusChecks([], null)).toEqual([]);
  });
});

describe("leaderboard real benchmark", () => {
  const perBand = (vis: number, y: number, j: number, h: number) => ({
    VIS: { hole_pct: vis }, Y_E: { hole_pct: y }, J_E: { hole_pct: j }, H_E: { hole_pct: h },
  });
  const agg = (vis: number, y: number, j: number, h: number, r = 1.1) => ({
    n_tiles: 9, per_band: perBand(vis, y, j, h), summary: { hole_pct_mean: (vis + y + j + h) / 4, hole_pct_max: Math.max(vis, y, j, h), median_R: r },
  });
  const EXPS: ExperimentSummary[] = [
    { id: "e-new", created: "2026-09-27T20:38:56Z", label: "seed vs pruning", tiles: ["t/a"],
      summary: { production: agg(17, 18, 22, 20, 1.18), mean: agg(20, 25, 26, 30, 1.2), "member:member_196": agg(30, 20, 20, 20, 0.9) } },
    { id: "e-old", created: "2026-09-26T02:00:00Z", label: "cached tile", tiles: ["t/b"], summary: { production: agg(1, 1, 1, 1) } },
  ];
  const CATALOG = { models: [
    { spec: "production", fingerprint: "p1" }, { spec: "mean", fingerprint: "m1" }, { spec: "member:member_196", fingerprint: "x9" },
  ] };

  it("scores gate, mean and best member from the newest Compare run of this production, fresh specs only", () => {
    const b = leaderboardBenchmark(EXPS, { id: "e-new", fingerprints: { production: "p1", mean: "m1", "member:member_196": "old" } }, CATALOG, "starfull");
    expect(b.state).toBe("current");
    if (b.state !== "current") return;
    expect(b.expId).toBe("e-new");
    expect(b.label).toBe("seed vs pruning");
    expect(b.facts("production")).toMatchObject({ worst: { short: "J", pct: 22 }, medianR: 1.18 });
    expect(b.facts("mean")?.worst?.short).toBe("H");
    expect(b.facts("member:member_196")).toBeNull();          // scored for an earlier checkpoint
    expect(b.facts("gate:nope")).toBeNull();
  });

  it("says there is no real benchmark for this membership when production changed since", () => {
    const b = leaderboardBenchmark(EXPS, { id: "e-new", fingerprints: { production: "p0" } }, CATALOG, "starfull");
    expect(b).toMatchObject({ state: "none", last: { expId: "e-new" } });
    expect(b.state === "none" && b.reason).toBe("no real benchmark for this membership: the last Sky › Compare run of production scored an earlier one");
  });

  it("waits for the run's fingerprints, and has nothing without a run or for starless", () => {
    expect(leaderboardBenchmark(EXPS, null, CATALOG, "starfull").state).toBe("loading");
    expect(leaderboardBenchmark(EXPS, { id: "e-new", fingerprints: { production: "p1" } }, null, "starfull").state).toBe("loading");
    expect(leaderboardBenchmark([], null, CATALOG, "starfull")).toMatchObject({ state: "none", reason: "no real benchmark yet: no Sky › Compare run has scored production" });
    expect(leaderboardBenchmark(EXPS, null, CATALOG, "starless")).toMatchObject({ state: "none", reason: "no real benchmark for starless: Sky › Compare scores starfull models" });
  });

  it("links a Compare run by its id", () => {
    expect(compareRunPath("20260927-203856-6cb6ad")).toBe("/sky/compare?exp=20260927-203856-6cb6ad");
    expect(benchmarkChoices(EXPS)[0].value).toBe("e-new");
  });
});

describe("members waiting on FASRC", () => {
  const job = (jobid: string, mode: string, state: string, names: string[], patch: Partial<TrainingJob> = {}): TrainingJob =>
    ({ jobid, mode, state, member_names: names, params: { mode }, ...patch });
  it("lists finished members not yet pulled (nor archived), in this regime only", () => {
    const w = waitingOnFasrc([
      job("3", "add", "RUNNING", ["member_203"]),
      job("2", "add", "COMPLETED", ["member_199", "member_200", "member_201"]),
      job("1", "add", "TIMEOUT", ["member_198"]),
      job("0", "add", "COMPLETED", ["member_150"], { params: { mode: "add", starless: "1" } }),
    ], [row(199), row(198)], ["member_201"], "starfull");
    expect(w.members).toEqual(["member_200"]);
  });
  it("skips members that are local in the other regime and batches that ended long ago", () => {
    const now = Date.parse("2026-09-27T20:00:00Z");
    const w = waitingOnFasrc([
      job("5", "add", "COMPLETED", ["member_199"], { ended_at: "2026-09-26T10:00:00Z" }),
      // an old job without the regime flag whose members live in starless here
      job("6", "add", "TIMEOUT", ["member_128", "member_129"], { ended_at: "2026-09-20T10:00:00Z" }),
      // finished in July and never pulled: not "new"
      job("7", "add", "COMPLETED", ["member_137"], { ended_at: "2026-07-10T10:00:00Z" }),
    ], [], ["member_129"], "starfull", { elsewhere: ["member_128"], now });
    expect(w.members).toEqual(["member_199"]);
  });

  it("lists continued members whose finished job went past the local step", () => {
    const w = waitingOnFasrc([job("4", "continue", "COMPLETED", ["member_178", "member_179"], { target_steps: 70000 })],
      [row(178, { step: 52000 }), row(179, { step: 70000 })], [], "starfull");
    expect(w.continued).toEqual(["member_178"]);
    expect(w.members).toEqual([]);
  });
});

describe("gate share", () => {
  it("is the peak share when the payload has one (what decides pruning), else the mean over bands", () => {
    expect(gateShare({ gate_usage: { VIS: 0.0002, Y_E: 0.0001, J_E: 0.0001, H_E: 0.0001 }, gate_usage_peak: { value: 0.48, band: "VIS", bin: "core" } }))
      .toMatchObject({ value: 0.48, mean: 0.000125, text: "<0.1% mean · 48% peak (VIS cores)" });
    expect(gateShare({ gate_usage: { VIS: 0.1, Y_E: 0.3, J_E: null, H_E: 0.2 } })).toMatchObject({ value: expect.closeTo(0.2, 9), text: "20%" });
    expect(gateShare({}).value).toBeNull();
  });
});

describe("combiner variant scope", () => {
  const v = (name: string, patch: Partial<Variant> = {}) => ({
    name, kind: "gate", production: false, backup: false, membership: { current: true, missing: [], extra: [] }, ...patch,
  }) as Variant;
  const VARIANTS = [
    v("spatial_gate_combiner", { production: true, membership: { current: false, missing: ["1"], extra: [] } }),
    v("spatial_gate_p20"), v("spatial_gate_old", { membership: { current: false, missing: [], extra: ["2"] } }),
    // every member it reads is still active, but 10 joined after its fit: an earlier membership
    v("spatial_gate_20m", { membership: { current: true, missing: [], extra: ["member_21", "member_22"] } }),
    v("spatial_gate_backup_20260927", { backup: true }), v("raw_rbf", { kind: "rbf" }),
  ];
  it("shows production and the variants fitted for the current membership; the rest behind history", () => {
    const s = variantScope(VARIANTS, false);
    expect(s.shown.map((x) => x.name)).toEqual(["spatial_gate_combiner", "spatial_gate_p20"]);
    expect(s.hidden).toBe(3);
    expect(variantScope(VARIANTS, true).shown.map((x) => x.name)).toEqual([
      "spatial_gate_combiner", "spatial_gate_p20", "spatial_gate_old", "spatial_gate_20m", "spatial_gate_backup_20260927"]);
  });
});

describe("synthetic stamps and SR → HR recovery", () => {
  const e = (id: string, grade: string, lr: number, sr: number, patch: Record<string, unknown> = {}) => ({
    id, grade, ok: "True", psnr_lr_hr: String(lr), psnr_sr_hr: String(sr), flux_ratio_sr_over_lr: "0.9",
    lr_total_e: "100", sr_total_e: "90", state: "stale", n_members: 9, viewer_id: id, ...patch,
  });
  it("groups the synthetic stamps with medians, the SR gain and how many SR brought closer to truth", () => {
    const sets = stampSets([
      e("a", "syn-lens", 90, 100), e("b", "syn-lens", 92, 91), e("c", "syn-lens", 94, 104),
      e("d", "syn-gal", 80, 85, { state: "current" }), e("x", "A", 0, 0), e("f", "syn-gal", 1, 2, { ok: "False" }),
    ]);
    expect(sets.map((s) => [s.grade, s.label, s.n])).toEqual([["syn-lens", "Syn lens", 3], ["syn-gal", "Syn gal", 1]]);
    expect(sets[0]).toMatchObject({ medianLr: 92, medianSr: 100, gain: 10, improved: 2, stale: 3, madeBy: 9 });
    expect(sets[0].points.map((p) => [p.x, p.y, p.id])).toEqual([[90, 100, "a"], [92, 91, "b"], [94, 104, "c"]]);
    expect(sets[1]).toMatchObject({ stale: 0, improved: 1 });
  });

  it("states SR's magnitude change vs LR, warn-toned beyond 0.1 mag", () => {
    expect(deltaMagText(100, 90)).toEqual({ text: "Δm +0.11 (flux ×0.90)", warn: true });
    expect(deltaMagText(100, 99)).toEqual({ text: "Δm +0.01 (flux ×0.99)", warn: false });
    expect(deltaMagText(100, 120)?.text).toBe("Δm −0.20 (flux ×1.20)");
    expect(deltaMagText(0, 5)).toBeNull();
    expect(deltaMagText(null, 5)).toBeNull();
  });
});

describe("diagnostics shaping", () => {
  it("orders the coherence dot plot by overall score, highest first, keeping missing last", () => {
    const rows = coherenceOrder([
      { id: "a", label: "a", overall: 0.5, sr: 0.1 }, { id: "b", label: "b", overall: null, sr: 0.3 }, { id: "c", label: "c", overall: 0.9, sr: 0.2 },
    ]);
    expect(rows.map((r) => r.id)).toEqual(["c", "a", "b"]);
  });

  it("facets the training loss by loss type (their scales differ)", () => {
    const f = lossFacets([{ loss_norm: "l2", name: "m1" }, { loss_norm: "L1", name: "m2" }, { loss_norm: "l2", name: "m3" }, { loss_norm: "", name: "m4" }]);
    expect(f.map((x) => [x.loss, x.curves.map((c) => c.name)])).toEqual([["l1", ["m2", "m4"]], ["l2", ["m1", "m3"]]]);
  });
});

describe("train", () => {
  it("names the regime and the count in the submit button", () => {
    expect(submitLabel("add", 4, "starfull")).toBe("Submit 4 STARFULL members to SLURM");
    expect(submitLabel("add", 1, "starless")).toBe("Submit 1 STARLESS member to SLURM");
    expect(submitLabel("fork", 2, "starfull")).toBe("Submit 2 STARFULL forks to SLURM");
    expect(submitLabel("continue", 3, "starfull")).toBe("Continue 3 members on SLURM");
    expect(submitLabel("continue", 0, "starfull")).toBe("Continue members on SLURM");
  });

  it("reads the running training batches from the live SLURM rows, with their members", () => {
    const b = runningBatches(
      [{ jobid: "48107719", state: "RUNNING", step_id: "ensemble_train", progress_step: 10650, progress_total: 70000, time: "1:02:00", time_limit: "3:00:00" },
        { jobid: "5", state: "RUNNING", step_id: "synthetic_generate" }],
      [{ jobid: "48107719", mode: "add", member_names: ["member_199", "member_200", "member_201", "member_202"] }],
    );
    expect(b).toEqual([{ jobid: "48107719", state: "RUNNING", members: ["member_199", "member_200", "member_201", "member_202"],
      text: "Job 48107719 · members 199–202 · running · step 10,650 / 70,000 (15%) · 1:02:00 of 3:00:00" }]);
    expect(runningBatches([{ jobid: "7", state: "PENDING", step_id: "ensemble_train", reason: "Priority" }], [])[0].text)
      .toBe("Job 7 · pending (Priority)");
  });

  it("lists the forward-model values the batch takes from System › Config", () => {
    const facts = forwardModelFacts({ psf_warp_prob: 1, psf_warp_alpha_max: 5, psf_warp_sigma: 3, saturation_mask_prob: 0.5, lr_peak: 2e-4 });
    expect(facts.map((f) => [f.label, f.value, f.unit ?? null])).toEqual([
      ["PSF warp probability", "1", null], ["PSF warp α max", "5", null], ["PSF warp σ", "3", "HR px"], ["Saturation mask probability", "0.5", null],
    ]);
    expect(lrScheduleText({ lr_peak: 0.0002, lr_final: 1e-6, lr_warmup_steps: 2000, plateau_lr_enabled: false }))
      .toBe("LR warmup 2,000 steps to 2e-4, then cosine to 1e-6; plateau guard off");
    expect(lrScheduleText({})).toBeNull();
  });
});
