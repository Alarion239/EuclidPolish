import { describe, expect, it } from "vitest";
import {
  bandMean, kneeHeadline, memberName, memberRange, productionFromStatus, productionHeadline, productionModel, runningItems, starfullMembers,
  trackingCatchUpNote, unloggedItems, type KneePayload,
} from "./homeModel";

describe("productionHeadline (eval_summary.json)", () => {
  it("headlines the production spatial gate from its spatial_gate_* keys", () => {
    const h = productionHeadline({
      ensemble_psnr: 58.375, mean_member_psnr: 57.22, ensemble_gain_db: 1.155,
      combiner_psnr: 99, // the RBF block — never the production number
      spatial_gate_combiner_psnr: 59.2354, spatial_gate_combiner_vs_mean_db: 0.8601,
      spatial_gate_combiner_vs_best_member_db: 0.2948,
    }, false);
    expect(h).toEqual({
      kind: "gate", psnr: 59.2354, vsMean: 0.8601, vsBest: 0.2948, meanPsnr: 58.375, stale: false,
    });
  });

  it("falls back to the plain mean, labelled, with its gain over the mean member", () => {
    const h = productionHeadline({ ensemble_psnr: 44.123, mean_member_psnr: 43.913, ensemble_gain_db: -0.4 }, true);
    if (h?.kind !== "mean") throw new Error("expected the plain-mean headline");
    expect(h.psnr).toBe(44.123);
    expect(h.vsMeanMember).toBeCloseTo(0.21, 6);   // derived, not the ambiguous ensemble_gain_db
    expect(h.stale).toBe(true);
  });

  it("is null without a summary", () => {
    expect(productionHeadline(null, false)).toBeNull();
    expect(productionHeadline({}, false)).toBeNull();
  });
});

const KNEE: KneePayload = {
  available: true, stale: false, n_fields: 100, bands: ["VIS", "Y_E", "J_E", "H_E"],
  models: [
    { id: "member_0", kind: "member", label: "169·psnr", integrated: [54, 63, 60, 59] },
    { id: "member_27", kind: "member", label: "196·psnr", integrated: [54.7435, 64.1229, 60.6616, 60.3038] },
    { id: "ensemble_mean", kind: "mean", label: "ensemble mean", integrated: [53.9358, 62.9001, 60.143, 59.4462] },
    { id: "spatial_gate", kind: "combiner", label: "spatial gate", integrated: [55.7978, 64.9395, 62.0264, 61.1288] },
    { id: "raw_incremental_minmeanmax_rbf", kind: "combiner", label: "RBF", integrated: [70, 70, 70, 70] },
  ],
};

describe("kneeHeadline (ensemble_knee_psnr.json)", () => {
  it("is the production gate's band-mean integrated PSNR vs the best member and the mean", () => {
    const h = kneeHeadline(KNEE)!;
    expect(h.gate).toBeCloseTo(60.9731, 3);
    expect(h.best?.name).toBe("member_196");
    expect(h.best?.value).toBeCloseTo(59.958, 3);
    expect(h.vsBest).toBeCloseTo(1.015, 2);
    expect(h.vsMean).toBeCloseTo(1.8669, 3);
    expect(h.perBand.map((b) => b.band)).toEqual(["VIS", "Y_E", "J_E", "H_E"]);
    expect(h.perBand[1].value).toBeCloseTo(64.9395, 4);
    expect(h.nFields).toBe(100);
    expect(h.stale).toBe(false);
  });

  it("never mistakes the RBF combiner for production", () => {
    const h = kneeHeadline({ ...KNEE, models: (KNEE.models ?? []).filter((m) => m.id !== "spatial_gate") })!;
    expect(h.gate).toBeNull();
    expect(h.mean).toBeCloseTo(59.1063, 3);
  });

  it("is null when the curves are not computed", () => {
    expect(kneeHeadline({ available: false, stale: false })).toBeNull();
    expect(kneeHeadline(null)).toBeNull();
  });
});

describe("starfullMembers", () => {
  it("counts the regime labels of /api/models (never all active members)", () => {
    expect(starfullMembers({ members: ["169·psnr", "170·psnr"] }, null)).toEqual({ count: 2, source: "models", starless: null });
  });

  it("falls back to /api/system/production's STARFULL count and reports the starless one", () => {
    const production = { members: 30, starless_members: 12 };
    expect(starfullMembers(null, production)).toEqual({ count: 30, source: "production", starless: 12 });
    expect(starfullMembers({ members: ["1·psnr"] }, production)).toEqual({ count: 1, source: "models", starless: 12 });
  });

  it("is null with nothing loaded", () => {
    expect(starfullMembers(null, null)).toBeNull();
    expect(starfullMembers(null, { eval_summary: null })).toBeNull();
  });
});

describe("productionFromStatus (/ensemble/status.json fallback)", () => {
  it("maps the heavy ensemble status onto the production payload", () => {
    const status = {
      eval_summary: { spatial_gate_combiner_psnr: 59.2, member_labels: ["169·psnr"], nested: { x: 1 } },
      eval_summary_stale: true,
      members: [{ name: "member_169", starless: false }, { name: "member_170", starless: false }, { name: "member_105", starless: true }],
    };
    expect(productionFromStatus(status)).toEqual({
      eval_summary: { spatial_gate_combiner_psnr: 59.2 },
      stale: true, stale_reason: "Membership changed since the last evaluation",
      members: 2, starless_members: 1,
    });
  });

  it("is null without a payload and tolerates a missing summary", () => {
    expect(productionFromStatus(null)).toBeNull();
    expect(productionFromStatus({ members: [] })).toEqual({
      eval_summary: null, stale: false, stale_reason: null, members: 0, starless_members: 0,
    });
  });
});

describe("productionModel (/api/models)", () => {
  const PROD = {
    spec: "production", kind: "production", available: true, reason: null, combiner_kind: "spatial_gate",
    label: "Production · spatial gate (convolutional, convex)",
    details: { mix_space: "linear", fitted_at: "2026-09-25T22:30:13+00:00" },
  };

  it("names the production combiner, its mix space and when it was fitted", () => {
    expect(productionModel({ production_kind: "spatial_gate", models: [PROD] })).toEqual({
      label: "spatial gate", mix: "linear", fittedAt: "2026-09-25T22:30:13+00:00", available: true, reason: null,
    });
  });

  it("falls back to the production kind and reports why it is unavailable", () => {
    const m = productionModel({
      production_kind: "spatial_gate",
      models: [{ spec: "production", available: false, reason: "fitted for other members" }],
    });
    expect(m).toEqual({ label: "spatial gate", mix: null, fittedAt: null, available: false, reason: "fitted for other members" });
  });

  it("is null without a production spec", () => {
    expect(productionModel({ models: [{ spec: "mean", available: true }] })).toBeNull();
    expect(productionModel(null)).toBeNull();
  });
});

describe("helpers", () => {
  it("averages finite band values", () => {
    expect(bandMean([1, 2, 3, 6])).toBe(3);
    expect(bandMean([1, null, Number.NaN, 3])).toBe(2);
    expect(bandMean([])).toBeNull();
    expect(bandMean(undefined)).toBeNull();
  });

  it("names members from their labels", () => {
    expect(memberName("196·psnr")).toBe("member_196");
    expect(memberName("member_7")).toBe("member_7");
  });
});

describe("tracking catch-up note (Home › Quick actions › Log to tracking)", () => {
  const check = { id: "tracking", label: "Tracking log", state: "warn" as const, title: "Results since the last tracking entry (2026-09-21)",
    facts: { last_entry: "2026-09-21T14:42:29+00:00", unlogged: [
      { at: "2026-09-27T02:44:21+00:00", label: "experiment" }, { at: "2026-09-25T23:33:42+00:00", label: "PSNR vs knee" },
      { at: "2026-09-25T23:32:26+00:00", label: "evaluation" }, { at: "2026-09-25T22:30:13+00:00", label: "production gate fit" },
    ] } };
  const facts = {
    knee: { gate: 60.97, mean: 59.1, best: { name: "member_196", label: "196·psnr", value: 59.96 }, vsBest: 1.01, vsMean: 1.87, perBand: [], nFields: 100, stale: false },
    prod: { kind: "gate" as const, psnr: 59.2354, vsMean: 0.8601, vsBest: 0.2948, meanPsnr: 58.4, stale: false },
    members: 30,
    production: { label: "spatial gate", mix: "linear", fittedAt: "2026-09-25T22:30:13+00:00", available: true, reason: null },
  };
  it("reads the alert's unlogged items (newest first)", () => {
    expect(unloggedItems(check).map((u) => u.label)).toEqual(["experiment", "PSNR vs knee", "evaluation", "production gate fit"]);
    expect(unloggedItems(undefined)).toEqual([]);
    expect(unloggedItems({ ...check, facts: { unlogged: "nope" } })).toEqual([]);
  });
  it("pre-fills one line per unlogged result with its headline numbers", () => {
    const md = trackingCatchUpNote(check, facts);
    expect(md).toContain("**Catch-up** — results since the last tracking entry (2026-09-21 14:42 UTC)");
    expect(md).toContain("- experiment — 2026-09-27 02:44 UTC: see Sky › Experiments");
    expect(md).toContain("- PSNR vs knee — 2026-09-25 23:33 UTC: ∫PSNR production gate 60.97 dB (+1.01 dB vs member 196, +1.87 dB vs plain mean, 100 fields)");
    expect(md).toContain("- evaluation — 2026-09-25 23:32 UTC: test PSNR production gate 59.24 dB (+0.29 dB vs best member, +0.86 dB vs plain mean), 30 STARFULL members");
    expect(md).toContain("- production gate fit — 2026-09-25 22:30 UTC: spatial gate, linear mix, fitted for the current members");
  });
  it("without unlogged results it notes the production model as it is", () => {
    const md = trackingCatchUpNote({ facts: { last_entry: "2026-09-27T00:00:00Z", unlogged: [] } }, facts);
    expect(md).toContain("**Production model** — 30 STARFULL members");
    expect(md).toContain("- ∫PSNR production gate 60.97 dB");
    expect(md).toContain("- Test PSNR production gate 59.24 dB");
  });
});

describe("running now (Home, the jobs feed)", () => {
  const local = (id: string, patch: Record<string, unknown> = {}) => ({
    job_id: id, label: `job ${id}`, status: "running", duration: 0, error: null, log: null, log_truncated: false,
    cancellable: false, cancel_requested: false, result: null, progress: { current: 0, total: 0, pct: 0, label: "" }, ...patch,
  });

  it("names a training array by its members, compressed to ranges", () => {
    expect(memberRange(["member_199", "member_200", "member_201", "member_202"])).toBe("members 199–202");
    expect(memberRange(["member_195", "member_196", "member_199", "member_200", "member_201"])).toBe("members 195, 196, 199–201");
    expect(memberRange(["member_07"])).toBe("member 07");
    expect(memberRange([])).toBeNull();
  });

  it("lists the live SLURM jobs first, with their members and progress, then the local ones", () => {
    const items = runningItems(
      [local("a", { progress: { current: 4, total: 10, pct: 40, label: "" } }), local("b", { status: "done" })],
      [
        { jobid: "1", state: "RUNNING", label: "Train ensemble members", progress_step: 10500, progress_total: 70000,
          params_json: JSON.stringify({ mode: "add", member_names: "member_199,member_200,member_201,member_202" }) },
        { jobid: "2", state: "PENDING", label: "Generate synthetic records" },
        { jobid: "3", state: "COMPLETED", label: "done already" },
      ],
    );
    expect(items.map((i) => i.text)).toEqual([
      "members 199–202 on FASRC · 15%", "Generate synthetic records on FASRC · queued", "job a on this laptop · 40%",
    ]);
  });

  it("reads a continue submission's members and falls back to the label", () => {
    const items = runningItems([local("c")], [
      { jobid: "4", state: "RUNNING", label: "Continue members", params_json: JSON.stringify({ mode: "continue", members: "member_178, member_179" }) },
      { jobid: "5", state: "RUNNING", label: null, step_id: "ensemble_evaluate", params_json: "{not json" },
    ]);
    expect(items.map((i) => i.text)).toEqual([
      "members 178, 179 on FASRC", "ensemble_evaluate on FASRC", "job c on this laptop",
    ]);
  });

  it("is empty when nothing runs", () => {
    expect(runningItems([local("d", { status: "failed" })], [])).toEqual([]);
  });
});
