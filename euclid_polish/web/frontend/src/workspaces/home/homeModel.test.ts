import { describe, expect, it } from "vitest";
import {
  bandMean, kneeHeadline, memberName, productionFromStatus, productionHeadline, productionModel, starfullMembers,
  type KneePayload,
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
