import { describe, expect, it } from "vitest";
import {
  bandMean, compareRunPath, homeCaption, memberRange, productionModel, productionRealHoles, trackingCatchUpNote, universalScore, unloggedItems,
  type ExperimentSummary,
} from "./homeModel";

describe("productionModel (/api/models)", () => {
  it("reads the universal model the catalogue names production", () => {
    expect(productionModel({
      production: { kind: "universal", spec: "universal:member_257", name: "member_257", label: "Universal 257", available: true, reason: null },
      models: [{ spec: "production", available: true, label: "Production · Universal 257" }],
    })).toEqual({ label: "Universal 257", name: "member_257", available: true, reason: null });
    // an older payload without `production`: the spec's label and details
    expect(productionModel({ models: [{ spec: "production", available: false, reason: "no checkpoint", label: "Production · Universal 258",
      details: { member: "member_258" } }] })).toEqual({ label: "Universal 258", name: "member_258", available: false, reason: "no checkpoint" });
  });

  it("is null without a production spec", () => {
    expect(productionModel({ models: [{ spec: "mean", available: true }] })).toBeNull();
    expect(productionModel(null)).toBeNull();
  });
});

describe("the production model's score (/ensemble/universal.json)", () => {
  it("is production's own ∫PSNR, with when its row was scored", () => {
    expect(universalScore({ production_model: "member_257", production: { integrated: [57, 67, 64, 63], mean: 62.82, stale: false },
      models: [{ name: "member_258" }, { name: "member_257", score: { scored_at: "2026-10-30T12:00:00Z" } }] }))
      .toEqual({ mean: 62.82, bands: [57, 67, 64, 63], stale: false, scoredAt: "2026-10-30T12:00:00Z" });
    // the band mean without a mean; nothing before it is scored
    expect(universalScore({ production: { integrated: [60, 62, 64, 66], mean: null } })?.mean).toBe(63);
    expect(universalScore({ production: null })).toBeNull();
  });

  it("captions the verdict with when it was scored, or that it cannot run", () => {
    const prod = { label: "Universal 257", name: "member_257", available: true, reason: null };
    const now = Date.parse("2026-10-30T15:00:00Z");
    expect(homeCaption({ production: prod, score: { mean: 62.8, bands: [], stale: false, scoredAt: "2026-10-30T12:00:00Z" }, now })).toEqual(["scored 3 h ago"]);
    expect(homeCaption({ production: { ...prod, available: false }, now })).toEqual(["cannot run"]);
    expect(homeCaption({ production: null, now })).toEqual([]);
  });
});

describe("helpers", () => {
  it("averages finite band values", () => {
    expect(bandMean([1, 2, 3, 6])).toBe(3);
    expect(bandMean([1, null, Number.NaN, 3])).toBe(2);
    expect(bandMean([])).toBeNull();
    expect(bandMean(undefined)).toBeNull();
  });
});

describe("tracking catch-up note (the Home notebook alert › Log)", () => {
  const check = { id: "tracking", label: "Tracking log", state: "warn" as const, title: "Results since the last tracking entry (2026-09-21)",
    facts: { last_entry: "2026-09-21T14:42:29+00:00", unlogged: [
      { at: "2026-09-27T02:44:21+00:00", label: "experiment" }, { at: "2026-09-25T23:33:42+00:00", label: "universal scores" },
    ] } };
  const facts = {
    production: { label: "Universal 257", name: "member_257", available: true, reason: null },
    score: { mean: 62.82, bands: [57, 67, 64, 63], stale: false, scoredAt: "2026-10-30T12:00:00Z" },
  };
  it("reads the alert's unlogged items (newest first)", () => {
    expect(unloggedItems(check).map((u) => u.label)).toEqual(["experiment", "universal scores"]);
    expect(unloggedItems(undefined)).toEqual([]);
    expect(unloggedItems({ ...check, facts: { unlogged: "nope" } })).toEqual([]);
  });
  it("pre-fills one line per unlogged result with the production model", () => {
    const md = trackingCatchUpNote(check, facts);
    expect(md).toContain("**Catch-up** — results since the last tracking entry (2026-09-21 14:42 UTC)");
    expect(md).toContain("- experiment — 2026-09-27 02:44 UTC: see Sky › Compare");
    expect(md).toContain("- universal scores — 2026-09-25 23:33 UTC: Universal 257 is production, ∫PSNR 62.82 dB");
  });
  it("without unlogged results it notes the production model as it is", () => {
    expect(trackingCatchUpNote({ facts: { last_entry: "2026-09-27T00:00:00Z", unlogged: [] } }, facts))
      .toBe("**Production model**\n\n- Universal 257 is production, ∫PSNR 62.82 dB");
    expect(trackingCatchUpNote({ facts: { unlogged: [] } }, { production: { ...facts.production, available: false, reason: "no checkpoint" } }))
      .toBe("**Production model**\n\n- Universal 257 is production, cannot run (no checkpoint)");
  });
});

describe("member ranges (the running line)", () => {
  it("names a training array by its members, compressed to ranges", () => {
    expect(memberRange(["member_199", "member_200", "member_201", "member_202"])).toBe("members 199–202");
    expect(memberRange(["member_195", "member_196", "member_199", "member_200", "member_201"])).toBe("members 195, 196, 199–201");
    expect(memberRange(["member_07"])).toBe("member 07");
    expect(memberRange([])).toBeNull();
  });
});

describe("real holes from Sky › Compare", () => {
  const perBand = (vis: number, y: number, j: number, h: number) => ({
    VIS: { hole_pct: vis }, Y_E: { hole_pct: y }, J_E: { hole_pct: j }, H_E: { hole_pct: h },
  });
  const exps: ExperimentSummary[] = [
    { id: "e-old", created: "2026-09-20T00:00:00Z", label: "poster", summary: { production: { n_tiles: 1, per_band: perBand(1, 1, 1, 1) } } },
    { id: "e-new", created: "2026-09-27T20:38:56Z", label: "seed vs pruning on real tiles",
      summary: { production: { n_tiles: 9, per_band: perBand(12.5, 15.2, 20.19, 17.5), summary: { hole_pct_mean: 17.9 } }, "gate:p20": {} } },
    { id: "e-mean", created: "2026-09-28T00:00:00Z", summary: { mean: {} } },                          // did not score production
  ];
  const catalog = { models: [{ spec: "production", fingerprint: "d9ef" }] };

  it("takes the newest run that scored production, when it scored this production model", () => {
    expect(productionRealHoles(exps, { id: "e-new", fingerprints: { production: "d9ef" } }, catalog)).toEqual({
      state: "current", expId: "e-new", created: "2026-09-27T20:38:56Z", label: "seed vs pruning on real tiles", nTiles: 9,
      worst: { band: "J", pct: 20.19 }, mean: 17.9,
    });
    expect(compareRunPath("e-new")).toBe("/sky/compare?exp=e-new");
  });

  it("never shows an older model's number: 'no real benchmark' for it", () => {
    expect(productionRealHoles(exps, { id: "e-new", fingerprints: { production: "0000" } }, catalog))
      .toMatchObject({ state: "none", why: "earlier", expId: "e-new" });
    expect(productionRealHoles([exps[2]], null, catalog)).toEqual({ state: "none", why: "no-run" });
  });

  it("waits for the run record and the catalogue", () => {
    expect(productionRealHoles(undefined, null, catalog)).toEqual({ state: "loading" });
    expect(productionRealHoles(exps, null, catalog)).toEqual({ state: "loading" });
    expect(productionRealHoles(exps, { id: "e-old" }, catalog)).toEqual({ state: "loading" });
    expect(productionRealHoles(exps, { id: "e-new", fingerprints: { production: "d9ef" } }, null)).toEqual({ state: "loading" });
  });
});
