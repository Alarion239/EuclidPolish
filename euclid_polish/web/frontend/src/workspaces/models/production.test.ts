/* What production is, in words (production.ts): a universal model's label,
 * the gate dir's name by what production is, and the Leaderboard's
 * comparison when a universal model is production. */
import { describe, expect, it } from "vitest";
import type { Comparison } from "./model";
import { capitalized, gateName, productionComparison, universalLabel } from "./production";

describe("production in words", () => {
  it("labels a universal model as the catalogue does", () => {
    expect(universalLabel("member_238")).toBe("Universal 238");
    expect(universalLabel("member_09")).toBe("Universal 09");
  });

  it("names the gate dir by what production is", () => {
    expect(gateName(null)).toBe("production gate");
    expect(gateName(undefined, false)).toBe("production gate");
    expect(gateName("member_238")).toBe("former production gate");
    expect(gateName("member_238", false)).toBe("archived gate");
    expect(capitalized("archived gate")).toBe("Archived gate");
  });
});

describe("the Leaderboard comparison with a universal production", () => {
  const cmp: Comparison = { source: "knee", bestNumber: "196", rows: [
    { id: "gate", label: "Production gate", integrated: 61, bands: [56, 65, 62, 61] },
    { id: "mean", label: "Plain mean", integrated: 59, bands: [54, 63, 60, 59] },
    { id: "best", label: "Best member (#196)", integrated: 60, bands: [55, 64, 61, 60] },
  ] };
  const pm = { name: "member_238", spec: "universal:member_238", label: "Universal 238", integrated: [56, 65.5, 63, 64], mean: 62.13,
    scored_at: "2026-10-21T08:00:00Z", stale: false };
  const gate = (available: boolean) => ({ available, n_members: available ? 18 : 0 });

  it("leads with the universal model's own score and renames the gate, which has no real facts", () => {
    const c = productionComparison(cmp, { production_model: pm, production_gate: gate(true) });
    expect(c.rows.map((r) => [r.id, r.label, r.integrated, r.spec])).toEqual([
      ["production", "Production · Universal 238", 62.13, "production"],
      ["gate", "Former production gate", 61, null],
      ["mean", "Plain mean", 59, undefined],
      ["best", "Best member (#196)", 60, undefined],
    ]);
    expect(c.rows[0].bands).toEqual([56, 65.5, 63, 64]);
    expect(c.bestNumber).toBe("196");
    // the gates archived: only the gate's curves remain
    expect(productionComparison(cmp, { production_model: pm, production_gate: gate(false) }).rows[1].label).toBe("Archived gate");
  });

  it("takes the band mean without a mean, has no number before a score, and leaves the gate's comparison alone", () => {
    expect(productionComparison(cmp, { production_model: { ...pm, mean: null }, production_gate: gate(true) }).rows[0].integrated).toBeCloseTo(62.125, 6);
    expect(productionComparison({ ...cmp, rows: [] }, { production_model: { ...pm, integrated: null, mean: null }, production_gate: gate(false) }).rows)
      .toEqual([{ id: "production", label: "Production · Universal 238", integrated: null, bands: [null, null, null, null], spec: "production" }]);
    expect(productionComparison(cmp, { production_model: null, production_gate: gate(true) })).toBe(cmp);
    expect(productionComparison(cmp, null)).toBe(cmp);
  });
});
