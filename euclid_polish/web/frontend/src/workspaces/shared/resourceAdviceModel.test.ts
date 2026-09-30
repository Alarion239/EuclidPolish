import { describe, expect, it } from "vitest";
import {
  adviceHeadline, adviceRows, appliedResources, confidenceText, fieldLabel, levelText, rateText, usageHref,
  type Recommendation,
} from "./resourceAdviceModel";

const REC: Recommendation = {
  ok: true, step_id: "synthetic_generate", available: true, confidence: "high",
  resources: { n_cpus: "20", n_gpus: "0", memory: "36G", time_limit: "3:15:00" },
  current: { n_cpus: "20", n_gpus: "0", memory: "32G", time_limit: "2:00:00" },
  changes: [
    { field: "time_limit", current: "2:00:00", recommended: "3:15:00", reason: "p90 2.4 s per image × 4,000 images × 1.2" },
    { field: "memory", current: "32G", recommended: "36G", reason: "p90 peak 29.8 GB of 32 GB over 8 runs; 1 OOM at 30 GB" },
  ],
  basis: { level: "similar", level_label: "same splits and image size", n_runs: 8, jobids: ["1", "2"], units: 4000,
    units_label: "images", rate_s_per_unit: 2.4 },
  notes: ["n1"], warnings: [],
};

describe("words", () => {
  it("labels fields, per array task when asked", () => {
    expect(fieldLabel("memory")).toBe("Memory");
    expect(fieldLabel("n_cpus", "member")).toBe("CPUs / member");
    expect(fieldLabel("odd_knob")).toBe("odd knob");
  });
  it("names the level from the backend, else from the level id", () => {
    expect(levelText(REC.basis)).toBe("same splits and image size");
    expect(levelText({ level: "exact", n_runs: 3 })).toBe("same plan and CPU count");
    expect(levelText({ level: "step", level_label: "", n_runs: 3 })).toBe("every run of the step");
    expect(levelText(null)).toBe("");
  });
  it("heads the callout with the run count, the level and the confidence", () => {
    expect(adviceHeadline(REC)).toBe("Recommended from 8 past runs (same splits and image size) · high confidence");
    expect(adviceHeadline({ ...REC, basis: { level: null, n_runs: 1 }, confidence: null })).toBe("Recommended from 1 past run");
    expect(confidenceText(" Low ")).toBe("low confidence");
    expect(adviceHeadline({ ...REC, basis: { level: "similar", level_label: "same settings (batch 4 · crop 256)", n_runs: 14 } }))
      .toBe("Recommended from 14 past runs · same settings (batch 4 · crop 256) · high confidence");
  });
  it("links Open usage to Runs › Resources on the step", () => {
    expect(usageHref("ensemble_train")).toBe("/runs/resources?step=ensemble_train");
  });
});

describe("changes", () => {
  it("lists the applicable changes in form order", () => {
    const rows = adviceRows(REC);
    expect(rows.map((r) => [r.field, r.current, r.recommended])).toEqual([["memory", "32G", "36G"], ["time_limit", "2:00:00", "3:15:00"]]);
    expect(rows[0].label).toBe("Memory");
    expect(rows[0].reason).toContain("1 OOM");
  });
  it("offers only the fields the host edits", () => {
    expect(adviceRows(REC, ["n_cpus", "memory"]).map((r) => r.field)).toEqual(["memory"]);
    expect(adviceRows(REC, ["n_cpus"])).toEqual([]);
  });
  it("offers nothing without history", () => {
    expect(adviceRows({ ...REC, available: false })).toEqual([]);
    expect(adviceRows(null)).toEqual([]);
  });
  it("applies the recommended values over the current resources", () => {
    const cur = { n_cpus: "16", n_gpus: "0", memory: "32G", time_limit: "2:00:00" };
    expect(appliedResources(cur, REC)).toEqual({ n_cpus: "16", n_gpus: "0", memory: "36G", time_limit: "3:15:00" });
    expect(appliedResources({ memory: "32G" }, REC, ["memory"])).toEqual({ n_cpus: "", n_gpus: "", memory: "36G", time_limit: "" });
  });
});

describe("rate", () => {
  it("scales the unit count until one lot takes a second", () => {
    expect(rateText(2.4, "images")).toBe("2.4s per image");
    expect(rateText(0.14, "steps")).toBe("2m 20s per 1,000 steps");
    expect(rateText(null, "steps")).toBe("");
    expect(rateText(0, "steps")).toBe("");
  });
});
