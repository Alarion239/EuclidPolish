/* The multipoint archive provenance strings (was test/archiveFields.test.ts;
   its SyntheticReal/fasrc source-text checks are behaviour tests now: see
   synthetic.test.tsx › fields). */
import { describe, expect, it } from "vitest";
import {
  archiveFieldBreakdown, archiveOverview, archiveSampleProvenance, shortArchiveFingerprint,
  type ArchiveAvailability,
} from "./archiveFields";

const READY: ArchiveAvailability = {
  available: true, valid: true, ready: true, complete: true, current: true, reasons: [],
  sample_count: 220, planned_sample_count: 220, parent_count: 44,
  fields: { "EDF-S": 45, "EDF-N": 80, "EDF-F": 95 },
  comparison_sample_count: 176, comparison_fields: { "EDF-S": 36, "EDF-N": 64, "EDF-F": 76 },
  comparison_excluded_positions: ["center"], bands: ["VIS", "Y_E", "J_E", "H_E"], tile_size: 256,
  manifest_fingerprint: "b".repeat(64), collection_fingerprint: "c".repeat(64), source_release: "Q1_R1",
  source_plan_fingerprint: "a".repeat(64), source_manifest_sha256: "d".repeat(64),
};

describe("archive fields", () => {
  it("summarizes independent archive pointings without calling tiles fields", () => {
    expect(archiveOverview(READY)).toBe(
      "44 independent parent pointings · 176 four-band samples (44 star-avoiding centre tiles left out) · Q1_R1");
    expect(archiveFieldBreakdown(READY)).toBe("EDF-F 76 · EDF-N 64 · EDF-S 36");
    const legacy = { ...READY, comparison_sample_count: undefined, comparison_fields: undefined };
    expect(archiveOverview(legacy)).toBe("44 independent parent pointings · 220 four-band samples · Q1_R1");
    expect(archiveFieldBreakdown(legacy)).toBe("EDF-F 95 · EDF-N 80 · EDF-S 45");
  });

  it("surfaces missing/stale reasons and exact per-sample provenance", () => {
    expect(archiveOverview({ ...READY, ready: false, current: false, reasons: ["source plan changed"] }))
      .toBe("source plan changed");
    expect(archiveSampleProvenance({
      label: "sample", tiers: ["lr"], sample_id: 17, source_sample_id: 3, parent_id: "parent-3",
      field: "EDF-N", ra: 12, dec: 65, position_name: "northeast",
    }, 176)).toBe("archive sample 18 · source pointing 4 · EDF-N · northeast");
    expect(archiveSampleProvenance(undefined, 176)).toBe("176 samples");
    expect(shortArchiveFingerprint("a".repeat(64))).toBe("aaaaaaaaaaaa…");
    expect(shortArchiveFingerprint(null)).toBe("unknown");
  });
});
