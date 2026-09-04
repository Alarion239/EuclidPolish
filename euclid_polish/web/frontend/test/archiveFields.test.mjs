import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  archiveFieldBreakdown,
  archiveOverview,
  archiveSampleProvenance,
  shortArchiveFingerprint,
} from "../src/pages/archiveFields.ts";

const ready = {
  available: true,
  valid: true,
  ready: true,
  complete: true,
  current: true,
  reasons: [],
  sample_count: 220,
  planned_sample_count: 220,
  parent_count: 44,
  fields: { "EDF-S": 45, "EDF-N": 80, "EDF-F": 95 },
  bands: ["VIS", "Y_E", "J_E", "H_E"],
  tile_size: 256,
  manifest_fingerprint: "b".repeat(64),
  collection_fingerprint: "c".repeat(64),
  source_release: "Q1_R1",
  source_plan_fingerprint: "a".repeat(64),
  source_manifest_sha256: "d".repeat(64),
};

test("summarizes independent archive pointings without calling tiles fields", () => {
  assert.equal(
    archiveOverview(ready),
    "44 independent parent pointings · 220 four-band samples · Q1_R1",
  );
  assert.equal(archiveFieldBreakdown(ready), "EDF-F 95 · EDF-N 80 · EDF-S 45");
});

test("surfaces missing/stale reasons and exact per-sample provenance", () => {
  assert.equal(
    archiveOverview({ ...ready, ready: false, current: false, reasons: ["source plan changed"] }),
    "source plan changed",
  );
  assert.equal(
    archiveSampleProvenance({
      label: "sample",
      tiers: ["lr"],
      sample_id: 17,
      source_sample_id: 3,
      parent_id: "parent-3",
      field: "EDF-N",
      ra: 12,
      dec: 65,
      position_name: "northeast",
    }, 17, 220),
    "sample 18 / 220 · source pointing 4 · EDF-N · northeast",
  );
  assert.equal(shortArchiveFingerprint("a".repeat(64)), "aaaaaaaaaaaa…");
});

test("Synthetic–Real uses only the multipoint collection and keeps legacy inference explicit", () => {
  const comparison = readFileSync(
    new URL("../src/pages/SyntheticReal.tsx", import.meta.url),
    "utf8",
  );
  assert.match(comparison, /\/viewer\/meta\/archive-fields/);
  assert.match(comparison, /collection="archive-fields"/);
  assert.match(comparison, /stepId="archive_field_sample"/);
  assert.match(comparison, /archiveSync\.run\("\/api\/archive-fields\/sync"/);
  assert.doesNotMatch(comparison, /\/api\/inference\/field\.json/);
  assert.doesNotMatch(comparison, /collection="real-field"/);

  const inference = readFileSync(
    new URL("../src/pages/Inference.tsx", import.meta.url),
    "utf8",
  );
  assert.match(inference, /collection="real-field"/);
  assert.match(inference, /Ad-hoc single-pointing workspace/);
});
