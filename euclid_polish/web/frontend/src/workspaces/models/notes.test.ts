/* The Universal tab's notebook note (notes.ts): the models table best first,
 * with its definitions. */
import { describe, expect, it } from "vitest";
import { universalNote } from "./notes";
import { payload, universalProduction } from "./testing/universalFixture";
import { shapeRows } from "./universal";

describe("universal note", () => {
  it("tables the models best first, with the definitions", () => {
    const p = payload();
    const md = universalNote(shapeRows(p, { metric: "mean" }), { nFields: 100, integration: p.integration });
    expect(md).toContain("**Universal models** — 5 models · ∫PSNR over 0.1–10k e⁻, mean of the bands, on 100 test fields");
    const lines = md.split("\n");
    const first = lines.findIndex((l) => l.startsWith("| #"));
    expect(lines[first]).toBe("| #211 | 16 | 1.0M | 4 × 8 | learned | plain | A100 | 12.7 | 61.60 | +1.10 |");
    expect(md).toContain("| #215 | 16 | 200k | 4 × 8 | learned | XLA + bf16 | A100 MIG slice | 1.90 | 60.80 (stale) | +0.30 (stale) |");
    expect(md).toContain("Training time: Σ of the training log's windows");
    expect(md).not.toMatch(/combiner|gate/i);
  });

  it("names the production model in the header and its row", () => {
    const p = universalProduction();
    const md = universalNote(shapeRows(p, { metric: "mean" }), { nFields: 100, integration: p.integration, production: p.production_model });
    expect(md).toContain("**Universal models** — 5 models · production: Universal 210 · ∫PSNR over 0.1–10k e⁻");
    expect(md).toContain("| #211 | 16 | 1.0M | 4 × 8 | learned | plain | A100 | 12.7 | 61.60 | +0.60 |");
    expect(md).toContain("| #210 (production) | 16 |");
  });
});
