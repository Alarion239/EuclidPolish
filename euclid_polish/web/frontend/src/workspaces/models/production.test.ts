/* What production is, in words (production.ts): a universal model's label,
 * the gate dir's name by what production is, and the Leaderboard's
 * comparison when a universal model is production. */
import { describe, expect, it } from "vitest";

import { universalLabel } from "./production";

describe("production in words", () => {
  it("labels a universal model as the catalogue does", () => {
    expect(universalLabel("member_238")).toBe("Universal 238");
    expect(universalLabel("member_09")).toBe("Universal 09");
  });
});
