import { describe, expect, it } from "vitest";
import { moreTooltip } from "./Bar";

describe("More menu tooltip", () => {
  it("names the moved groups in the menu's own order, with their menu labels", () => {
    expect(moreTooltip(["export", "layout", "tools", "compare", "zoom"]))
      .toBe("More controls: compare, tools, zoom, arrange, export");
    expect(moreTooltip(["layout", "export"])).toBe("More controls: arrange, export");
    expect(moreTooltip(["export"])).toBe("More controls: export");
  });
});
