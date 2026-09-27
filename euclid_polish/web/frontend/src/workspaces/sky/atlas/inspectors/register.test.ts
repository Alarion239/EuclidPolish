import { describe, expect, it } from "vitest";
import { useInspectorRegistry } from "../../../../app/inspector";
import { sourceTitle } from "./register";

describe("atlas inspector registration", () => {
  it("importing the module registers `tile` and `source` (so the shell can import it eagerly)", () => {
    const kinds = useInspectorRegistry.getState().kinds;
    expect(kinds.tile?.kind).toBe("tile");
    expect(kinds.source?.kind).toBe("source");
    const title = kinds.tile?.title;
    expect(typeof title === "function" ? title("nexus/12") : title).toBe("Tile nexus/f200w-0012");
  });

  it("titles source cards by layer and id", () => {
    expect(sourceTitle("at/268.4,65.2")).toBe("Sky point 268.4,65.2");
    expect(sourceTitle("lens-candidates/L1")).toBe("lens-candidates · L1");
  });
});
