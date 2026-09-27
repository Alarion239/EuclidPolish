import { describe, expect, it } from "vitest";
import { serverCodeText } from "./versionText";

describe("server code state", () => {
  it("says what changed and what to do, never 'behind'", () => {
    expect(serverCodeText({ behind: true }, false)).toEqual({ tone: "warn", badge: "code changed", title: "Backend code changed — restart the server to load it" });
    expect(serverCodeText({ behind: true }, true).title).toBe("Backend code changed — restart the server to load it");
    expect(serverCodeText({ behind: false }, true)).toEqual({ tone: "warn", badge: "new build", title: "The console build changed — reload" });
    expect(serverCodeText({ behind: false }, false)).toEqual({ tone: "good", badge: "current code", title: null });
    for (const s of [serverCodeText({ behind: true }, false), serverCodeText({ behind: false }, false)]) {
      expect(`${s.badge} ${s.title ?? ""}`).not.toMatch(/behind|HEAD/);
    }
  });
});
