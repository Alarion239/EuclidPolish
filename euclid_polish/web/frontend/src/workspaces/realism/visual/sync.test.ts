import { describe, expect, it } from "vitest";
import { DEFAULT_GAIN, DEFAULT_KNEE, editOf, shows, viewPatch } from "./sync";

const lupton = { color: "lupton", knee: 100, gain: 1 };

describe("synthetic–real transfer sync", () => {
  it("treats a viewer's first report as its state, not an edit", () => {
    expect(editOf(null, lupton)).toBeNull();
    expect(editOf(lupton, { ...lupton })).toBeNull();
    expect(editOf(lupton, { ...lupton, knee: 100 + 1e-9 })).toBeNull();
  });

  it("reports only the edited fields", () => {
    expect(editOf(lupton, { ...lupton, color: "VIS" })).toEqual({ color: "VIS" });
    expect(editOf(lupton, { ...lupton, knee: 250, gain: 2 })).toEqual({ knee: 250, gain: 2 });
  });

  it("skips viewers that already show the shared transfer (unset = the default)", () => {
    expect(shows(null, { color: "lupton", knee: null, gain: null })).toBe(false);
    expect(shows(lupton, { color: "lupton", knee: null, gain: null })).toBe(true);
    expect(shows(lupton, { color: "lupton", knee: 250, gain: null })).toBe(false);
    expect(shows({ ...lupton, knee: 250 }, { color: "lupton", knee: 250, gain: 1 })).toBe(true);
    // A viewer still on a custom knee does NOT show a reset (default) transfer.
    expect(shows({ ...lupton, knee: 1256, gain: 2 }, { color: "lupton", knee: null, gain: null })).toBe(false);
  });

  it("always patches an explicit knee and gain, the defaults when unset", () => {
    expect([DEFAULT_KNEE, DEFAULT_GAIN]).toEqual([100, 1]);
    expect(viewPatch({ color: "lupton", knee: null, gain: null })).toEqual({ color: "lupton", knee: 100, gain: 1 });
    expect(viewPatch({ color: "J_E", knee: 40, gain: 0.5 })).toEqual({ color: "J_E", knee: 40, gain: 0.5 });
  });
});
