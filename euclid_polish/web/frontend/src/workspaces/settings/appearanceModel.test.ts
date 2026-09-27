import { describe, expect, it } from "vitest";
import { DEFAULT_DISPLAY, type DisplaySettings } from "../../state/display";
import { displaySummary } from "./appearanceModel";

const fresh = (): DisplaySettings => JSON.parse(JSON.stringify(DEFAULT_DISPLAY)) as DisplaySettings;

describe("Settings › Appearance image summary", () => {
  it("reads the live Display settings in plain words, none changed at the defaults", () => {
    const rows = displaySummary(fresh());
    expect(rows.map((r) => [r.label, r.value])).toEqual([
      ["Colour", "VIS"], ["Stretch", "Asinh (absolute)"], ["Knee", "100 e⁻, brightness ×1"], ["Colormap", "Gray"],
      ["Residual colormap", "Red–blue (diverging)"], ["NaN colour", "#404040"], ["Invert", "Off"],
      ["Viewers", "Every viewer follows these settings"], ["Mouse wheel", "Zoom when the viewer is focused (or ⌘/Ctrl)"],
    ]);
    expect(rows.some((r) => r.changed)).toBe(false);
  });
  it("marks what differs from the defaults", () => {
    const d = fresh();
    d.stretch = "sqrt";
    d.groups = { ...d.groups, default: { knee: 3.1, gain: 2, black: 0 } };
    d.linked = false;
    const rows = displaySummary(d);
    expect(rows.filter((r) => r.changed).map((r) => r.label)).toEqual(["Stretch", "Knee", "Viewers"]);
    expect(rows.find((r) => r.label === "Knee")?.value).toBe("3.1 e⁻, brightness ×2");
    expect(rows.find((r) => r.label === "Viewers")?.value).toBe("Each viewer keeps its own settings");
  });
});
