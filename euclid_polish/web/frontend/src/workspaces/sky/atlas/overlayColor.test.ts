import { describe, expect, it } from "vitest";
import { DEFAULT_DISPLAY, mergeDisplay } from "../../../state/display";
import { WHITE_KNEES, colorKey, isElectronTier, overlayColor } from "./overlayColor";
import { DEFAULT_OVERLAY_STRETCH } from "./store";

const D = mergeDisplay(DEFAULT_DISPLAY, null);

describe("pixel overlay colour", () => {
  it("follows the Display panel: locked absolute asinh anchors for electron tiers", () => {
    const c = overlayColor("m:production", D, DEFAULT_OVERLAY_STRETCH);
    expect(c).toEqual({ colormap: "grayscale", stretch: "asinh", reversed: false, minCut: 0, maxCut: WHITE_KNEES * 100 });
    expect(overlayColor("lr", D, DEFAULT_OVERLAY_STRETCH).maxCut).toBe(3000);
  });

  it("maps the Euclid transfer group (knee, gain, black) and the colormap / invert", () => {
    const d = mergeDisplay(D, { colormap: "magma", invert: true, groups: { ...D.groups, euclid: { knee: 250, gain: 2, black: 5 } } });
    expect(overlayColor("lr", d, DEFAULT_OVERLAY_STRETCH)).toEqual({ colormap: "magma", stretch: "asinh", reversed: true, minCut: 5, maxCut: 3750 });
    const lin = mergeDisplay(D, { stretch: "sqrt" });
    expect(overlayColor("lr", lin, DEFAULT_OVERLAY_STRETCH).stretch).toBe("sqrt");
  });

  it("uses Aladin's own cuts for JWST and for the auto stretches", () => {
    expect(overlayColor("jwst", D, DEFAULT_OVERLAY_STRETCH)).toEqual({ colormap: "grayscale", stretch: "asinh", reversed: false });
    const auto = mergeDisplay(D, { stretch: "zscale" });
    expect(overlayColor("lr", auto, DEFAULT_OVERLAY_STRETCH)).toEqual({ colormap: "grayscale", stretch: "linear", reversed: false });
  });

  it("own settings when not following", () => {
    const own = { follow: false, colormap: "viridis", stretch: "log", minCut: 1, maxCut: 50 };
    expect(overlayColor("lr", D, own)).toEqual({ colormap: "viridis", stretch: "log", reversed: false, minCut: 1, maxCut: 50 });
    expect(overlayColor("jwst", D, { ...own, minCut: null, maxCut: null })).toEqual({ colormap: "viridis", stretch: "log", reversed: false });
  });

  it("classifies tiers and keys colours", () => {
    expect(isElectronTier("lr")).toBe(true);
    expect(isElectronTier("m:gate:b")).toBe(true);
    expect(isElectronTier("jwst")).toBe(false);
    expect(colorKey({ colormap: "a" })).not.toBe(colorKey({ colormap: "b" }));
  });
});
