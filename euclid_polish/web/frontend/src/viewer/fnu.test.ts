import { describe, expect, it } from "vitest";
import { fnuBand, fnuFollows, fnuReference, fnuScale, isMjySr } from "./fnu";

// The served constants (viewer_data.color_constants): E_b(1″) = E_b(0.1″) / 0.01.
const color = {
  bands: {
    VIS: { zeropoint_ab_e_total: 0, e_per_mjy_sr_arcsec2: 338688.8023 },
    Y_E: { zeropoint_ab_e_total: 0, e_per_mjy_sr_arcsec2: 23427.514 },
    H_E: { zeropoint_ab_e_total: 0, e_per_mjy_sr_arcsec2: 27398.4636 },
    F200W: { zeropoint_ab_e_total: 0, display_only: true },
  },
  lr_pixscale: 0.1,
};

describe("fnu", () => {
  it("recognises MJy/sr", () => {
    expect(isMjySr("MJy/sr")).toBe(true);
    expect(isMjySr(" mjy / sr ")).toBe(true);
    expect(isMjySr("e-")).toBe(false);
    expect(isMjySr("arb")).toBe(false);
  });

  it("follows only one-image JWST MJy/sr frames, when on and a Euclid group exists", () => {
    const jwst = { transferGroup: "jwst", unit: "MJy/sr" };
    expect(fnuFollows(jwst, true, true)).toBe(true);
    expect(fnuFollows(jwst, false, true)).toBe(false);
    expect(fnuFollows(jwst, true, false)).toBe(false);
    expect(fnuFollows({ ...jwst, unit: "arb" }, true, true)).toBe(false);
    expect(fnuFollows({ ...jwst, directRgb: true }, true, true)).toBe(false);
    expect(fnuFollows({ transferGroup: "euclid", unit: "e-" }, true, true)).toBe(false);
  });

  it("translates through the shown Euclid band, else VIS", () => {
    expect(fnuBand("H_E", color)).toBe("H_E");
    expect(fnuBand("VIS", color)).toBe("VIS");
    expect(fnuBand("lupton", color)).toBe("VIS");
    expect(fnuBand("temp", color)).toBe("VIS");
    expect(fnuBand("F200W", color)).toBe("VIS");
    expect(fnuBand("J_E", color)).toBe("VIS"); // no served constant
  });

  it("takes the coarsest shown Euclid e⁻ frame as the reference pixel", () => {
    const recs = [
      { transferGroup: "euclid", unit: "e-", pixscale: 0.05, displayScale: 1 },
      { transferGroup: "euclid", unit: "e-", pixscale: 0.1, displayScale: 2 },
      { transferGroup: "jwst", unit: "MJy/sr", pixscale: 0.2, displayScale: 3000 },
    ];
    expect(fnuReference(recs, (r) => r.displayScale, color)).toEqual({ pixscale: 0.1, factor: 2 });
    expect(fnuReference([recs[2]], (r) => r.displayScale, color)).toEqual({ pixscale: 0.1, factor: 1 });
  });

  it("gives φ = E_b(1″)·p²·D (VIS at 0.1″ = 3386.9 e⁻ per MJy/sr)", () => {
    expect(fnuScale(color, "VIS", { pixscale: 0.1, factor: 1 })).toBeCloseTo(3386.888, 2);
    expect(fnuScale(color, "H_E", { pixscale: 0.1, factor: 1 })).toBeCloseTo(273.985, 2);
    expect(fnuScale(color, "H_E", { pixscale: 0.05, factor: 4 })).toBeCloseTo(273.985, 2);
    expect(fnuScale(color, "J_E", { pixscale: 0.1, factor: 1 })).toBe(0);
  });
});
