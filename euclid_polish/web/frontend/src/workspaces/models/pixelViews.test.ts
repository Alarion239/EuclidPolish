/* The native pixel-types data helpers: the file names, the guards against a
 * non-view body, stamp decoding and colouring, quantile series and the
 * σ texts. */
import { describe, expect, it } from "vitest";
import {
  asAnalysisView, asCompareView, asPairs, asStamps, CIVIDIS, decodeStamp, lut, MAGMA, normText, pairsFile, quantSeries, sigmaText,
  stampRgba, stampsFile, viewFile,
} from "./pixelViews";

describe("pixel views", () => {
  it("names the files and refuses bodies that are not views", () => {
    expect(viewFile("synthetic")).toBe("view_synthetic.json");
    expect(stampsFile("real", "H_E")).toBe("stamps_real_H_E.json");
    expect(pairsFile("synthetic", "VIS")).toBe("pairs_synthetic_VIS.json");
    expect(asAnalysisView({ ok: false, error: "no such pixel-types file" })).toBeNull();
    expect(asAnalysisView({ bands: {}, centres_log: [] })).not.toBeNull();
    expect(asStamps({ side: 41, groups: {}, types: [] })).not.toBeNull();
    expect(asStamps(null)).toBeNull();
    expect(asCompareView({ curves: {}, models: ["a", "b"] })).not.toBeNull();
    expect(asPairs({ side: 41, rows: [] })).not.toBeNull();
    expect(asPairs({ rows: [] })).toBeNull();
  });

  it("decodes a stamp and colours it through a 256-entry table", () => {
    const bytes = decodeStamp(btoa(String.fromCharCode(0, 128, 255)));
    expect([...bytes]).toEqual([0, 128, 255]);
    const magma = lut(MAGMA);
    expect([...magma.slice(0, 3)]).toEqual([0, 0, 4]);                 // #000004
    expect([...magma.slice(255 * 3)]).toEqual([0xfc, 0xfd, 0xbf]);     // #fcfdbf
    const rgba = stampRgba(bytes, lut(CIVIDIS));
    expect(rgba.length).toBe(12);
    expect([...rgba.slice(0, 4)]).toEqual([0x00, 0x22, 0x4e, 255]);
    expect(rgba[11]).toBe(255);
  });

  it("turns quantile triples into a line with its band, and states σ values", () => {
    expect(quantSeries([[0, 1, 2], null, [1, null, 3]])).toEqual({ y: [1, null, null], low: [0, null, 1], high: [2, null, 3] });
    expect(sigmaText(0)).toBe("1.0σ");
    expect(sigmaText(-2)).toBe("0.01σ");
    expect(sigmaText(2.5)).toBe("316σ");
    expect(sigmaText(null)).toBe("—");
    expect(normText([-0.3, 42.5, 0.3])).toBe("-0.30 … 42.5σ, asinh");
    expect(normText(undefined)).toBe("");
  });
});
