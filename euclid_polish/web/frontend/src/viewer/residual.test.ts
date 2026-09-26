import { describe, expect, it } from "vitest";
import { commonGrid, computeResidual, parseResidualKey, residualKey, residualLabel, residualMismatch, upsample } from "./residual";

const rec = (h: number, w: number, c: number, fill: (y: number, x: number, k: number) => number) => {
  const data = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let k = 0; k < c; k++) data[(y * w + x) * c + k] = fill(y, x, k);
  return { data, h, w, c, bands: ["VIS", "Y_E", "J_E", "H_E"].slice(0, c), unit: "e-" };
};

describe("residual tier keys", () => {
  it("round-trip and label", () => {
    expect(residualKey("diff", "sr", "hr")).toBe("res:diff:sr:hr");
    expect(parseResidualKey("res:chi:sr:mean")).toEqual({ op: "chi", a: "sr", b: "mean" });
    expect(parseResidualKey("res:xx:sr:hr")).toBeNull();
    expect(parseResidualKey("sr")).toBeNull();
    expect(residualLabel("ratio", "SR", "HR")).toBe("log₂(SR / HR)");
  });
});

describe("grids", () => {
  it("the common grid is the finer one when the factor is an integer", () => {
    expect(commonGrid({ h: 255, w: 255 }, { h: 510, w: 510 })).toEqual({ h: 510, w: 510, fa: 2, fb: 1 });
    expect(commonGrid({ h: 510, w: 510 }, { h: 510, w: 510 })).toEqual({ h: 510, w: 510, fa: 1, fb: 1 });
    expect(commonGrid({ h: 256, w: 256 }, { h: 850, w: 850 })).toBeNull();
  });
  it("upsampling replicates each pixel and conserves flux", () => {
    const lr = rec(1, 2, 1, (_y, x) => (x + 1) * 4);
    const up = upsample(lr, 2);
    expect(up.h).toBe(2);
    expect(Array.from(up.data)).toEqual([1, 1, 2, 2, 1, 1, 2, 2]);
  });
});

describe("computeResidual", () => {
  const A = rec(2, 2, 2, (y, x, k) => 10 + x + y + k);
  const B = rec(2, 2, 2, () => 10);
  it("A − B per band", () => {
    const r = computeResidual("diff", A, B)!;
    expect(r.c).toBe(2);
    expect(Array.from(r.data)).toEqual([0, 1, 1, 2, 1, 2, 2, 3]);
    expect(r.unit).toBe("e-");
  });
  it("log₂ A / B (NaN where either side is not positive)", () => {
    const Z = rec(2, 2, 2, (y) => (y ? 20 : -1));
    const r = computeResidual("ratio", Z, B)!;
    expect(Number.isNaN(r.data[0])).toBe(true);
    expect(r.data[4]).toBeCloseTo(1, 12);
    expect(r.unit).toBe("log₂");
  });
  it("(A − B)/σ with a σ tier, else the robust σ of the difference", () => {
    const S = rec(2, 2, 2, () => 0.5);
    const r = computeResidual("chi", A, B, S)!;
    expect(Array.from(r.data)).toEqual([0, 2, 2, 4, 2, 4, 4, 6]);
    expect(r.sigmaSource).toBe("tier");
    const m = computeResidual("chi", A, B)!;
    expect(m.sigmaSource).toBe("mad");
    expect(Number.isFinite(m.data[0])).toBe(true);
  });
  it("pairs channels by band name, not by position", () => {
    const four = rec(1, 1, 4, (_y, _x, k) => 100 * (k + 1));            // VIS 100, Y 200, J 300, H 400
    const two = { ...rec(1, 1, 2, (_y, _x, k) => (k ? 1 : 3)), bands: ["J_E", "VIS"] };   // J 3, VIS 1
    const r = computeResidual("diff", four, two)!;
    expect(r.bands).toEqual(["VIS", "J_E"]);
    expect(Array.from(r.data)).toEqual([100 - 1, 300 - 3]);
    // the σ tier is matched by band name too
    const S = { ...rec(1, 1, 2, (_y, _x, k) => (k ? 10 : 20)), bands: ["J_E", "VIS"] };   // σ_J 20, σ_VIS 10
    const chi = computeResidual("chi", four, two, S)!;
    expect(chi.sigmaSource).toBe("tier");
    expect(chi.data[0]).toBeCloseTo(99 / 10, 5);
    expect(chi.data[1]).toBeCloseTo(297 / 20, 5);
    // unnamed channels fall back to positional pairing
    const anon = { ...rec(1, 1, 2, () => 1), bands: [] };
    expect(computeResidual("diff", four, anon)!.bands).toEqual(["VIS", "Y_E"]);
  });
  it("refuses tiers without a common band or in different units", () => {
    const vis = rec(2, 2, 4, () => 1);
    const jwst = { ...rec(2, 2, 1, () => 0.02), bands: ["F200W"], unit: "MJy/sr" };
    expect(residualMismatch(vis, jwst)).toBe("are in different units (e⁻ vs MJy/sr)");
    expect(computeResidual("diff", vis, jwst)).toBeNull();
    const f200 = { ...rec(2, 2, 1, () => 0.02), bands: ["F200W"] };
    expect(residualMismatch(vis, f200)).toBe("have no band in common (VIS, Y_E, J_E, H_E vs F200W)");
    expect(computeResidual("ratio", vis, f200)).toBeNull();
    // spellings of the electron unit agree; an unknown unit is no conflict
    expect(residualMismatch({ ...vis, unit: "e⁻" }, vis)).toBeNull();
    expect(residualMismatch({ ...vis, unit: "" }, vis)).toBeNull();
    expect(residualMismatch(rec(3, 3, 1, () => 1), rec(2, 2, 1, () => 1))).toBe("are not on matching grids");
  });
  it("mixed grids are resampled onto the finer grid; incompatible grids give null", () => {
    const LR = rec(1, 1, 1, () => 8);
    const SR = rec(2, 2, 1, () => 3);
    const r = computeResidual("diff", SR, LR)!;
    expect(r.h).toBe(2);
    expect(Array.from(r.data)).toEqual([1, 1, 1, 1]);   // 3 − 8/4
    expect(computeResidual("diff", rec(3, 3, 1, () => 1), rec(2, 2, 1, () => 1))).toBeNull();
  });
});
