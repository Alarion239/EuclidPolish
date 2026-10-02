/* The pure shaping behind the Combiner, Diagnostics and Images tabs (plan
 * "Console phases 2–6", Team M): held-out curves on one loss scale with
 * unique styles, the gate share per member (production from members.json, a
 * variant from its compare report), the per-set stamp caption, the spectrum's
 * SR-scale focus and non-colliding guides, the spread section's coverage
 * table, answer and z-pdf domain, the coherence labels, and SR's Δm vs LR
 * from the viewer's cubes. */
import { describe, expect, it } from "vitest";
import type { MemberRow, Variant } from "./api";
import {
  bandIndex, changedConfigKnobs, coherenceLabel, coverageRows, cubeDeltaMag, heldOutCurves, pdfYDomain, planeSum, productionShares, reportShares,
  share, spectrumDomain, spectrumGuides, spreadVerdict, stampCaption, statusChecks, type StampSet,
} from "./model";
import { kneeColor, kneeOrderOf } from "./common";
import { categorical, viridis } from "../../colors";

const LOSS_A = "per-field relative asinh MSE; at every loss knee";
const gate = (name: string, patch: Partial<Variant> = {}): Variant => ({
  name, kind: "gate", spec: `gate:${name}`, production: false, backup: false, member_labels: [], reads: [], n_members: 3, n_reads: 3,
  pruned: false, use_lr: false, membership: { current: true, missing: [], extra: [] }, applies_to_test_cubes: true,
  fit: { loss: LOSS_A }, history: [{ step: 0, loss: 1, vis_psnr: 58, integrated_psnr: [55, 64, 61, 60] }, { step: 250, loss: 0.7, vis_psnr: 59 }],
  ...patch,
});

describe("combiner held-out curves", () => {
  const prod = gate("spatial_gate_combiner", { production: true });
  const other = gate("spatial_gate_v1", { fit: { loss: "band-weighted asinh squared error" } });
  const plain = gate("spatial_gate_p20");
  const empty = gate("spatial_gate_new", { history: [] });

  it("draws only the fits whose loss is on production's scale, and names the others", () => {
    const c = heldOutCurves([plain, prod, other, empty], "loss");
    expect(c.curves.map((x) => [x.name, x.production])).toEqual([["spatial_gate_combiner", true], ["spatial_gate_p20", false]]);
    expect(c.curves[1]).toMatchObject({ x: [0, 250], y: [1, 0.7] });
    expect(c.offScale).toEqual(["spatial_gate_v1"]);
  });

  it("compares every fit on PSNR (one scale), production first", () => {
    const c = heldOutCurves([plain, prod, other], "vis");
    expect(c.curves.map((x) => x.name)).toEqual(["spatial_gate_combiner", "spatial_gate_p20", "spatial_gate_v1"]);
    expect(c.offScale).toEqual([]);
    const int = heldOutCurves([prod], "int");
    expect(int.curves[0]).toMatchObject({ x: [0], y: [60] });           // the mean of the four bands; a step without it is dropped
  });

  it("gives every curve a unique style: 8 colours, then the same colours dashed", () => {
    const many = Array.from({ length: 11 }, (_, i) => gate(`spatial_gate_v${i}`));
    const c = heldOutCurves([prod, ...many], "vis");
    const styles = c.curves.map((x) => `${x.slot}:${x.dash}`);
    expect(new Set(styles).size).toBe(styles.length);
    expect(c.curves[0]).toMatchObject({ slot: -1, dash: false });        // production: the combiner colour
    expect(c.curves[9]).toMatchObject({ slot: 0, dash: true });
    // a palette without the combiner colour has 7 slots: the 8th variant is dashed
    const seven = heldOutCurves([prod, ...many], "vis", 7);
    expect(seven.curves[8]).toMatchObject({ slot: 0, dash: true });
  });
});

const row = (n: number, patch: Partial<MemberRow> = {}): MemberRow => ({
  name: `member_${n}`, label: `${n}·psnr`, origin: null, loss: "l2",
  status: "complete", timeout: false, job: null, ...patch,
} as MemberRow);

describe("gate share per member", () => {
  it("ranks production's members by their share (the peak when recorded), unread and unknown last", () => {
    const rows = productionShares([
      row(1, { gate_usage: { VIS: 0.01, Y_E: 0.01, J_E: 0.01, H_E: 0.01 }, used_by_gate: false }),
      row(2, { gate_usage: { VIS: 0.2, Y_E: 0.2, J_E: 0.2, H_E: 0.2 }, used_by_gate: true }),
      row(3, {}),
      row(4, { gate_usage: { VIS: 0.0001, Y_E: 0, J_E: 0, H_E: 0 }, gate_usage_peak: { value: 0.4, band: "VIS", bin: "core" }, used_by_gate: true }),
    ]);
    expect(rows.map((r) => [r.num, r.share, r.read])).toEqual([["04", 0.4, true], ["02", 0.2, true], ["01", 0.01, false], ["03", null, null]]);
    expect(rows[1].text).toBe("20%");
  });

  it("reads a variant's shares from its compare report (members × bands), in either orientation", () => {
    const usage = { labels: ["196·psnr", "178·psnr"], all_pixels: [[0.6, 0.5, 0.7, 0.6], [0.4, 0.5, 0.3, 0.4]], source_pixels: [] };
    expect(reportShares(usage)?.map((r) => [r.name, r.num, Number(r.share?.toFixed(3))])).toEqual([
      ["member_196", "196", 0.6], ["member_178", "178", 0.4]]);
    const transposed = { labels: ["1", "2"], all_pixels: [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]], source_pixels: [] };
    expect(reportShares(transposed)?.map((r) => Number(r.share?.toFixed(2)))).toEqual([0.75, 0.25]);
    expect(reportShares(undefined)).toBeNull();
    expect(reportShares({ labels: ["1"], all_pixels: [], source_pixels: [] })).toBeNull();
  });
});

describe("stamp caption", () => {
  const set: StampSet = { grade: "syn-lens", label: "Syn lens", n: 30, points: [], medianLr: 31.2, medianSr: 33.4, gain: 2.2, improved: 28, stale: 0, madeBy: 30 };
  it("gives the set's PSNR vs HR, LR → SR, the median gain and how many SR brought closer", () => {
    expect(stampCaption({ ...set, points: Array.from({ length: 30 }, (_, i) => ({ id: String(i), x: 1, y: 2, flux: null })) }))
      .toBe("Median PSNR vs HR: LR 31.20 → SR 33.40 dB, a gain of +2.20 dB · SR is closer to the truth on 28 of 30");
    expect(stampCaption({ ...set, improved: 2, points: [{ id: "a", x: 1, y: 2, flux: null }, { id: "b", x: 1, y: 2, flux: null }] }))
      .toMatch(/on every stamp$/);
    expect(stampCaption({ ...set, n: 4, points: [], medianLr: null, medianSr: null, gain: null, improved: 0 }))
      .toBe("No PSNR vs HR recorded");
  });
});

describe("spectrum and transfer", () => {
  const theta = [7.6, 3.1, 1.2, 0.49, 0.2, 0.1, 0.055, null];
  it("focuses x on the SR scales (θ < 0.5″) unless every scale is asked for", () => {
    expect(spectrumDomain(theta, 0.05, true)).toEqual([0.05, 0.5]);
    expect(spectrumDomain(theta, 0.05, false)).toEqual([0.05, 7.6]);
    expect(spectrumDomain([0.3, 0.2], undefined, true)).toEqual([0.05, 0.3]);
  });

  it("puts the LR-pixel and VIS-FWHM labels on opposite sides so they never overprint", () => {
    const g = spectrumGuides({ lr_scale: 0.1, vis_fwhm: 0.16 }, [0.05, 0.5]);
    expect(g.map((x) => [x.kind, x.v, x.side])).toEqual([["lr", 0.1, "before"], ["fwhm", 0.16, "after"]]);
    expect(spectrumGuides({ lr_scale: 0.1, vis_fwhm: 0.16 }, [0.12, 0.5]).map((x) => x.kind)).toEqual(["fwhm"]);
    expect(spectrumGuides({}, [0.05, 7]).map((x) => [x.kind, x.v])).toEqual([["lr", 0.1], ["fwhm", 0.16]]);
  });
});

describe("spread", () => {
  it("answers whether σ is an error bar from the median RMSE/σ ratio", () => {
    expect(spreadVerdict(9.7)).toEqual({ text: "Cross-member σ is not an error bar", warn: true });
    expect(spreadVerdict(1)).toEqual({ text: "Cross-member σ tracks the error", warn: false });
    expect(spreadVerdict(0.5)).toEqual({ text: "Cross-member σ over-states the error", warn: true });
    expect(spreadVerdict(null)).toBeNull();
  });

  it("tabulates |z| coverage observed vs Gaussian", () => {
    expect(coverageRows({ cover1: 0.889, cover2: 0.953, cover3: 0.977 })).toEqual([
      { k: 1, observed: 0.889, gaussian: 0.683 }, { k: 2, observed: 0.953, gaussian: 0.954 }, { k: 3, observed: 0.977, gaussian: 0.997 }]);
  });

  it("sizes the z-pdf to its tallest curve (the measured peak reached 0.89, clipped at 0.6)", () => {
    expect(pdfYDomain([0.1, 0.8875, null], [0.399])).toEqual([0, expect.closeTo(0.976, 3)]);
    expect(pdfYDomain([null], [])).toEqual([0, 0.45]);
  });
});

describe("coherence labels", () => {
  it("names each row the way the other tabs do", () => {
    expect(coherenceLabel({ id: "ensemble_mean", label: "ensemble mean" })).toBe("plain mean");
    expect(coherenceLabel({ id: "spatial_gate_combiner", label: "gate" })).toBe("production gate");
    expect(coherenceLabel({ id: "lr_baseline", label: "LR" })).toBe("LR (bicubic)");
    expect(coherenceLabel({ id: "member_3", label: "196·psnr" })).toBe("#196");
    expect(coherenceLabel({ id: "model_agreement", label: "agreement" })).toBe("member agreement");
  });
});

describe("Δm vs LR from the viewer's cubes", () => {
  const cube = (c: number, planes: number[][]) => {
    const n = planes[0].length;
    const data = new Float32Array(n * c);
    planes.forEach((p, b) => p.forEach((v, i) => { data[i * c + b] = v; }));
    return { data, c, bands: ["VIS", "Y_E", "J_E", "H_E"].slice(0, c) };
  };
  it("sums one band plane, skipping non-finite pixels", () => {
    const { data } = cube(2, [[1, 2, NaN], [10, 20, 30]]);
    expect(planeSum(data, 2, 0)).toBe(3);
    expect(planeSum(data, 2, 1)).toBe(60);
  });

  it("follows the viewer's band; a colour composite reads VIS", () => {
    expect(bandIndex(["VIS", "Y_E", "J_E", "H_E"], "J_E")).toBe(2);
    expect(bandIndex(["VIS", "Y_E"], "lupton")).toBe(0);
    expect(bandIndex(undefined, "Y_E")).toBe(0);
  });

  it("states SR's flux change against LR in the shown band", () => {
    const lr = cube(2, [[50, 50], [10, 10]]);
    const sr = cube(2, [[45, 45], [10, 10]]);
    expect(cubeDeltaMag(lr, sr, "VIS")).toEqual({ band: "VIS", text: "Δm +0.11 (flux ×0.90)", warn: true });
    expect(cubeDeltaMag(lr, sr, "Y_E")).toEqual({ band: "Y", text: "Δm 0.00 (flux ×1.00)", warn: false });
    expect(cubeDeltaMag(lr, cube(2, [[0, 0], [0, 0]]), "VIS")).toBeNull();
  });
});

describe("one rule for the numbers the Models tabs repeat", () => {
  it("formats every gate share to two significant figures", () => {
    expect([0.48, 0.046, 0.4, 0.47, 0.0048, 0.0002, 0, null].map(share)).toEqual(["48%", "4.6%", "40%", "47%", "0.48%", "<0.1%", "0%", "—"]);
  });

  it("ends a long member list with 'and N more', not an ellipsis before the period", () => {
    const members = Array.from({ length: 11 }, (_, i) => ({ name: `member_${130 + i}`, psnr: null }) as unknown as MemberRow);
    const detail = statusChecks([], members)[0].detail;
    expect(detail).toBe("11 members have no test PSNR yet: 130, 131, 132, 133, 134, 135, 136, 137 and 3 more.");
    expect(detail).not.toContain("…");
  });

  it("colours a training knee the same on every tab: ordered viridis, multi-knee in fixed hues", () => {
    const rows = [{ asinh_knee: 1000 }, { asinh_knee: 10 }, { asinh_knee: 100 }, { asinh_knees: [0.1, 1, 10], output_knee: 10 }];
    const order = kneeOrderOf(rows);
    expect(order).toEqual([10, 100, 1000]);
    expect(kneeColor(rows[1], order)).toBe(viridis(0.1));
    expect(kneeColor(rows[0], order)).toBe(viridis(0.9));
    expect(kneeColor(rows[3], order)).toBe(categorical(1));
    expect(kneeColor({ asinh_knees: [0.1, 1], output_knee: null }, order)).toBe(categorical(3));
  });

  it("counts the Config knobs a step reads that differ from their defaults", () => {
    const payload = {
      config: { a: 1, b: "x", c: 0.30000000000000004, d: 5 }, defaults: { a: 2, b: "x", c: 0.3, d: 5 },
      used_by: { a: ["ensemble_train"], b: ["ensemble_train"], c: ["ensemble_train"], d: ["generate"] },
    };
    expect(changedConfigKnobs(payload, "ensemble_train")).toEqual(["a"]);
    expect(changedConfigKnobs({ ...payload, config: { ...payload.config, d: 6 } }, "ensemble_train")).toEqual(["a"]);
    expect(changedConfigKnobs(payload, "ensemble_train", { except: ["a"] })).toEqual([]);
    expect(changedConfigKnobs(null, "ensemble_train")).toEqual([]);
  });
});
