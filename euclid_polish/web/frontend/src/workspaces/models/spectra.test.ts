/* The pure shaping of Models › Spectra: the payloads normalised (a null kept
 * as a gap, what is not a number dropped, the measurements' defaults), the
 * toolbar's choices, the default pick (the best ∫PSNR, the newest without
 * the isotropy rule, then the newest; current spectra first), the URL's
 * pick (in pick order) and its cap, a colour per shown model, the notes the
 * page does not repeat, the curves by period (ascending, gaps kept) and by
 * angle, the axes' domains and ticks, where the coherence falls to ½, the
 * axis / diagonal rows and the grid verdict, the map as a heat layer with
 * its rings and a clicked cell, and the confirm's estimate. */
import { describe, expect, it } from "vitest";
import type { SpectraModel } from "./spectraApi";
import {
  DEFAULT_DIRECTION, DEFAULT_PERIOD_BANDS, DEFAULT_RADIAL, MAP_COLOR_TICKS, angularCurve, angularKeys, axisDiagCell, byNumberOf, cellText,
  coherenceDomain, coherencePeriod, coherenceTicks, colourSlots, defaultPick, directionRows, divergingColor, effectivePb, fieldsOf,
  freshNotes, gridVerdict, hasAngular, hasAxisDiag, hasRadial, hasSpectra, hasValues, isGrid, isotropyText, limitPick, mapCell, mapLayer,
  mapTicks, modelHint, modelLabel, modelText, normaliseRecord, normaliseSpectra, notDrawn, parseBand, parseSection, pbLabel, periodBandKeys,
  periodCurve, periodText, pickable, radialRow, ratioDomain, reasonGroups, ringCurves, shownModels, spectraEstimate, truthRow,
} from "./spectra";
import {
  ANGLES, BANDS, EMPTY_BINS, K, PERIODS, PERIOD_BANDS, REASON, SPECS, coherenceAt, mapValues, payload, record, transferAt, truthPower,
} from "./testing/spectraFixture";

const data = () => normaliseSpectra(payload())!;
const model = (n: number) => data().models.find((m) => m.number === n)!;
const names = (ns: number[]) => ns.map((n) => `member_${n}`);

describe("toolbar choices", () => {
  it("parses only the offered values", () => {
    expect(parseSection("direction")).toBe("direction");
    expect(parseSection("spectrum")).toBeUndefined();
    expect(parseBand("H_E")).toBe("H_E");
    expect(parseBand("K")).toBeUndefined();
    expect(pbLabel("12-21")).toBe("12–21 px");
  });
});

describe("normalise", () => {
  it("reads the summary with every collection present", () => {
    const p = data();
    expect(p.models.map((m) => m.name)).toEqual(SPECS.map((s) => `member_${s.n}`));
    expect(p.period_bands).toEqual(PERIOD_BANDS);
    expect(p.period_px).toEqual(PERIODS);
    expect(p.angle_deg).toEqual(ANGLES);
    expect(p.radial).toEqual({ window_px: 128, stretch_knee_e: 1 });
    expect(p.direction).toEqual({ window_px: 64, long_window_px: 128, stretch_knee_e: 0.01 });
    expect(p.pixel_scale_arcsec).toBe(0.05);
    const m = model(218);
    expect(m.radial?.power_sr[0]).toHaveLength(63);
    expect(m.radial?.transfer[0][62]).toBeNull();                       // the finest ring: non-finite
    expect(m.radial?.coherence[0][0]).toBeNull();
    expect(m.axis_diag?.["2-3"][0]).toEqual([2, 1]);
    expect(m.angular?.["12-21"][0][EMPTY_BINS[0]]).toBeNull();
    expect(model(230).radial).toBeNull();
  });

  it("keeps a null as a gap, drops what is not a number, and fills the defaults", () => {
    const p = normaliseSpectra({
      models: [
        { name: "member_240", status: 7, isotropy_penalty: "0.5",
          radial: { power_sr: [[1, null, "x", Infinity]], coherence: "bad" },
          axis_diag: { "2-3": [[1.2, null], "x", null] }, angular: { "3-6": [[1, null]], "6-12": "bad" } },
        { status: "current" },
      ],
      period_px: [128, null, "64"], period_bands: [[3, 2], "x"], records_local: "no",
    })!;
    expect(p.models).toHaveLength(1);                                    // a row without a name is dropped
    const m = p.models[0];
    expect(m).toMatchObject({ number: 240, status: "missing", isotropy_penalty: null, reason: null, label: null });
    expect(m.radial).toEqual({ power_sr: [[1, null, null, null]], coherence: [], transfer: [] });
    expect(m.axis_diag).toEqual({ "2-3": [[1.2, null], null, null] });
    expect(m.angular).toEqual({ "3-6": [[1, null]], "6-12": [] });
    expect(p.period_px).toEqual([128, NaN, NaN]);
    expect(p.period_bands).toEqual(DEFAULT_PERIOD_BANDS);
    expect(p).toMatchObject({ bands: BANDS, records_local: true, missing: [], notes: [], truth_power: null, pixel_scale_arcsec: null });
    expect(p.radial).toEqual(DEFAULT_RADIAL);
    expect(p.direction).toEqual(DEFAULT_DIRECTION);
    expect(normaliseSpectra({ records_local: false })?.records_local).toBe(false);
  });

  it("is null for a reply that is not a payload", () => {
    expect(normaliseSpectra(null)).toBeNull();
    expect(normaliseSpectra([])).toBeNull();
    expect(normaliseSpectra("spectra")).toBeNull();
  });

  it("reads a record with its map; the rings default to 21 and 3 px", () => {
    const r = normaliseRecord(record(218))!;
    expect(r.map?.k).toEqual(K);
    expect(r.map?.values).toHaveLength(4);
    expect(r.map?.values[0]).toHaveLength(64);
    expect(r.map).toMatchObject({ period_lines_px: [21, 3], n_windows: 3600 });
    expect(normaliseRecord({ name: "member_1", map: { k: [0], values: [[[1]]] } })?.map).toEqual({
      k: [0], values: [[[1]]], period_lines_px: [21, 3], n_windows: null,
    });
    expect(normaliseRecord({ name: "member_1", map: "x" })?.map).toBeNull();
    expect(normaliseRecord({ ok: false, error: "no spectra" })).toBeNull();
  });
});

describe("models", () => {
  it("lists the models with spectra by number", () => {
    const p = data();
    expect(pickable(p.models).map((m) => m.name)).toEqual(names([218, 227, 233, 235, 238]));
    expect(hasSpectra(model(239))).toBe(false);                          // current, every spectrum null
    expect([hasRadial(model(218)), hasAngular(model(218)), hasAxisDiag(model(218))]).toEqual([true, true, true]);
    expect(modelLabel(model(238))).toBe("#238");
    expect(modelLabel({ name: "odd", number: null })).toBe("odd");
  });

  it("picks the best ∫PSNR, the newest without the isotropy rule, then the newest: current spectra only", () => {
    const p = data();
    expect(defaultPick(p.models)).toEqual(names([218, 227, 233, 238]));  // 235 is stale
    expect(defaultPick(p.models, 2)).toEqual(names([227, 238]));         // the best is also the newest without the rule
    const best238 = p.models.map((m) => (m.number === 238 ? { ...m, mean_psnr: 63 } : m));
    expect(defaultPick(best238, 3)).toEqual(names([227, 233, 238]));    // 238 best, 227 the reference, then the newest
    const stale = p.models.map((m): SpectraModel => ({ ...m, status: "stale" }));
    expect(defaultPick(stale)).toEqual(names([218, 227, 233, 235, 238]));
    expect(defaultPick([])).toEqual([]);
  });

  it("shows the URL's pick (with spectra, capped) in pick order, else the default; draws them by number", () => {
    const p = data();
    expect(shownModels(p.models, ["member_238", "member_230", "nope", "member_218", "member_238"])).toEqual(names([238, 218]));
    expect(shownModels(p.models, ["member_230"])).toEqual(defaultPick(p.models));
    expect(shownModels(p.models, [])).toEqual(defaultPick(p.models));
    expect(shownModels(p.models, names([238, 233, 227, 218]), 2)).toEqual(names([238, 233]));
    expect(byNumberOf(p.models, names([238, 218, 230])).map((m) => m.name)).toEqual(names([218, 238]));
    expect(limitPick(["a", "b", "c"], 2)).toEqual(["b", "c"]);
    expect(limitPick([])).toEqual([]);
  });

  it("gives each shown model its own colour slot by its place in the pick, kept when another is added", () => {
    expect([...colourSlots(names([238, 218, 233]))]).toEqual([["member_238", 0], ["member_218", 1], ["member_233", 2]]);
    const added = colourSlots(names([238, 218, 233, 227]));
    expect([added.get("member_238"), added.get("member_218"), added.get("member_227")]).toEqual([0, 1, 3]);
    const many = colourSlots(Array.from({ length: 8 }, (_, i) => `member_${300 + i}`));
    expect(new Set(many.values()).size).toBe(8);
  });

  it("describes a model for the picker", () => {
    expect(modelText(model(238))).toBe("#238 · depth 16 · 1.9M steps · ∫PSNR 62.13 dB · isotropy 0.5");
    expect(modelText(model(235))).toMatch(/· isotropy 0\.05 · stale spectra \(an earlier checkpoint or other test records\)$/);
    expect(modelHint(model(235))).toBe("depth 16 · 200k steps · ∫PSNR 60.88 dB · isotropy 0.05 · stale");
    expect(isotropyText(0)).toBe("no isotropy rule");
    expect(isotropyText(null)).toBe("no isotropy rule");
  });

  it("names the models a figure cannot draw, and the reasons", () => {
    const p = data();
    expect(notDrawn(p.models, hasRadial)).toEqual(["#230", `#239 (${REASON})`]);
    expect(reasonGroups(p.models)).toEqual([{ reason: REASON, labels: ["#239"] }]);
    expect(fieldsOf(p)).toBe(100);
    expect(fieldsOf(null)).toBeNull();
  });

  it("drops a note that only lists the models without spectra", () => {
    const p = data();
    expect(freshNotes(p.notes, p.missing)).toEqual([]);
    expect(freshNotes(["the test records changed", ...p.notes], p.missing)).toEqual(["the test records changed"]);
    expect(freshNotes(["member_230 failed"], p.missing)).toEqual(["member_230 failed"]);
    expect(freshNotes(["anything"], [])).toEqual(["anything"]);
  });
});

describe("curves", () => {
  it("draws a radial row ascending in period, the finest ring's null kept as a gap", () => {
    const p = data();
    const c = periodCurve(p, radialRow(p, model(218), "transfer", "VIS"));
    expect(c.x).toEqual([...PERIODS].reverse());
    expect(c.y[0]).toBeNull();
    expect(c.y[62]).toBe(transferAt(0.3, 128));
    expect(hasValues(c)).toBe(true);
    expect(hasValues({ x: [1], y: [null] })).toBe(false);
    const truth = periodCurve(p, truthRow(p, "H_E"));
    expect(truth.y[62]).toBe(truthPower(3, 128));
  });

  it("reads the band by the payload's order, and the periods from the rings when the payload has none", () => {
    const p = { ...data(), bands: ["H_E", "J_E", "Y_E", "VIS"] };
    const m = model(218);
    expect(radialRow(p, m, "transfer", "VIS")).toBe(m.radial?.transfer[3]);
    const c = periodCurve({ period_px: [] }, [1, 2, 3, 4]);
    expect(c.x).toEqual([32, 128 / 3, 64, 128]);
    expect(c.y).toEqual([4, 3, 2, 1]);
    expect(periodCurve({ period_px: [128, 64, NaN] }, [1])).toEqual({ x: [64, 128], y: [null, 1] });
    expect(periodCurve({ period_px: [] }, null)).toEqual({ x: [], y: [] });
  });

  it("draws an angular row against the bins, empty bins as gaps", () => {
    const p = data();
    const c = angularCurve(p, model(218), "12-21", "VIS");
    expect(c.x).toEqual(ANGLES);
    expect(EMPTY_BINS.map((i) => c.y[i])).toEqual([null, null, null]);
    expect(c.y[0]).toBeGreaterThan(2);                                   // the pixel axis
    expect(angularCurve({ ...p, angle_deg: [] }, model(218), "3-6", "VIS").x.slice(0, 2)).toEqual([2.5, 7.5]);
    expect(angularCurve(p, model(218), "21-42", "VIS")).toEqual({ x: [], y: [] });
  });

  it("gives a ratio a log domain symmetric about 1, at least ½ to 2", () => {
    expect(ratioDomain([0.7, 1.6, null])).toEqual([0.5, 2]);
    const [lo, hi] = ratioDomain([0.2, 3]);
    expect(hi).toBeCloseTo(5.75, 9);
    expect(lo).toBeCloseTo(1 / 5.75, 9);
    expect(ratioDomain([1e-6])).toEqual([0.01, 100]);
    expect(ratioDomain([])).toEqual([0.5, 2]);
  });

  it("runs the coherence from 0 (or its lowest value) to 1, in quarters", () => {
    expect(coherenceDomain([0.2, 0.9, null])).toEqual([0, 1.02]);
    expect(coherenceDomain([-0.055, 0.9])).toEqual([-0.12, 1.02]);
    expect(coherenceDomain([-3])).toEqual([-1.02, 1.02]);
    expect(coherenceTicks([-0.12, 1.02]).map((t) => t.label)).toEqual(["0", "0.25", "0.5", "0.75", "1"]);
    expect(coherenceTicks([-0.62, 1.02]).map((t) => t.v)).toEqual([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]);
  });

  it("names a period in px, and in arcsec with the pixel scale", () => {
    expect(periodText(4)).toBe("4 px");
    expect(periodText(2.0317)).toBe("2.03 px");
    expect(periodText(4, 0.05)).toBe("4 px (0.2″)");
  });

  it("finds where the coherence falls to ½, interpolated in log period", () => {
    const h = coherencePeriod({ x: [2, 4, 8], y: [0.2, 0.4, 0.8] });
    expect(h?.holds).toBe(false);
    expect(h?.period).toBeCloseTo(8 / 2 ** 0.75, 9);
    expect(coherencePeriod({ x: [2, 3, 4], y: [0.2, null, 0.9] })?.period).toBeCloseTo(4 / 2 ** (0.4 / 0.7), 9);
    expect(coherencePeriod({ x: [2, 4], y: [0.6, 0.9] })).toEqual({ period: 2, holds: true });
    expect(coherencePeriod({ x: [2, 4], y: [0.1, 0.3] })).toBeNull();
    expect(coherencePeriod({ x: [2], y: [null] })).toBeNull();
    // the fixture: 1 − c·(2/p)^0.7 = ½ at p = 2·(2c)^(1/0.7) for 218 (c = 0.9); 238 (c = 0.45) never falls
    const p = data();
    const at = (n: number) => coherencePeriod(periodCurve(p, radialRow(p, model(n), "coherence", "VIS")));
    expect(at(218)?.period).toBeCloseTo(2 * (2 * 0.9) ** (1 / 0.7), 1);
    expect(at(238)).toEqual({ period: PERIODS[61], holds: true });
    expect(coherenceAt(0.45, PERIODS[61])).toBeGreaterThan(0.5);
  });
});

describe("direction", () => {
  it("orders the period bands, the angular ones, and the band drawn", () => {
    const p = data();
    expect(periodBandKeys(p)).toEqual(["2-3", "3-6", "6-12", "12-21", "21-42"]);
    const extra = { ...p, models: [{ ...p.models[0], axis_diag: { "42-85": [], "1-2": [] } }] };
    expect(periodBandKeys(extra)).toEqual(["2-3", "3-6", "6-12", "12-21", "21-42", "1-2", "42-85"]);
    expect(angularKeys(p)).toEqual(["2-3", "3-6", "6-12", "12-21"]);
    expect(angularKeys({ ...p, models: [] })).toEqual(["2-3", "3-6", "6-12", "12-21"]);
    expect(effectivePb("2-3", angularKeys(p))).toBe("2-3");
    expect(effectivePb("21-42", angularKeys(p))).toBe("3-6");
    expect(effectivePb("x", ["6-12", "12-21"])).toBe("6-12");
  });

  it("reads an axis / diagonal pair and calls 1.2 or more a grid", () => {
    expect(axisDiagCell([2, 1])).toEqual({ axis: 2, diag: 1, ratio: 2 });
    expect(axisDiagCell([1, 0])?.ratio).toBeNull();
    expect(axisDiagCell([1.1, null])).toEqual({ axis: 1.1, diag: null, ratio: null });
    expect(axisDiagCell([null, null])).toBeNull();
    expect(axisDiagCell(undefined)).toBeNull();
    expect([isGrid(1.2), isGrid(1.19), isGrid(null)]).toEqual([true, false, false]);
  });

  it("lists every model with axis / diagonal values, newest first", () => {
    const p = data();
    const rows = directionRows(p, "VIS");
    expect(rows.map((r) => r.name)).toEqual(names([238, 235, 233, 227, 218]));
    const r218 = rows.find((r) => r.num === 218)!;
    expect(r218.cells["2-3"]?.ratio).toBe(2);
    expect(r218.cells["3-6"]?.ratio).toBe(1.5);
    expect(r218).toMatchObject({ isotropy: 0, label: "#218", status: "current", steps: 2_000_000, mean_psnr: 62.1543 });
    expect(rows.find((r) => r.num === 233)?.cells["21-42"]).toBeNull();
    expect(directionRows(p, "H_E").find((r) => r.num === 238)?.cells["2-3"]?.ratio).toBeCloseTo(1.3, 9);
  });

  it("judges a grid one period band at a time, newest first", () => {
    const p = data();
    const vis = gridVerdict(directionRows(p, "VIS"), "3-6");
    expect(vis.grid.map((r) => r.label)).toEqual(["#227", "#218"]);
    expect(vis.round.map((r) => r.label)).toEqual(["#238", "#235", "#233"]);
    expect(gridVerdict(directionRows(p, "H_E"), "2-3").grid.map((r) => r.label)).toEqual(["#238", "#227", "#218"]);
    expect(gridVerdict(directionRows(p, "VIS"), "21-42").measured.map((r) => r.label)).toEqual(["#238", "#235", "#227", "#218"]);
    expect(gridVerdict([], "3-6")).toEqual({ measured: [], grid: [], round: [] });
  });
});

describe("the error map", () => {
  const layer = () => mapLayer(normaliseRecord(record(218)), "VIS")!;

  it("is a heat layer of log₂ ratios in x-major cells, blank outside the measured disc", () => {
    const l = layer();
    expect(l.z).toHaveLength(64);
    expect(l.z[0]).toHaveLength(64);
    expect(l.z[32][48]).toBeCloseTo(Math.log2(3), 9);                   // kx 0, ky 16: on the axis (1 + g, g = 2)
    expect(l.z[37][37]).toBeCloseTo(Math.log2(0.9), 9);                 // kx 5, ky 5: off the axes
    expect(l.z[32][32]).toBeNaN();                                       // the DC bin
    expect(l.z[0][0]).toBeNaN();                                         // beyond |k| = 32
    expect([l.xEdges[0], l.xEdges[64], l.size, l.extent]).toEqual([-32.5, 31.5, 64, 32.5]);
    expect(l.periodLines).toEqual([21, 3]);
    expect(mapValues(0)[0][32][40]).toBe(1);
  });

  it("transposes the server's rows (ky) into the plot's columns, and centres a k grid it cannot use", () => {
    const rec = normaliseRecord({ name: "member_1", map: { k: [0, 1], values: [[[1, 2], [4, 8]]] } });
    const l = mapLayer(rec, "VIS")!;
    expect(l.z[1][0]).toBe(1);                                           // kx 1, ky 0 → log₂ 2
    expect(l.z[0][1]).toBe(2);                                           // kx 0, ky 1 → log₂ 4
    const odd = mapLayer(normaliseRecord({ name: "member_1", map: { k: [3, 2, 1], values: [[[1, 1, 1]]] } }), "VIS")!;
    expect(odd.kx).toEqual([-1, 0, 1]);
    expect(odd.ky).toEqual([0]);
    expect(mapLayer(normaliseRecord({ name: "member_1" }), "VIS")).toBeNull();
    expect(mapLayer(normaliseRecord({ name: "member_1", map: { k: [], values: [] } }), "VIS")).toBeNull();
    expect(mapLayer(null, "VIS")).toBeNull();
  });

  it("rings the periods inside the map: |k| = size / period", () => {
    const rings = ringCurves([21, 3, 1.5], 64, 32.5);
    expect(rings.map((r) => r.period)).toEqual([21, 3]);
    expect(rings[0].x[0]).toBeCloseTo(64 / 21, 9);
    expect(rings[1].y[24]).toBeCloseTo(64 / 3, 9);                      // a quarter turn
    expect(mapTicks(32.5).map((t) => t.v)).toEqual([-32, -16, 0, 16, 32]);
  });

  it("reads a clicked cell: its wave vector, period, direction and ratio", () => {
    const l = layer();
    const axis = mapCell(l, { i: 32, j: 48 })!;
    expect(axis).toMatchObject({ kx: 0, ky: 16, period: 4, angle: 90 });
    expect(axis.ratio).toBeCloseTo(3, 9);
    expect(cellText(axis)).toBe("kx 0, ky 16 · period 4 px at 90° · 3.00× the round expectation");
    const corner = mapCell(l, { i: 0, j: 0 })!;
    expect(corner.ratio).toBeNull();
    expect(cellText(corner)).toMatch(/at 45° · outside the measured disc$/);
    expect(mapCell(l, { i: 31, j: 32 })?.angle).toBe(0);                 // kx −1: the same axis as +1
    expect(cellText(mapCell(l, { i: 32, j: 32 })!)).toBe("kx 0, ky 0 · the mean (no direction)");
    expect(mapCell(l, { i: 64, j: 0 })).toBeNull();
  });

  it("colours a ratio on a diverging scale: blue under 1, white at 1, red over", () => {
    const rgb = (t: number) => (/^rgb\((\d+), (\d+), (\d+)\)$/.exec(divergingColor(t)) ?? []).slice(1).map(Number);
    const [r0, , b0] = rgb(0);
    const [r1, , b1] = rgb(1);
    expect(b0).toBeGreaterThan(r0);
    expect(r1).toBeGreaterThan(b1);
    expect(Math.min(...rgb(0.5))).toBeGreaterThan(220);
    expect(divergingColor(NaN)).toBe(divergingColor(0.5));
    expect(MAP_COLOR_TICKS).toEqual([{ v: -1, label: "0.5" }, { v: 0, label: "1" }, { v: 1, label: "2" }]);
  });
});

describe("the compute job", () => {
  it("estimates ~25 s of spectra per model", () => {
    expect(spectraEstimate(2)).toBe("local job: 2 models, ~50 s of spectra");
    expect(spectraEstimate(1)).toBe("local job: 1 model, ~30 s of spectra");
    expect(spectraEstimate(35)).toBe("local job: 35 models, ~15 min of spectra");
  });
});
