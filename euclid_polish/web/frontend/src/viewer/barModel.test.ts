import { describe, expect, it } from "vitest";
import {
  BAR_ROWS_STORAGE_KEY, bandLabel, barLayout, navPosition, parsePosition, rememberBarRows, reservedBarRows, chipTiers, colourOptions, formatSig, groupUnit, isSinglePlane, parseNumber, sameBarLayout, sentenceLabel, sequencePending, shortTierLabel, plainLabel, readoutTierName,
} from "./barModel";

describe("shortTierLabel", () => {
  it("keeps the name before the detail", () => {
    expect(shortTierLabel("SR · production gate")).toBe("SR");
    expect(shortTierLabel("BHR (blurred HR)")).toBe("BHR");
    expect(shortTierLabel("SR 169·psnr")).toBe("SR 169");
    expect(shortTierLabel("RBF · minibatched convex all-asinh RBF")).toBe("RBF");
    expect(shortTierLabel("Mean of members")).toBe("Mean of members");
    expect(shortTierLabel("stdSR")).toBe("stdSR");
  });
  it("capitalises an all-lowercase label", () => {
    expect(shortTierLabel("disagreement movie")).toBe("Disagreement movie");
    expect(sentenceLabel("disagreement movie")).toBe("Disagreement movie");
    expect(sentenceLabel("stdSR")).toBe("stdSR");
    expect(sentenceLabel("SR 169·psnr")).toBe("SR 169·psnr");
  });
  it("keeps the readout's tier names short (the first word of a long one)", () => {
    expect(readoutTierName("Mean of 30 STARFULL members")).toBe("Mean");
    expect(readoutTierName("SR · production gate")).toBe("SR");
    expect(readoutTierName("Production")).toBe("Production");
    expect(readoutTierName("disagreement movie")).toBe("Disagreement");
  });
  it("names a FITS HDU tier by its HDU name, not its index", () => {
    expect(shortTierLabel("1 · LR_VIS")).toBe("LR VIS");
    expect(shortTierLabel("6 · SR_Y_E")).toBe("SR Y");
    expect(shortTierLabel("12 · SCI")).toBe("SCI");
    expect(shortTierLabel("LR · 4-band colour")).toBe("LR");
  });
  it("reads served labels in plain words (regimes in lower case, '(native)')", () => {
    expect(plainLabel("Mean of 30 STARFULL members")).toBe("Mean of 30 starfull members");
    expect(shortTierLabel("Mean of 30 STARLESS members")).toBe("Mean of 30 starless members");
    expect(sentenceLabel("NEXUS F200W · native")).toBe("NEXUS F200W (native)");
    expect(sentenceLabel("JWST · native")).toBe("JWST (native)");
    expect(shortTierLabel("JWST · native")).toBe("JWST");
    expect(plainLabel("stdSR (STARFULL members)")).toBe("stdSR (starfull members)");
  });
});

describe("bands and colours", () => {
  it("shortens the NISP band names only", () => {
    expect(["VIS", "Y_E", "J_E", "H_E", "F200W"].map(bandLabel)).toEqual(["VIS", "Y", "J", "H", "F200W"]);
  });
  it("lists the bands, Lupton and Temp with the old Q–Y keys", () => {
    const o = colourOptions(["VIS", "Y_E", "J_E", "H_E"], { logMode: false, singlePlane: false });
    expect(o.map((c) => `${c.label}:${c.shortcut}`)).toEqual(["VIS:Q", "Y:W", "J:E", "H:R", "Lupton:T", "Temp:Y"]);
    expect(o[1]).toMatchObject({ key: "Y_E", title: "Y_E" });
  });
  it("offers no band choice for a log collection or single-plane tiers", () => {
    expect(colourOptions(["VIS"], { logMode: true, singlePlane: false })).toEqual([]);
    expect(colourOptions(["VIS", "Y_E"], { logMode: false, singlePlane: true })).toEqual([]);
    expect(isSinglePlane([{ c: 1 }, { c: 1 }])).toBe(true);
    expect(isSinglePlane([{ c: 1 }, { c: 4 }])).toBe(false);
    expect(isSinglePlane([])).toBe(false);
  });
});

describe("chipTiers", () => {
  const t = (key: string, hidden = false) => ({ key, label: key.toUpperCase(), hidden });
  it("gives every visible tier a chip, hidden ones none", () => {
    expect(chipTiers([t("lr"), t("sr"), t("std", true), t("hr")], []).map((x) => x.key)).toEqual(["lr", "sr", "hr"]);
  });
  it("past the cap: the first ones plus the selected ones", () => {
    const many = ["a", "b", "c", "d", "e"].map((k) => t(k));
    expect(chipTiers(many, ["e"], 3).map((x) => x.key)).toEqual(["a", "b", "e"]);
  });
});

describe("groupUnit", () => {
  it("reads a JWST group in its display unit with its scale", () => {
    const recs = [{ transferGroup: "euclid", unit: "e-", displayScale: 1 }, { transferGroup: "jwst", unit: "MJy/sr", displayScale: 3888.34 }];
    expect(groupUnit(recs, (r) => r.transferGroup === "jwst")).toEqual({ unit: "MJy/sr", scale: 3888.34 });
    expect(groupUnit(recs, (r) => r.transferGroup === "euclid")).toEqual({ unit: "e⁻", scale: 1 });
  });
  it("falls back to the meta unit without a shown cube", () => {
    expect(groupUnit([], () => true, "e-")).toEqual({ unit: "e⁻", scale: 1 });
    expect(groupUnit([], () => true)).toEqual({ unit: "", scale: 1 });
  });
});

describe("numbers", () => {
  it("formats control values compactly", () => {
    expect([0.1, 3.16227, 100, 1e4, 0.0257, -3, 0].map(formatSig)).toEqual(["0.1", "3.16", "100", "10000", "0.0257", "−3", "0"]);
    expect(formatSig(2.5e-5)).toBe("2.50e-5");
  });
  it("parses typed numbers", () => {
    expect(parseNumber("1e4")).toBe(1e4);
    expect(parseNumber(" 0,5 ")).toBe(0.5);
    expect(parseNumber("−3")).toBe(-3);
    expect(parseNumber("abc")).toBeNull();
    expect(parseNumber("")).toBeNull();
  });
});

describe("sequencePending", () => {
  it("is true right after a plain g", () => {
    expect(sequencePending({ key: "g", t: 1000 }, 1500)).toBe(true);
    expect(sequencePending({ key: "g", t: 1000 }, 2100)).toBe(false);
    expect(sequencePending({ key: "e", t: 1000 }, 1100)).toBe(false);
    expect(sequencePending(null, 1100)).toBe(false);
  });
});

describe("barLayout", () => {
  const items = (w1: number[], w2: number[]) => [...w1.map((width) => ({ width, row: 1 as const })), ...w2.map((width) => ({ width, row: 2 as const }))];
  it("one row with the texts when everything fits", () => {
    expect(barLayout({ items: items([200, 190], [96, 92, 86, 82, 160, 82]), textWidth: 110, gap: 8, available: 1100 })).toEqual({ rows: 1, compact: false, wrap: [] });
  });
  it("one row icon-only when only the texts overflow (1280 px window)", () => {
    // 988 + 64 gaps = 1052 > 1000; without the 110 px of text: 942
    expect(barLayout({ items: items([200, 190], [96, 92, 86, 82, 160, 82]), textWidth: 110, gap: 8, available: 1000 })).toEqual({ rows: 1, compact: true, wrap: [] });
  });
  it("two rows below that, texts kept while the second row fits (792 px stage)", () => {
    expect(barLayout({ items: items([200, 190], [96, 92, 86, 82, 160, 82]), textWidth: 110, gap: 8, available: 740 })).toEqual({ rows: 2, compact: false, wrap: [] });
  });
  it("two rows icon-only in a narrow inspector", () => {
    expect(barLayout({ items: items([150, 190], [96, 92, 86, 82, 82]), textWidth: 110, gap: 8, available: 360 })).toEqual({ rows: 2, compact: true, wrap: [] });
  });
  it("a row too wide even icon-only wraps (never hides a control) (a 300 px viewer beside the inspector)", () => {
    // what: 227 + 8 + 194 = 429 > 288; how: 98 + 26 + 82 + 26 + 26 + 5 gaps = 298, icon-only 248
    const it2 = items([227, 194], [98, 26, 82, 26, 26, 26]);
    expect(barLayout({ items: it2, textWidth: 71, gap: 8, available: 288 })).toEqual({ rows: 2, compact: true, wrap: [1] });
    expect(barLayout({ items: it2, textWidth: 71, gap: 8, available: 200 })).toEqual({ rows: 2, compact: true, wrap: [1, 2] });
  });
  it("sameBarLayout compares every field", () => {
    expect(sameBarLayout({ rows: 2, compact: true, wrap: [1] }, { rows: 2, compact: true, wrap: [1] })).toBe(true);
    expect(sameBarLayout({ rows: 2, compact: true, wrap: [1] }, { rows: 2, compact: true, wrap: [] })).toBe(false);
  });
  it("unknown width: one row", () => {
    expect(barLayout({ items: items([100], [100]), textWidth: 0, gap: 8, available: 0 })).toEqual({ rows: 1, compact: false, wrap: [] });
  });
});

describe("no control out of sight", () => {
  it("every row either fits the bar or wraps, at every width", () => {
    const rowsOf = (w1: number[], w2: number[]) => [...w1.map((width) => ({ width, row: 1 as const })), ...w2.map((width) => ({ width, row: 2 as const }))];
    const items = rowsOf([227, 194], [98, 86, 26, 82, 154, 82]);
    const span = (ws: number[]) => ws.reduce((s, w) => s + w, 0) + 8 * (ws.length - 1);
    for (let available = 120; available <= 1400; available += 7) {
      const lay = barLayout({ items, textWidth: 70, gap: 8, available });
      if (lay.rows === 1) {
        expect(span([227, 194, 98, 86, 26, 82, 154, 82]) - (lay.compact ? 70 : 0)).toBeLessThanOrEqual(available);
        continue;
      }
      const first = span([227, 194]);
      const second = span([98, 86, 26, 82, 154, 82]) - (lay.compact ? 70 : 0);
      expect(first <= available || lay.wrap.includes(1)).toBe(true);
      expect(second <= available || lay.wrap.includes(2)).toBe(true);
    }
  });
});

describe("reserved bar height", () => {
  const mem = () => { const m = new Map<string, string>(); return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => { m.set(k, v); } }; };
  it("remembers the rows per collection, else guesses from the width", () => {
    const s = mem();
    expect(reservedBarRows("records", 1000, s)).toBe(1);
    expect(reservedBarRows("records", 600, s)).toBe(2);
    rememberBarRows("records", 2, s);
    expect(reservedBarRows("records", 1000, s)).toBe(2);
    expect(JSON.parse(s.getItem(BAR_ROWS_STORAGE_KEY)!)).toEqual({ records: 2 });
    expect(reservedBarRows("other", 1000, s)).toBe(1);
  });
  it("survives broken or missing storage", () => {
    const bad = { getItem: () => { throw new Error("denied"); }, setItem: () => { throw new Error("denied"); } };
    expect(reservedBarRows("x", 500, bad)).toBe(2);
    expect(() => rememberBarRows("x", 2, bad)).not.toThrow();
    expect(reservedBarRows("x", 900, null)).toBe(1);
    const junk = { getItem: () => "not json", setItem: () => {} };
    expect(reservedBarRows("x", 900, junk)).toBe(1);
  });
});

describe("navigation counter", () => {
  it("counts from 1 over the object count", () => {
    expect(navPosition(0, 100)).toEqual({ position: "1", total: "100" });
    expect(navPosition(11, 100)).toEqual({ position: "12", total: "100" });
  });
  it("a typed position is 1-based", () => {
    expect(parsePosition("12")).toBe(11);
    expect(parsePosition(" 1 ")).toBe(0);
    expect(parsePosition("abc")).toBeNull();
    expect(parsePosition("")).toBeNull();
  });
});
