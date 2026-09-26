import { describe, expect, it } from "vitest";
import {
  DASH,
  formatBytes,
  formatCount,
  formatDate,
  formatDateTime,
  formatDec,
  formatDeg,
  formatDuration,
  formatMagnitude,
  formatNumber,
  formatPercent,
  formatPow10,
  formatRA,
  formatRaDec,
  formatRelative,
  formatSI,
  isFiniteNumber,
  parseSkyCoord,
  parseTimestamp,
  superscript,
} from "./format";

describe("formatNumber", () => {
  it("falls back to a dash for missing / non-finite values", () => {
    for (const v of [null, undefined, NaN, Infinity, "12" as unknown as number]) {
      expect(formatNumber(v)).toBe(DASH);
    }
    expect(formatNumber(null, { fallback: "n/a" })).toBe("n/a");
  });

  it("uses fixed decimals when asked", () => {
    expect(formatNumber(3.14159, { digits: 2 })).toBe("3.14");
    expect(formatNumber(2, { digits: 1 })).toBe("2.0");
    expect(formatNumber(12345.678, { digits: 1 })).toBe("12,345.7");
    expect(formatNumber(12345.678, { digits: 1, grouping: false })).toBe("12345.7");
  });

  it("uses significant digits by default and exponent notation at the extremes", () => {
    expect(formatNumber(0)).toBe("0");
    expect(formatNumber(42)).toBe("42");
    expect(formatNumber(3.14159)).toBe("3.14");
    expect(formatNumber(0.012345)).toBe("0.0123");
    expect(formatNumber(1234.5)).toBe("1,235");
    expect(formatNumber(-0.5)).toBe("-0.5");
    expect(formatNumber(2.5e-5)).toBe("2.5e-5");
    expect(formatNumber(3.2e9)).toBe("3.2e9");
    expect(formatNumber(1234.5, { sig: 2 })).toBe("1,200");
  });

  it("can force a sign and append a unit", () => {
    expect(formatNumber(0.25, { digits: 2, signed: true })).toBe("+0.25");
    expect(formatNumber(-1, { digits: 0, signed: true })).toBe("-1");
    expect(formatNumber(43.21, { digits: 1, unit: "dB" })).toBe("43.2 dB");
  });

  it("formats integer counts with grouping", () => {
    expect(formatCount(43401)).toBe("43,401");
    expect(formatCount(12.7)).toBe("13");
    expect(formatCount(undefined)).toBe(DASH);
  });

  it("narrows unknown values", () => {
    expect(isFiniteNumber(1)).toBe(true);
    expect(isFiniteNumber(NaN)).toBe(false);
    expect(isFiniteNumber("1")).toBe(false);
  });
});

describe("formatSI / formatBytes / formatPercent", () => {
  it("scales by SI prefixes", () => {
    expect(formatSI(0)).toBe("0");
    expect(formatSI(950)).toBe("950");
    expect(formatSI(1234)).toBe("1.23 k");
    expect(formatSI(2.5e6, { unit: "e⁻" })).toBe("2.5 Me⁻");
    expect(formatSI(0.00042, { unit: "s" })).toBe("420 µs");
    expect(formatSI(-12_300)).toBe("-12.3 k");
    expect(formatSI(999_999)).toBe("1 M");
    expect(formatSI(NaN)).toBe(DASH);
  });

  it("formats byte counts in binary units", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(10 * 1024 ** 2)).toBe("10 MB");
    expect(formatBytes(3.25 * 1024 ** 3)).toBe("3.3 GB");
    expect(formatBytes(-1)).toBe(DASH);
  });

  it("formats fractions as percentages", () => {
    expect(formatPercent(0.1234)).toBe("12.3%");
    expect(formatPercent(1)).toBe("100.0%");
    expect(formatPercent(0.5, 0)).toBe("50%");
    expect(formatPercent(null)).toBe(DASH);
  });
});

describe("formatDuration", () => {
  it("covers seconds to days without 60s / 60m carries", () => {
    expect(formatDuration(4.24)).toBe("4.2s");
    expect(formatDuration(42.4)).toBe("42s");
    expect(formatDuration(59.6)).toBe("1m 00s");
    expect(formatDuration(185)).toBe("3m 05s");
    expect(formatDuration(119.7)).toBe("2m 00s");
    expect(formatDuration(7440)).toBe("2h 04m");
    expect(formatDuration(3599.9)).toBe("1h 00m");
    expect(formatDuration(3 * 86400 + 4 * 3600)).toBe("3d 4h");
    expect(formatDuration(-1)).toBe(DASH);
    expect(formatDuration(null)).toBe(DASH);
  });
});

describe("magnitudes and powers of ten", () => {
  it("formats AB magnitudes with optional uncertainty", () => {
    expect(formatMagnitude(19.2345)).toBe("19.23");
    expect(formatMagnitude(19.2345, { sigma: 0.051 })).toBe("19.23 ± 0.05");
    expect(formatMagnitude(19.2, { unit: true })).toBe("19.20 mag");
    expect(formatMagnitude(undefined)).toBe(DASH);
  });

  it("writes superscripts and powers of ten", () => {
    expect(superscript(-12)).toBe("⁻¹²");
    expect(superscript("3")).toBe("³");
    expect(formatPow10(0)).toBe("1");
    expect(formatPow10(1)).toBe("10");
    expect(formatPow10(-3)).toBe("10⁻³");
    expect(formatPow10(5)).toBe("10⁵");
  });
});

describe("sky coordinates", () => {
  it("formats RA as hours and Dec as signed degrees", () => {
    expect(formatRA(267.4229)).toBe("17h49m41.50s");
    expect(formatRA(267.4229, { style: "colon" })).toBe("17:49:41.50");
    expect(formatDec(64.8873)).toBe("+64°53′14.3″");
    expect(formatDec(-27.78)).toBe("-27°46′48.0″");
    expect(formatDec(-0.5)).toBe("-00°30′00.0″");
    expect(formatDec(64.8873, { style: "colon" })).toBe("+64:53:14.3");
  });

  it("carries rounding into the next unit", () => {
    expect(formatRA(359.99999999)).toBe("00h00m00.00s");
    expect(formatRA(14.999999999)).toBe("01h00m00.00s");
    expect(formatDec(29.99999999)).toBe("+30°00′00.0″");
  });

  it("formats decimal degrees and pairs", () => {
    expect(formatDeg(267.4229)).toBe("267.42290°");
    expect(formatDeg(-27.78, 2)).toBe("-27.78°");
    expect(formatRaDec(267.4229, 64.8873)).toBe("17h49m41.50s +64°53′14.3″");
    expect(formatRaDec(267.4229, 64.8873, { mode: "degrees" })).toBe("267.42290° +64.88730°");
    expect(formatRaDec(267.4229, 64.8873, { mode: "both" }))
      .toBe("17h49m41.50s +64°53′14.3″ (267.42290°, +64.88730°)");
    expect(formatRaDec(null, 1)).toBe(DASH);
  });

  it("parses degrees and sexagesimal input", () => {
    expect(parseSkyCoord("267.4229 64.8873")).toEqual({ ra: 267.4229, dec: 64.8873 });
    expect(parseSkyCoord("53.16, -27.78")).toEqual({ ra: 53.16, dec: -27.78 });
    const hms = parseSkyCoord("17:49:41.50 +64:53:14.3")!;
    expect(hms.ra).toBeCloseTo(267.42292, 4);
    expect(hms.dec).toBeCloseTo(64.88731, 4);
    const letters = parseSkyCoord("17h49m41.5s -27d46m48s")!;
    expect(letters.ra).toBeCloseTo(267.42292, 4);
    expect(letters.dec).toBeCloseTo(-27.78, 6);
    const unicode = parseSkyCoord("03h32m38.4s −27°46′48″")!;
    expect(unicode.ra).toBeCloseTo(53.16, 6);
    expect(unicode.dec).toBeCloseTo(-27.78, 6);
    expect(parseSkyCoord("-00:30:00 is not ra")).toBeNull();
    expect(parseSkyCoord("400 0")).toBeNull();
    expect(parseSkyCoord("10 95")).toBeNull();
    expect(parseSkyCoord("NGC 1300")).toBeNull();
  });
});

describe("dates", () => {
  // 2026-09-25T14:03:07Z
  const t = Date.UTC(2026, 8, 25, 14, 3, 7);

  it("accepts epoch seconds, epoch ms, ISO strings and Dates", () => {
    expect(parseTimestamp(t / 1000)?.getTime()).toBe(t);
    expect(parseTimestamp(t)?.getTime()).toBe(t);
    expect(parseTimestamp("2026-09-25T14:03:07Z")?.getTime()).toBe(t);
    expect(parseTimestamp(new Date(t))?.getTime()).toBe(t);
    expect(parseTimestamp("nope")).toBeNull();
    expect(parseTimestamp(null)).toBeNull();
  });

  it("formats in UTC when asked (local time by default)", () => {
    expect(formatDate(t, { utc: true })).toBe("2026-09-25");
    expect(formatDateTime(t, { utc: true })).toBe("2026-09-25 14:03");
    expect(formatDateTime(t, { utc: true, seconds: true })).toBe("2026-09-25 14:03:07");
    expect(formatDateTime(undefined)).toBe(DASH);
  });

  it("describes relative times", () => {
    expect(formatRelative(t - 5_000, t)).toBe("just now");
    expect(formatRelative(t - 90_000, t)).toBe("2 min ago");
    expect(formatRelative(t - 3 * 3600_000, t)).toBe("3 h ago");
    expect(formatRelative(t - 2 * 86400_000, t)).toBe("2 d ago");
    expect(formatRelative(t + 3600_000, t)).toBe("in 1 h");
    expect(formatRelative(null, t)).toBe(DASH);
  });
});
