/* WCS maths against astropy (fixture: __fixtures__/wcs_astropy.json, made
 * with astropy.wcs.WCS(header).all_pix2world(pix, 0) on the same headers). */
import { describe, expect, it } from "vitest";
import ref from "./__fixtures__/wcs_astropy.json";
import { angularSeparationDeg, parseWcs, pixelScaleArcsec, pixToSky, skyToPix, type Wcs } from "./wcs";

type Case = { header: Record<string, number | string>; points: [number, number, number, number][] };
const CASES = ref as unknown as Record<string, Case>;

const raDiff = (a: number, b: number) => Math.abs((((a - b) % 360) + 540) % 360 - 180);

describe("pixel → sky matches astropy", () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(name, () => {
      const w = parseWcs(c.header)!;
      expect(w).not.toBeNull();
      for (const [x, y, ra, dec] of c.points) {
        const s = pixToSky(w, x, y)!;
        expect(raDiff(s.ra, ra) * Math.cos((dec * Math.PI) / 180)).toBeLessThan(1e-9);
        expect(Math.abs(s.dec - dec)).toBeLessThan(1e-9);
        expect(s.ra).toBeGreaterThanOrEqual(0);
        expect(s.ra).toBeLessThan(360);
      }
    });
  }
});

describe("sky → pixel round trip", () => {
  for (const [name, c] of Object.entries(CASES)) {
    it(name, () => {
      const w = parseWcs(c.header)!;
      for (const [x, y, ra, dec] of c.points) {
        const p = skyToPix(w, ra, dec)!;
        expect(Math.abs(p.x - x)).toBeLessThan(1e-6);
        expect(Math.abs(p.y - y)).toBeLessThan(1e-6);
      }
    });
  }
});

describe("header parsing", () => {
  it("accepts the X-Cube-WCS JSON string, CD or PC+CDELT, and CDELT alone", () => {
    const nexus = CASES.tan_cd_nexus.header;
    expect(parseWcs(JSON.stringify(nexus))).toMatchObject({ proj: "TAN", crpix: [-3885.5, 12656.5] });
    const pc = parseWcs(CASES.tan_pc_cdelt.header) as Wcs;
    expect(pc.cd[0][0]).toBeCloseTo(0.8660254037844387 * -2.777777777778e-05, 15);
    expect(pc.cd[0][1]).toBeCloseTo(-0.5 * -2.777777777778e-05, 15);
    const cdelt = parseWcs(CASES.sin_cdelt.header) as Wcs;
    expect(cdelt.cd).toEqual([[-0.0002, 0], [0, 0.0002]]);
  });
  it("rejects missing, malformed, non-celestial and unsupported projections", () => {
    expect(parseWcs(null)).toBeNull();
    expect(parseWcs("")).toBeNull();
    expect(parseWcs("{not json")).toBeNull();
    expect(parseWcs({ CTYPE1: "RA---TAN", CTYPE2: "DEC--TAN" })).toBeNull();          // no CRVAL/CRPIX/scale
    expect(parseWcs({ ...CASES.tan_cd_nexus.header, CTYPE1: "RA---CAR", CTYPE2: "DEC--CAR" })).toBeNull();
    expect(parseWcs({ ...CASES.tan_cd_nexus.header, CD1_1: 0, CD2_2: 0 })).toBeNull(); // singular
  });
  it("swapped axes (DEC first) are read in the right order", () => {
    const h = CASES.tan_cd_rot.header;
    const swapped = {
      CTYPE1: "DEC--TAN", CTYPE2: "RA---TAN", CRVAL1: h.CRVAL2, CRVAL2: h.CRVAL1, CRPIX1: h.CRPIX1, CRPIX2: h.CRPIX2,
      CD1_1: h.CD2_1, CD1_2: h.CD2_2, CD2_1: h.CD1_1, CD2_2: h.CD1_2,
    };
    const a = pixToSky(parseWcs(h)!, 10, 20)!;
    const b = pixToSky(parseWcs(swapped)!, 10, 20)!;
    expect(raDiff(a.ra, b.ra)).toBeLessThan(1e-10);
    expect(Math.abs(a.dec - b.dec)).toBeLessThan(1e-10);
  });
});

describe("helpers", () => {
  it("pixel scale in arcsec from the CD matrix", () => {
    expect(pixelScaleArcsec(parseWcs(CASES.tan_cd_nexus.header)!)).toBeCloseTo(0.03, 9);
    expect(pixelScaleArcsec(parseWcs(CASES.tan_pc_cdelt.header)!)).toBeCloseTo(0.1, 9);
  });
  it("angular separation (haversine)", () => {
    expect(angularSeparationDeg(10, 20, 10, 20)).toBe(0);
    expect(angularSeparationDeg(0, 0, 90, 0)).toBeCloseTo(90, 12);
    expect(angularSeparationDeg(359.9, 0, 0.1, 0)).toBeCloseTo(0.2, 12);
  });
  it("a point behind the TAN plane has no pixel", () => {
    expect(skyToPix(parseWcs(CASES.tan_cd_nexus.header)!, 268.46 + 180, -65.2)).toBeNull();
  });
});
