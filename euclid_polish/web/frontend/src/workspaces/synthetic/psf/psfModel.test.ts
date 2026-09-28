import { describe, expect, it } from "vitest";
import type { PsfBand, PsfCluster, StarsPayload } from "../dataApi";
import {
  catalogueHeadline, clusterMapGroups, generationPsfLine, magnitudeWindowCaption, psfFwhmRows, starMarker, validityBars,
} from "./psfModel";

type Summary = NonNullable<StarsPayload["summary"]>;
const SUMMARY: Summary = {
  total: 43401, valid: 38599, valid_all4: 18062, navigator: { count: 17917, size: 511 },
  corrupted: 4802, failed: 0, pending: 0, mag_min: 16.0002, mag_max: 19.0054,
};

describe("PSF › catalogue", () => {
  it("opens with the stars and the usable ones (valid in all four bands at one size)", () => {
    expect(catalogueHeadline(SUMMARY)).toEqual({ total: "43,401", usable: "17,917", size: 511 });
    expect(catalogueHeadline(null)).toBeNull();
  });

  it("explains the magnitude windows from the last euclid_query run", () => {
    expect(magnitudeWindowCaption({ num_stars: 10000, magnitude_min: 18, magnitude_limit: 19, snr_min: 50 }))
      .toBe("Each euclid_query run keeps the brightest point sources inside its own VIS window above an S/N cut, "
        + "so the edges in the histogram are those windows' edges (last run: brightest 10,000, VIS 18–19, S/N ≥ 50).");
    expect(magnitudeWindowCaption(null)).toMatch(/^Each euclid_query run keeps .*windows' edges\.$/);
    expect(magnitudeWindowCaption({ num_stars: 500, magnitude_min: "", magnitude_limit: 19.5 }))
      .toMatch(/\(last run: brightest 500, VIS ≤ 19.5\)\.$/);
  });
});

describe("PSF › cutouts", () => {
  it("draws one 100% stacked bar per band, in the state order valid › corrupted › failed › pending", () => {
    const bars = validityBars([
      { band: "VIS", valid: 38049, corrupted: 5352, failed: 0, pending: 0, by_size: {} },
      { band: "Y_E", valid: 26743, corrupted: 16658, failed: 0, pending: 0, by_size: {} },
    ]);
    expect(bars.map((b) => b.band)).toEqual(["VIS", "Y"]);
    expect(bars[0].parts.map((p) => p.state)).toEqual(["valid", "corrupted"]);   // zero parts dropped
    expect(bars[0].parts[0].fraction + bars[0].parts[1].fraction).toBeCloseTo(1, 9);
    expect(bars[1].total).toBe(43401);
    expect(bars[1].label).toBe("Y: 26,743 valid, 16,658 corrupted");
  });

  it("marks the target star at the centre of the cutout", () => {
    expect(starMarker(511, "12")).toEqual({ grid: { width: 511, height: 511 }, items: [{ key: "12", x: 255, y: 255, r: 14, kind: "star", title: "Star 12 (the catalogue target)" }] });
    expect(starMarker(null, "12")).toBeNull();
    expect(starMarker(511, null)).toBeNull();
  });
});

const band = (name: string, state: PsfBand["state"], fwhm: number, measured?: number): PsfBand =>
  ({ name, fwhm, oversampling: 2, epsf_pixel_scale: 0.05, state, empirical: state === "empirical", measured_fwhm: measured } as PsfBand);

describe("PSF › ePSF", () => {
  it("says which kernels generation uses, from the synced ePSFs", () => {
    expect(generationPsfLine([band("VIS", "empirical", 0.16, 0.17), band("Y_E", "empirical", 0.4, 0.41)]))
      .toEqual({ tone: "good", lead: "Used by generation", text: "Empirical ePSFs in every band (VIS, Y)" });
    expect(generationPsfLine([band("VIS", "empirical", 0.16, 0.17), band("J_E", "no_empirical", 0.45), band("H_E", "no_empirical", 0.48)]))
      .toEqual({ tone: "warn", lead: "Used by generation", text: "Gaussian fallback in J, H; empirical in VIS" });
    expect(generationPsfLine([band("VIS", "not_cached", 0.16), band("Y_E", "not_cached", 0.4)]).tone).toBe("neutral");
    expect(generationPsfLine([band("VIS", "not_cached", 0.16)]).text).toMatch(/not synced to this machine/);
  });

  it("leads with what the last generation run recorded, when its records carry it", () => {
    const bands = [band("VIS", "not_cached", 0.16)];
    expect(generationPsfLine(bands, { subset: "test", psf_kinds: { VIS: "empirical", Y_E: "empirical", J_E: "empirical", H_E: "empirical" } }))
      .toEqual({ tone: "good", lead: "Used by the last generation run", text: "empirical ePSFs in every band (VIS, Y, J, H; test records)" });
    expect(generationPsfLine(bands, { subset: "validate", psf_kinds: { VIS: "empirical", Y_E: "gaussian" } }))
      .toEqual({ tone: "warn", lead: "Used by the last generation run", text: "Gaussian fallback in Y; empirical in VIS (validate records)" });
    // records generated before the stamp: the synced ePSFs speak instead
    expect(generationPsfLine(bands, null).lead).toBe("Used by generation");
    expect(generationPsfLine(bands, { subset: "test", psf_kinds: {} }).lead).toBe("Used by generation");
  });

  it("compares the ePSF with the Gaussian fallback FWHM per band", () => {
    const rows = psfFwhmRows([band("VIS", "empirical", 0.16, 0.1712), band("J_E", "no_empirical", 0.45)]);
    expect(rows).toEqual([
      { band: "VIS", epsf: "0.171", gaussian: "0.160", state: "Empirical" },
      { band: "J", epsf: "—", gaussian: "0.450", state: "Gaussian fallback" },
    ]);
  });

  it("maps the clusters on the sky coloured by their FWHM in one band", () => {
    const c = (index: number, ra: number, dec: number, f: number | null): PsfCluster =>
      ({ index, id: `c${index}`, ra, dec, n_stars: 400, fwhm_by_band: f == null ? {} : { VIS: f } } as PsfCluster);
    const { groups, domain } = clusterMapGroups([c(0, 10, 1, 0.16), c(1, 11, 2, 0.18), c(2, 12, 3, null), c(3, null as never, 3, 0.2)], "VIS");
    expect(groups.flatMap((g) => g.ids).sort()).toEqual([0, 1, 2]);        // no position → not on the map
    expect(groups.find((g) => g.key === "none")?.ids).toEqual([2]);
    expect(domain).toEqual([0.16, 0.18]);
  });
});
