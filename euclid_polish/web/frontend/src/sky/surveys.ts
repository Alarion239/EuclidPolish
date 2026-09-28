/* The remote sky imagery the atlas can show (verified 2026-09-25, all with
 * CORS; see the rework spec §7.1): background HiPS (one at a time), overlay
 * HiPS (any number, with opacity), coverage MOCs and the quick-jump targets.
 * Pure data. HiPS are given by service URL, never by ID (IDs need a
 * MocServer round trip, and the ESASky Euclid HiPS is not registered there). */

export type HipsFormat = "fits" | "png" | "jpeg";

export type HipsColor = {
  colormap?: string;
  stretch?: string;
  minCut?: number;
  maxCut?: number;
  reversed?: boolean;
};

export type BaseSurvey = {
  id: string;
  label: string;
  group: "Euclid" | "All-sky" | "Off";
  /** HiPS service URL; null = no imagery (black background, works offline). */
  url: string | null;
  format?: HipsFormat;
  /** Initial colour settings (FITS tiles: from the HiPS `hips_pixel_cut`). */
  color?: HipsColor;
  credit?: string;
};

const CDS = "https://alasky.cds.unistra.fr";
const Q1 = `${CDS}/Euclid/Q1`;
const EUCLID_CREDIT = "Euclid Q1 (ESA/Euclid Consortium, CC BY-NC 3.0 IGO) via CDS";

export const BASE_SURVEYS: readonly BaseSurvey[] = [
  { id: "q1-color", label: "Euclid Q1 colour", group: "Euclid", url: `${Q1}/CDS_P_Euclid_Q1_color`, format: "png", credit: EUCLID_CREDIT },
  {
    id: "q1-vis", label: "Euclid Q1 VIS", group: "Euclid", url: `${Q1}/CDS_P_Euclid_Q1_VIS`, format: "fits",
    color: { colormap: "grayscale", stretch: "asinh", minCut: -0.0008109, maxCut: 0.04716 }, credit: EUCLID_CREDIT,
  },
  { id: "q1-y", label: "Euclid Q1 NISP Y", group: "Euclid", url: `${Q1}/CDS_P_Euclid_Q1_NISP.Y`, format: "fits", color: { colormap: "grayscale", stretch: "asinh" }, credit: EUCLID_CREDIT },
  { id: "q1-j", label: "Euclid Q1 NISP J", group: "Euclid", url: `${Q1}/CDS_P_Euclid_Q1_NISP.J`, format: "fits", color: { colormap: "grayscale", stretch: "asinh" }, credit: EUCLID_CREDIT },
  { id: "q1-h", label: "Euclid Q1 NISP H", group: "Euclid", url: `${Q1}/CDS_P_Euclid_Q1_NISP.H`, format: "fits", color: { colormap: "grayscale", stretch: "asinh" }, credit: EUCLID_CREDIT },
  {
    id: "esa-vis", label: "ESA Euclid VIS", group: "Euclid",
    url: "https://esdchips.esac.esa.int/Euclid/Q1_MER/VIS/ESAC_P_EUC_VIS/", format: "fits",
    color: { colormap: "grayscale", stretch: "asinh" }, credit: "Euclid Q1 via ESA ESASky",
  },
  { id: "dss2", label: "DSS2 colour", group: "All-sky", url: `${CDS}/DSS/DSSColor`, format: "jpeg", credit: "DSS2 (STScI / Caltech) via CDS" },
  { id: "2mass", label: "2MASS colour", group: "All-sky", url: `${CDS}/2MASS/Color`, format: "jpeg", credit: "2MASS (UMass / IPAC-Caltech) via CDS" },
  { id: "panstarrs", label: "Pan-STARRS DR1", group: "All-sky", url: `${CDS}/Pan-STARRS/DR1/color-z-zg-g`, format: "jpeg", credit: "Pan-STARRS1 DR1 via CDS" },
  { id: "unwise", label: "unWISE", group: "All-sky", url: `${CDS}/unWISE/color-W2-W1W2-W1`, format: "jpeg", credit: "unWISE (WISE / NEOWISE) via CDS" },
  { id: "desi", label: "DESI Legacy DR10", group: "All-sky", url: `${CDS}/DESI-legacy-surveys/DR10/CDS_P_DESI-Legacy-Surveys_DR10_color`, format: "png", credit: "DESI Legacy Imaging Surveys DR10 via CDS" },
  { id: "none", label: "None (black · offline)", group: "Off", url: null },
];

export const DEFAULT_BASE = "q1-color";

export type OverlaySurvey = {
  id: string;
  label: string;
  group: "JWST · ESA" | "JWST · CDS";
  url: string;
  format?: HipsFormat;
  defaultOpacity: number;
};

const ESA_JWST = "https://cdn.skies.esac.esa.int/JWST";

export const OVERLAY_SURVEYS: readonly OverlaySurvey[] = [
  { id: "jwst-nircam", label: "JWST NIRCam", group: "JWST · ESA", url: `${ESA_JWST}/NIRCam_Imaging/`, defaultOpacity: 0.8 },
  { id: "jwst-niriss", label: "JWST NIRISS", group: "JWST · ESA", url: `${ESA_JWST}/NIRISS_Imaging/`, defaultOpacity: 0.8 },
  { id: "jwst-miri", label: "JWST MIRI", group: "JWST · ESA", url: `${ESA_JWST}/MIRI_Imaging/`, defaultOpacity: 0.8 },
  ...["F115W", "F150W", "F200W", "F210M", "F444W"].map((f) => ({
    id: `cds-${f.toLowerCase()}`, label: `JWST ${f}`, group: "JWST · CDS" as const,
    url: `${CDS}/JWST/CDS_P_JWST_${f}`, defaultOpacity: 0.8,
  })),
];

export type CoverageMoc = { id: string; label: string; url: string; description: string };

export const COVERAGE_MOCS: readonly CoverageMoc[] = [
  {
    id: "moc-q1", label: "Euclid Q1 coverage", url: `${Q1}/CDS_P_Euclid_Q1_VIS/Moc.fits`,
    description: "Footprint of the Euclid Q1 VIS HiPS (CDS MOC).",
  },
  {
    id: "moc-jwst", label: "JWST HiPS coverage",
    url: `${CDS}/MocServer/query?expr=ID%3DESAVO%2FP%2FJWST%2F*&get=moc&fmt=fits`,
    description: "Union of the ESA JWST HiPS (2023–2025; misses NEXUS — use JWST discovery for current coverage).",
  },
];

export type QuickJump = { id: string; label: string; ra: number; dec: number; fov: number };

export const QUICK_JUMPS: readonly QuickJump[] = [
  { id: "edf-n", label: "EDF-N", ra: 269.733, dec: 66.018, fov: 14 },
  { id: "edf-s", label: "EDF-S", ra: 61.241, dec: -48.423, fov: 14 },
  { id: "edf-f", label: "EDF-F", ra: 52.932, dec: -28.088, fov: 10 },
  { id: "ldn1641", label: "LDN1641", ra: 85.761, dec: -8.437, fov: 2.5 },
  { id: "nexus", label: "NEXUS", ra: 268.4615, dec: 65.1964, fov: 0.45 },
  { id: "poster", label: "Poster galaxy", ra: 273.2309, dec: 68.3637, fov: 0.07 },
];

export const PROJECTIONS = ["MOL", "AIT", "SIN", "TAN"] as const;
export type Projection = (typeof PROJECTIONS)[number];
export const DEFAULT_VIEW = { ra: 165, dec: 0, fov: 360, proj: "MOL" as Projection };

export function baseSurvey(id: string): BaseSurvey {
  return BASE_SURVEYS.find((b) => b.id === id) ?? BASE_SURVEYS[0];
}

export function overlaySurvey(id: string): OverlaySurvey | undefined {
  return OVERLAY_SURVEYS.find((o) => o.id === id);
}

/** Aladin's colormaps (3.8) and stretches, for the Display panel's Sky section. */
export const ALADIN_COLORMAPS = [
  "native", "grayscale", "viridis", "magma", "inferno", "plasma", "cividis", "cubehelix", "eosb",
  "parula", "rainbow", "rdbu", "rdylbu", "redtemperature", "sinebow", "spectral", "summer",
  "ylgnbu", "ylorbr", "blues", "red", "green", "blue",
] as const;
export const ALADIN_STRETCHES = ["linear", "asinh", "log", "sqrt", "pow"] as const;
