/* Synthetic › PSF: the chain real Euclid stars → their cutouts → the
   empirical PSFs, as pure helpers (psfModel.test.ts): the catalogue's
   headline counts, the magnitude-window caption, one 100% stacked validity
   bar per band, the target star's marker, the "which PSF does generation
   use" line, the ePSF vs Gaussian FWHM rows and the cluster FWHM map. */
import { formatCount, formatNumber } from "../../../format";
import type { Tone } from "../../../ui";
import type { PsfBand, PsfCluster, PsfGeneration, StarsPayload } from "../dataApi";
import { bandShort, quantileEdges, type BandState, type ScatterGroup } from "../dataModel";

type Summary = NonNullable<StarsPayload["summary"]>;

/** The header's numbers: every star, and the usable ones (valid in all four
 *  bands at the navigator's one cutout size). */
export function catalogueHeadline(summary: Summary | null | undefined): { total: string; usable: string; size: number | null } | null {
  if (!summary) return null;
  return { total: formatCount(summary.total), usable: formatCount(summary.navigator.count), size: summary.navigator.size };
}

const num = (v: unknown): number | null => {
  const n = typeof v === "number" ? v : typeof v === "string" && v.trim() ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
};
const mag = (v: number) => formatNumber(v, { digits: Number.isInteger(v) ? 0 : 1 });

/** What the magnitude histogram's edges are: the euclid_query windows, with
 *  the last run's settings when the step registry knows them. */
export function magnitudeWindowCaption(last: Record<string, unknown> | null | undefined): string {
  const base = "Each euclid_query run keeps the brightest point sources inside its own VIS window above an S/N cut, "
    + "so the edges in the histogram are those windows' edges";
  if (!last) return `${base}.`;
  const n = num(last.num_stars), lo = num(last.magnitude_min), hi = num(last.magnitude_limit), snr = num(last.snr_min);
  const window = lo != null && hi != null ? `VIS ${mag(lo)}–${mag(hi)}` : hi != null ? `VIS ≤ ${mag(hi)}` : lo != null ? `VIS ≥ ${mag(lo)}` : null;
  const parts = [n != null ? `brightest ${formatCount(n)}` : null, window, snr != null ? `S/N ≥ ${mag(snr)}` : null].filter(Boolean);
  return parts.length ? `${base} (last run: ${parts.join(", ")}).` : `${base}.`;
}

const STATES: BandState[] = ["valid", "corrupted", "failed", "pending"];

export type ValidityBar = {
  band: string; total: number; label: string;
  parts: { state: BandState; count: number; fraction: number }[];
};

/** One 100% stacked bar per band: each star counts once per band, under its
 *  best outcome (valid › corrupted › failed › pending); empty parts dropped. */
export function validityBars(stats: StarsPayload["band_stats"]): ValidityBar[] {
  return stats.map((s) => {
    const total = STATES.reduce((n, st) => n + (s[st] ?? 0), 0);
    const parts = STATES.filter((st) => (s[st] ?? 0) > 0)
      .map((state) => ({ state, count: s[state], fraction: total ? s[state] / total : 0 }));
    const band = bandShort(s.band);
    return { band, total, parts, label: `${band}: ${parts.map((p) => `${formatCount(p.count)} ${p.state}`).join(", ")}` };
  });
}

/** The catalogue target at the centre of its cutout (a star cross). */
export function starMarker(size: number | null | undefined, id: string | null | undefined) {
  if (!size || !id) return null;
  const c = (size - 1) / 2;
  return {
    grid: { width: size, height: size },
    items: [{ key: id, x: Math.floor(c), y: Math.floor(c), r: 14, kind: "star", title: `Star ${id} (the catalogue target)` }],
  };
}

/** "Used by the last generation run": the kernel of each band as the records'
 *  provenance recorded it (`lead` names the run), else — records generated
 *  before the stamp, or none synced — what the local copy of the FASRC ePSFs
 *  says generation would use: empirical everywhere, a Gaussian fallback where
 *  FASRC has none, or unknown here when nothing is synced. */
export function generationPsfLine(bands: readonly PsfBand[], generation?: PsfGeneration | null):
  { tone: Tone; text: string; lead: string } {
  const recorded = generation ? recordedPsfLine(generation) : null;
  if (recorded) return { ...recorded, lead: "Used by the last generation run" };
  return { ...syncedPsfLine(bands), lead: "Used by generation" };
}

function recordedPsfLine(generation: PsfGeneration): { tone: Tone; text: string } | null {
  const entries = Object.entries(generation.psf_kinds ?? {});
  if (!entries.length) return null;
  const empirical = entries.filter(([, k]) => k === "empirical").map(([b]) => bandShort(b));
  const fallback = entries.filter(([, k]) => k !== "empirical").map(([b]) => bandShort(b));
  const split = `${generation.subset} records`;
  if (!fallback.length) return { tone: "good", text: `empirical ePSFs in every band (${empirical.join(", ")}; ${split})` };
  return { tone: "warn", text: `Gaussian fallback in ${fallback.join(", ")}${empirical.length ? `; empirical in ${empirical.join(", ")}` : ""} (${split})` };
}

function syncedPsfLine(bands: readonly PsfBand[]): { tone: Tone; text: string } {
  const by = (state: PsfBand["state"]) => bands.filter((b) => b.state === state).map((b) => bandShort(b.name));
  const empirical = by("empirical"), fallback = by("no_empirical"), uncached = by("not_cached");
  if (bands.length && empirical.length === bands.length) return { tone: "good", text: `Empirical ePSFs in every band (${empirical.join(", ")})` };
  if (fallback.length) {
    const rest = [empirical.length ? `empirical in ${empirical.join(", ")}` : null,
      uncached.length ? `${uncached.join(", ")} not synced here` : null].filter(Boolean);
    return { tone: "warn", text: `Gaussian fallback in ${fallback.join(", ")}${rest.length ? `; ${rest.join("; ")}` : ""}` };
  }
  if (empirical.length) return { tone: "neutral", text: `Empirical in ${empirical.join(", ")}; ${uncached.join(", ")} not synced to this machine` };
  return { tone: "neutral", text: "unknown here, the ePSFs are not synced to this machine (generation on FASRC reads its own)" };
}

export type FwhmRow = { band: string; epsf: string; gaussian: string; state: string };

/** ePSF vs Gaussian-fallback FWHM per band (arcsec, 3 decimals). */
export function psfFwhmRows(bands: readonly PsfBand[]): FwhmRow[] {
  const f = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(3));
  const state = { empirical: "Empirical", no_empirical: "Gaussian fallback", not_cached: "Not synced here" } as const;
  return bands.map((b) => ({ band: bandShort(b.name), epsf: f(b.measured_fwhm), gaussian: f(b.fwhm), state: state[b.state] }));
}

/** The clusters on the sky (RA, Dec), grouped into FWHM quintiles of one
 *  band (+ "no FWHM"); `domain` is that band's FWHM range. */
export function clusterMapGroups(clusters: readonly PsfCluster[], band: string): {
  groups: (ScatterGroup & { lo?: number; hi?: number })[]; domain: [number, number] | null;
} {
  const placed = clusters.filter((c) => c.ra != null && c.dec != null && Number.isFinite(c.ra) && Number.isFinite(c.dec));
  const values = placed.map((c) => c.fwhm_by_band[band]).filter((v): v is number => v != null && Number.isFinite(v));
  const edges = quantileEdges(values, Math.min(5, Math.max(1, values.length)));
  const n = Math.max(0, edges.length - 1);
  const f3 = (v: number) => v.toFixed(3);
  const groups: (ScatterGroup & { lo?: number; hi?: number })[] = Array.from({ length: n }, (_x, i) => ({
    key: `q${i}`, label: `${f3(edges[i])}–${f3(edges[i + 1])}″`, t: n === 1 ? 0.5 : i / (n - 1), x: [], y: [], ids: [],
    lo: edges[i], hi: edges[i + 1],
  }));
  const none: ScatterGroup = { key: "none", label: "no FWHM", t: -1, x: [], y: [], ids: [] };
  for (const c of placed) {
    const v = c.fwhm_by_band[band];
    let g: ScatterGroup = none;
    if (v != null && Number.isFinite(v) && groups.length) {
      let k = groups.length - 1;
      for (let j = 1; j < edges.length - 1; j++) if (v < edges[j]) { k = j - 1; break; }
      g = groups[k];
    }
    g.x.push(c.ra as number); g.y.push(c.dec as number); g.ids.push(c.index);
  }
  const shown = [...groups.filter((g) => g.x.length), ...(none.x.length ? [none] : [])];
  return { groups: shown, domain: values.length ? [edges[0], edges[edges.length - 1]] : null };
}
