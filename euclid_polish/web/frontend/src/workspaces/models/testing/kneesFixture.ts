/* Fixtures of Models › Knees (the knee-attribution API, spec 2026-10-15),
 * imported by tests only: knees.test.ts and kneesTab.test.tsx.
 * One finished run ("20261015-101500-abc123", record version 3) of two
 * universal models: #222 at eight knees (10⁻³ … 10⁴ e⁻) and #221 at six
 * (0.1 … 10⁴ e⁻), over two synthetic test fields and the poster galaxy.
 * Shares are deterministic: a pixel of brightness b leans on the knee nearest
 * b (a Gaussian over the knee index); the ablation split is the gradient
 * split reversed, so the two disagree visibly. Each input band's own split
 * (`knee_share_by_input`) tilts the all-input split: VIS toward the brighter
 * knees, Y toward the fainter ones, J and H equal to it; with the band shares
 * [0.4, 0.4, 0.1, 0.1] the identity knee_share = Σ_c band_share_c ·
 * knee_share_by_input_c holds. The local (3×3) splits lean on the knee
 * nearest 10·b, sharper (`localSplit`), with the same tilt per input and the
 * band shares [0.45, 0.45, 0.05, 0.05], so the identity holds for them too.
 * VIS has no `saturated` class (n = 0 is never drawn); the brightness bins
 * below 10⁻² e⁻ and above 10⁵·⁵ e⁻ are empty (null). `v2Record()` is the
 * same run before the local split, `v1Record()` before the noise weighting. */
import type {
  ClassShares, KneeAggregate, KneeField, KneeModelChoice, KneeModelResult, KneeRunRecord, KneeRunSummary,
} from "../kneesApi";

export const RUN_ID = "20261015-101500-abc123";
export const OLD_RUN_ID = "20261014-090000-def456";
export const KNEES_8 = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000];
export const KNEES_6 = [0.1, 1, 10, 100, 1000, 10000];
export const EDGES = Array.from({ length: 21 }, (_, i) => +(10 ** (-3 + i / 2)).toPrecision(6));

const round = (v: number) => Math.round(v * 1e4) / 1e4;

/** A share vector over `knees` peaked at the knee nearest `brightness`. */
export function peaked(knees: readonly number[], brightness: number, width = 1): number[] {
  const lb = Math.log10(Math.max(brightness, 1e-6));
  let c = 0;
  knees.forEach((q, k) => { if (Math.abs(Math.log10(q) - lb) < Math.abs(Math.log10(knees[c]) - lb)) c = k; });
  const w = knees.map((_, k) => Math.exp(-((k - c) ** 2) / (2 * width * width)));
  const s = w.reduce((a, b) => a + b, 0);
  return w.map((v) => round(v / s));
}

/** The class band shares (input VIS, Y, J, H): VIS and Y equal, so the
 *  opposite tilts of their own splits cancel in the all-input split. */
export const BAND_SHARE = [0.4, 0.4, 0.1, 0.1];

/** Each input band's own split (C × K rows: VIS, Y, J, H) around the
 *  all-input split `g`: g·(1 ± tilt·s) with s ∈ [−1, 1] the knee index's
 *  offset from g's mean index, so every row sums to Σ g and stays ≥ 0. */
export function byInput(g: readonly number[], tilt = 0.9): number[][] {
  const total = g.reduce((a, b) => a + b, 0) || 1;
  const mean = g.reduce((a, v, k) => a + v * k, 0) / total;
  const spread = Math.max(...g.map((_, k) => Math.abs(k - mean))) || 1;
  const lean = (sign: number) => g.map((v, k) => round(v * (1 + (sign * tilt * (k - mean)) / spread)));
  return [lean(1), lean(-1), [...g], [...g]];
}

/** The local (3×3) class band shares: VIS and Y equal again. */
export const LOCAL_BAND_SHARE = [0.45, 0.45, 0.05, 0.05];

/** The local (3×3) knee split of a pixel of brightness b: peaked at the knee
 *  nearest 10·b, sharper than the receptive field's. */
export const localSplit = (knees: readonly number[], brightness: number): number[] => peaked(knees, brightness * 10, 0.6);

/** The 4×4 local band shares (rows: output band, cols: input band). */
export const CROSS_LOCAL = [[0.93, 0.03, 0.02, 0.02], [0.12, 0.8, 0.04, 0.04], [0.1, 0.05, 0.8, 0.05], [0.08, 0.04, 0.04, 0.84]];

const SYN_BRIGHT: Record<string, number> = {
  saturated: 5e4, "star core": 2e3, "star wing": 20, "galaxy core": 300, "galaxy outskirts": 8, faint: 0.3, sky: 0.02,
};
const REAL_BRIGHT: Record<string, number> = { sky: 0.5, faint: 5, mid: 30, bright: 150, core: 3000, blank: 0 };

function classes(knees: readonly number[], bright: Record<string, number>, skip: string[] = []): Record<string, ClassShares> {
  const out: Record<string, ClassShares> = {};
  Object.entries(bright).forEach(([cls, b], i) => {
    if (skip.includes(cls)) return;
    const g = peaked(knees, b || 1e-3);
    const l = localSplit(knees, b || 1e-3);
    out[cls] = {
      knee_share: g, ablation_share: [...g].reverse(), band_share: [...BAND_SHARE], knee_share_by_input: byInput(g),
      knee_share_local: l, band_share_local: [...LOCAL_BAND_SHARE], knee_share_by_input_local: byInput(l), n: 24 + 2 * i,
    };
  });
  return out;
}

export function aggregate(knees: readonly number[], set: "synthetic" | "real"): KneeAggregate {
  const bright = set === "synthetic" ? SYN_BRIGHT : REAL_BRIGHT;
  const byClass: KneeAggregate["by_class"] = {};
  const bands: KneeAggregate["by_brightness"]["bands"] = {};
  for (let b = 0; b < 4; b++) {
    byClass[String(b)] = classes(knees, bright, b === 0 && set === "synthetic" ? ["saturated"] : []);
    const shares: (number[] | null)[] = [];
    const inputs: (number[][] | null)[] = [];
    const local: (number[] | null)[] = [];
    const localInputs: (number[][] | null)[] = [];
    const n: number[] = [];
    for (let i = 0; i < 20; i++) {
      const centre = Math.sqrt(EDGES[i] * EDGES[i + 1]);
      const empty = i < 2 || i > 16;
      shares.push(empty ? null : peaked(knees, centre));
      inputs.push(empty ? null : byInput(peaked(knees, centre)));
      local.push(empty ? null : localSplit(knees, centre));
      localInputs.push(empty ? null : byInput(localSplit(knees, centre)));
      n.push(empty ? 0 : 10 + i);
    }
    bands[String(b)] = { knee_share: shares, knee_share_by_input: inputs, knee_share_local: local, knee_share_by_input_local: localInputs, n };
  }
  return {
    by_class: byClass,
    by_brightness: { edges_e: EDGES, bands },
    cross_band: [[0.82, 0.06, 0.06, 0.06], [0.3, 0.5, 0.1, 0.1], [0.25, 0.1, 0.55, 0.1], [0.2, 0.1, 0.1, 0.6]],
    cross_band_local: CROSS_LOCAL.map((row) => [...row]),
    n_samples: 400,
  };
}

/** One field: four VIS samples (classes at known brightness) and two Y. */
export function field(ref: string, kind: "synthetic" | "real", knees: readonly number[], image: string | null): KneeField {
  const rows: [number, number, number, string, number][] = kind === "synthetic"
    ? [[0, 10, 12, "galaxy core", 300], [0, 40, 50, "sky", 0.02], [0, 60, 30, "star core", 2000], [0, 100, 90, "faint", 0.3],
      [1, 20, 20, "galaxy outskirts", 8], [1, 70, 80, "star wing", 20]]
    : [[0, 64, 64, "core", 3000], [0, 5, 7, "sky", 0.5], [0, 30, 90, "mid", 30], [0, 90, 12, "bright", 150],
      [1, 50, 50, "faint", 5], [1, 10, 100, "mid", 30]];
  const g = rows.map((r) => peaked(knees, r[4]));
  return {
    ref, kind, shape: [128, 128], image, noise_sigma_e: kind === "synthetic" ? [17.6, 0.87, 0.98, 0.9] : [12.4, 0.81, 0.93, 0.88],
    samples: {
      band: rows.map((r) => r[0]), y: rows.map((r) => r[1]), x: rows.map((r) => r[2]), class: rows.map((r) => r[3]),
      value_e: rows.map((r) => r[4]), knee_share: g, ablation_share: g.map((s) => [...s].reverse()),
      band_share: rows.map(() => [0.7, 0.1, 0.1, 0.1]), sensitivity: rows.map((_, i) => 0.01 * (i + 1)),
      knee_share_local: rows.map((r) => localSplit(knees, r[4])),
    },
  };
}

const modelSlug = (spec: string) => spec.replace(/[^A-Za-z0-9._-]+/g, "__");

function result(spec: string, knees: readonly number[], depth: number, withImages = true): KneeModelResult {
  const img = (ref: string) => (withImages ? `/api/knee-attribution/${RUN_ID}/field/${modelSlug(spec)}/${ref.replace(/[^A-Za-z0-9._-]+/g, "__")}.png` : null);
  return {
    knees: [...knees], bands: ["VIS", "Y_E", "J_E", "H_E"], depth,
    aggregate: { synthetic: aggregate(knees, "synthetic"), real: aggregate(knees, "real") },
    fields: [
      field("test/0", "synthetic", knees, img("test/0")),
      field("test/1", "synthetic", knees, img("test/1")),
      field("poster/target_181255", "real", knees, img("poster/target_181255")),
    ],
  };
}

export const record = (patch: Partial<KneeRunRecord> = {}): KneeRunRecord => ({
  version: 3, id: RUN_ID, created: "2026-10-15T10:15:00Z", finished: "2026-10-15T10:21:00Z", status: "done",
  params: { models: ["universal:member_222", "universal:member_221"], synthetic: 2, tiles: ["poster/target_181255"], per_class: 24, seed: 0 },
  knee_grid_note: null,
  models: {
    "universal:member_222": result("universal:member_222", KNEES_8, 16),
    "universal:member_221": result("universal:member_221", KNEES_6, 16, false),
  },
  errors: {},
  ...patch,
});

/** `rec` without the keys `drop`, at any depth. */
function without(rec: KneeRunRecord, drop: readonly string[]): KneeRunRecord {
  const strip = (v: unknown): unknown => (Array.isArray(v) ? v.map(strip)
    : v && typeof v === "object" ? Object.fromEntries(Object.entries(v).filter(([k]) => !drop.includes(k)).map(([k, x]) => [k, strip(x)]))
      : v);
  return strip(rec) as KneeRunRecord;
}

const LOCAL_KEYS = ["knee_share_local", "band_share_local", "knee_share_by_input_local", "cross_band_local"];

/** The same run as a version-2 record: no local (3×3) splits. */
export const v2Record = (): KneeRunRecord => ({ ...without(record(), LOCAL_KEYS), version: 2 });

/** The same run as a version-1 record (per-electron weights): no version,
 *  no per-input or local split, no noise σ. */
export const v1Record = (): KneeRunRecord => without(record(), ["version", "knee_share_by_input", "noise_sigma_e", ...LOCAL_KEYS]);

export const runs = (): { runs: KneeRunSummary[] } => ({
  runs: [
    { id: RUN_ID, created: "2026-10-15T10:15:00Z", finished: "2026-10-15T10:21:00Z", status: "done",
      models: ["universal:member_222", "universal:member_221"], synthetic: 2, tiles: ["poster/target_181255"], per_class: 24 },
    { id: OLD_RUN_ID, created: "2026-10-14T09:00:00Z", finished: "2026-10-14T09:01:00Z", status: "failed",
      models: ["universal:member_211"], synthetic: 5, tiles: [], per_class: 24 },
  ],
});

export const failedRecord = (): KneeRunRecord => ({
  version: 3, id: OLD_RUN_ID, created: "2026-10-14T09:00:00Z", finished: "2026-10-14T09:01:00Z", status: "failed",
  params: { models: ["universal:member_211"], synthetic: 5, tiles: [], per_class: 24, seed: 0 },
  models: {}, errors: { "universal:member_211": "OSError: no checkpoint for member_211" },
});

export const catalogue = (): { models: KneeModelChoice[] } => ({
  models: [
    { spec: "universal:member_211", label: "Universal 211", knees: KNEES_6, depth: 16, kind: "universal", available: true },
    { spec: "universal:member_221", label: "Universal 221", knees: KNEES_6, depth: 16, kind: "universal", available: true },
    { spec: "universal:member_222", label: "Universal 222", knees: KNEES_8, depth: 16, kind: "universal", available: true },
    { spec: "universal:member_223", label: "Universal 223", knees: KNEES_8, depth: 16, kind: "universal", available: true },
    { spec: "universal:member_224", label: "Universal 224", knees: [], depth: 16, kind: "universal", available: false,
      reason: "no checkpoint: pull it from FASRC" },
    { spec: "member:member_196", label: "Member 196·psnr", knees: KNEES_6, depth: 32, kind: "member", available: true },
    { spec: "member:member_178", label: "Member 178·psnr", knees: [10], depth: 32, kind: "member", available: false,
      reason: "a single-knee model (nothing to split over knees)" },
  ],
});

export const SOURCES = { sources: [
  { id: "nexus", label: "NEXUS", count: 2 }, { id: "tile", label: "Cached tiles", count: 0 },
  { id: "eval", label: "Eval", count: 3 }, { id: "poster", label: "Poster", count: 1 },
] };
export const POSTER_TILES = { source: "poster", label: "Poster", count: 1, tiles: [
  { source: "poster", id: "target_181255", ref: "poster/target_181255", label: "Poster galaxy", ra: 273.2, dec: 68.4 },
] };
export const NEXUS_TILES = { source: "nexus", label: "NEXUS", count: 2, tiles: [
  { source: "nexus", id: "f200w-0001", ref: "nexus/f200w-0001", label: "NEXUS tile 1", ra: 268.4, dec: 65.1 },
  { source: "nexus", id: "f200w-0002", ref: "nexus/f200w-0002", label: "NEXUS tile 2", ra: 268.5, dec: 65.2 },
] };
