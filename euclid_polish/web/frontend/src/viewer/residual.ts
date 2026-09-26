/* Client-side residual tiers: A − B, log₂(A / B) and (A − B)/σ for any two
 * loaded tiers on matching grids. A coarser tier is resampled onto the finer
 * grid by an integer factor (block replication, divided by f² so the flux per
 * pixel is conserved: LR e⁻/0.1″-px → e⁻/0.05″-px). σ is the `std` tier
 * (ensemble disagreement) when it is loaded, else the robust (MAD) σ of the
 * difference, per band.
 *
 * Channels are paired BY BAND NAME (X-Cube-Bands; positionally only when a
 * tier has no names), in A's order, and two tiers are refused when they share
 * no band or carry different units (a 4-band e⁻ cube and a 1-band F200W
 * MJy/sr cube are never differenced): `residualMismatch` says why. The values
 * are native (the served Float32), so the display scale plays no part.
 *
 * A residual tier's key is `res:<op>:<a>:<b>` with the tier keys
 * URI-encoded (real-collection tiers such as `m:member:…` contain colons). */
import { robustStats } from "./stats";

export type ResidualOp = "diff" | "ratio" | "chi";
export const RESIDUAL_OPS: readonly ResidualOp[] = ["diff", "ratio", "chi"];

export type Grid = { data: Float32Array; h: number; w: number; c: number; bands?: string[]; unit?: string };

export type Residual = {
  data: Float32Array; h: number; w: number; c: number; bands: string[]; unit: string;
  sigmaSource?: "tier" | "mad";
};

export function residualKey(op: ResidualOp, a: string, b: string): string {
  return `res:${op}:${encodeURIComponent(a)}:${encodeURIComponent(b)}`;
}

export function parseResidualKey(key: string): { op: ResidualOp; a: string; b: string } | null {
  const m = /^res:(diff|ratio|chi):([^:]+):([^:]+)$/.exec(key);
  if (!m) return null;
  try {
    return { op: m[1] as ResidualOp, a: decodeURIComponent(m[2]), b: decodeURIComponent(m[3]) };
  } catch {
    return null;
  }
}

export function residualLabel(op: ResidualOp, a: string, b: string): string {
  if (op === "ratio") return `log₂(${a} / ${b})`;
  if (op === "chi") return `(${a} − ${b}) / σ`;
  return `${a} − ${b}`;
}

/** The finer of two grids when one is an integer multiple of the other. */
export function commonGrid(a: { h: number; w: number }, b: { h: number; w: number }): { h: number; w: number; fa: number; fb: number } | null {
  if (a.h === b.h && a.w === b.w) return { h: a.h, w: a.w, fa: 1, fb: 1 };
  const up = (lo: { h: number; w: number }, hi: { h: number; w: number }) => {
    const f = hi.h / lo.h;
    return Number.isInteger(f) && f > 1 && hi.w === lo.w * f ? f : 0;
  };
  const fa = up(a, b);
  if (fa) return { h: b.h, w: b.w, fa, fb: 1 };
  const fb = up(b, a);
  if (fb) return { h: a.h, w: a.w, fa: 1, fb };
  return null;
}

/** Nearest-pixel upsampling by an integer factor, flux-conserving (÷ f²). */
export function upsample<T extends Grid>(rec: T, f: number): Grid {
  if (f === 1) return rec;
  const h = rec.h * f, w = rec.w * f, c = rec.c;
  const out = new Float32Array(h * w * c);
  const k = 1 / (f * f);
  for (let y = 0; y < h; y++) {
    const sy = Math.floor(y / f);
    for (let x = 0; x < w; x++) {
      const si = (sy * rec.w + Math.floor(x / f)) * c, di = (y * w + x) * c;
      for (let b = 0; b < c; b++) out[di + b] = rec.data[si + b] * k;
    }
  }
  return { ...rec, data: out, h, w };
}

/** A unit string's canonical form ("" when unknown): e-, e⁻, electron(s) are one unit. */
export function canonicalUnit(unit: string | null | undefined): string {
  const u = String(unit ?? "").trim();
  if (/^(e-|e⁻|electrons?)$/i.test(u)) return "e⁻";
  return u;
}

const named = (g: Pick<Grid, "bands" | "c">): string[] | null =>
  g.bands && g.bands.length === g.c && g.bands.every(Boolean) ? g.bands : null;

/** Which channel of A pairs with which of B: by band name (in A's order)
 *  when both tiers name their channels, else by position. */
export function channelPairs(A: Pick<Grid, "bands" | "c">, B: Pick<Grid, "bands" | "c">): { ka: number[]; kb: number[]; bands: string[] } {
  const na = named(A), nb = named(B);
  if (na && nb) {
    const ka: number[] = [], kb: number[] = [], bands: string[] = [];
    na.forEach((name, i) => { const j = nb.indexOf(name); if (j >= 0) { ka.push(i); kb.push(j); bands.push(name); } });
    return { ka, kb, bands };
  }
  const c = Math.min(A.c, B.c);
  const idx = Array.from({ length: c }, (_, i) => i);
  return { ka: idx, kb: idx, bands: (na ?? nb ?? []).slice(0, c) };
}

/** Why A and B cannot be combined (the tail of "<A> and <B> …"), or null. */
export function residualMismatch(A: Grid, B: Grid): string | null {
  if (!commonGrid(A, B)) return "are not on matching grids";
  const ua = canonicalUnit(A.unit), ub = canonicalUnit(B.unit);
  if (ua && ub && ua !== ub) return `are in different units (${ua} vs ${ub})`;
  if (!channelPairs(A, B).bands.length && named(A) && named(B)) {
    return `have no band in common (${(A.bands ?? []).join(", ")} vs ${(B.bands ?? []).join(", ")})`;
  }
  return null;
}

/** The residual of A and B (null when residualMismatch refuses them). */
export function computeResidual(op: ResidualOp, A: Grid, B: Grid, sigma?: Grid | null): Residual | null {
  if (residualMismatch(A, B)) return null;
  const grid = commonGrid(A, B) as NonNullable<ReturnType<typeof commonGrid>>;
  const a = upsample(A, grid.fa), b = upsample(B, grid.fb);
  const { ka, kb, bands } = channelPairs(A, B);
  const c = ka.length;
  const { h, w } = grid;
  const npx = h * w;
  const out = new Float32Array(npx * c);
  const at = (g: Grid, p: number, k: number) => g.data[p * g.c + k];
  if (op === "ratio") {
    for (let p = 0; p < npx; p++) for (let k = 0; k < c; k++) {
      const x = at(a, p, ka[k]), y = at(b, p, kb[k]);
      out[p * c + k] = x > 0 && y > 0 ? Math.log2(x / y) : NaN;
    }
  } else {
    for (let p = 0; p < npx; p++) for (let k = 0; k < c; k++) out[p * c + k] = at(a, p, ka[k]) - at(b, p, kb[k]);
  }
  let sigmaSource: "tier" | "mad" | undefined;
  if (op === "chi") {
    const sg = sigma ? commonGrid(sigma, grid) : null;
    const S = sigma && sg && sg.h === h && sg.w === w && sg.fb === 1 ? upsample(sigma, sg.fa) : null;
    // σ's channel for each output band: by name when σ names its channels,
    // else A's channel index (the std tier is on the SR's band order).
    const sn = S ? named(S) : null;
    const ks = S ? bands.map((name, k) => (sn ? sn.indexOf(name) : ka[k] < S.c ? ka[k] : -1)) : [];
    if (S && c && ks.every((k) => k >= 0)) {
      sigmaSource = "tier";
      for (let p = 0; p < npx; p++) for (let k = 0; k < c; k++) {
        const s = at(S, p, ks[k]);
        out[p * c + k] = s > 0 ? out[p * c + k] / s : NaN;
      }
    } else {
      sigmaSource = "mad";
      for (let k = 0; k < c; k++) {
        const band = new Float32Array(npx);
        for (let p = 0; p < npx; p++) band[p] = out[p * c + k];
        const s = robustStats(band).sigma;
        for (let p = 0; p < npx; p++) out[p * c + k] = s > 0 ? out[p * c + k] / s : NaN;
      }
    }
  }
  const unit = op === "ratio" ? "log₂" : op === "chi" ? "σ" : (A.unit || B.unit || "");
  return { data: out, h, w, c, bands, unit, sigmaSource };
}
