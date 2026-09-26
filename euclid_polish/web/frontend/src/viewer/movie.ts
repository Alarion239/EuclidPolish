/* The ensemble-disagreement movie (the `morph` tier), pure and worker-friendly.
 *
 * A fixed loop of MORPH_FRAMES frames, periodic in φ ∈ [0, 1):
 *   frame(φ) = centre + Σ_k amp_k · morphAmp · sin(2π f_k φ + φ_k) · PC_k
 * with f = [1, 2, 3] and φ_k = [0, π/2, π/3] (k mod 3). The centre is the tier
 * the PCs are components about, `meta.morph_base_tier` (the ensemble's
 * "mean"; C6), falling back to "sr". Amplitudes come from each PC cube's
 * X-Cube-Amp header (subset-aware), else `meta.pca_amps[index]`.
 *
 * The viewer pre-renders every frame's colour prepare once (the expensive
 * half) into a MovieStore entry, then each animation tick re-runs only the
 * cheap transfer, so the brightness sliders rescale the playing movie live. */

export const MORPH_FRAMES = 48;
export const FRQ = [1, 2, 3] as const;
export const MPH = [0, Math.PI / 2, Math.PI / 3] as const;
/** Fields cached ahead in each direction while a movie plays. */
export const MOVIE_RADIUS = 10;
/** ~1.4 GB cap on the cached movies. */
export const MOVIE_BUDGET_BYTES = 1.4e9;

export type MovieMeta = { morph_base_tier?: string; pca_n?: number; pca_max?: number; pca_amps?: number[][] };

/** The movie's centre tier. */
export function morphBaseTier(meta: MovieMeta | null | undefined): string {
  return (meta && meta.morph_base_tier) || "sr";
}

/** Number of PCA components to animate: all of the full ensemble's
 *  (`pca_n`), or members − 1 (at most `pca_max`, default 3) for a subset. */
export function pcaCount(meta: MovieMeta | null | undefined, subset: string | null): number {
  if (subset) {
    const nSub = subset.split(",").filter(Boolean).length;
    const pcaMax = (meta && meta.pca_max) || 3;
    return Math.max(0, Math.min(pcaMax, nSub - 1));
  }
  return ((meta && meta.pca_n) || 0) | 0;
}

/** Per-component weights at phase φ. */
export function morphCoefficients(amps: number[], morphAmp: number, phase: number): number[] {
  return amps.map((a, k) => {
    const c = (Number.isFinite(a) ? a : 0) * morphAmp * Math.sin(2 * Math.PI * FRQ[k % 3] * phase + MPH[k % 3]);
    return Number.isFinite(c) ? c + 0 : 0;   // (+ 0: no -0)
  });
}

/** out = base + Σ coeffs[k] · comps[k]. */
export function synthesizeMorphFrame(base: Float32Array, comps: Float32Array[], coeffs: number[], out: Float32Array): Float32Array {
  out.set(base);
  const len = base.length;
  for (let k = 0; k < comps.length; k++) {
    const ck = coeffs[k] || 0;
    if (!ck) continue;
    const cd = comps[k];
    for (let i = 0; i < len; i++) out[i] += ck * cd[i];
  }
  return out;
}

/** The frame slot shown at (unbounded) phase φ. */
export function slotAt(phase: number, frames = MORPH_FRAMES): number {
  const frac = phase - Math.floor(phase);
  return Math.min(frames - 1, Math.floor(frac * frames));
}

/** Approximate bytes a cached movie holds (gray keeps one Float32 plane per
 *  frame, colour modes four). */
export function movieBytes(frames: number, w: number, h: number, mode: string): number {
  return frames * w * h * 4 * (mode === "gray" || mode === "gray-log" ? 1 : 4);
}

/** "disagreement movie · 3 members · 2 PCs ≈ 60% of variance". */
export function movieLabel(subset: string | null, comps: { varexp: number }[]): string {
  const nSub = subset ? subset.split(",").filter(Boolean).length : 0;
  const varTot = comps.reduce((a, c) => a + (c.varexp || 0), 0);
  const subLbl = subset ? ` · ${nSub} members` : "";
  const varLbl = varTot > 0 ? ` · ${comps.length} PCs ≈ ${(varTot * 100).toFixed(0)}% of variance` : "";
  return `disagreement movie${subLbl}${varLbl}`;
}

export const movieKey = (index: number, subset: string | null) => `${index}|${subset || ""}`;

/** LRU of built movies bounded in bytes; the entry on screen (`playing`) is
 *  never evicted. */
export class MovieStore<T extends { bytes: number }> {
  budget: number;
  playing: string | null = null;
  private map = new Map<string, T>();

  constructor(budget = MOVIE_BUDGET_BYTES) {
    this.budget = budget;
  }

  has(key: string): boolean { return this.map.has(key); }
  peek(key: string): T | undefined { return this.map.get(key); }

  get(key: string): T | undefined {
    const v = this.map.get(key);
    if (v) { this.map.delete(key); this.map.set(key, v); }
    return v;
  }

  set(key: string, value: T): void {
    this.map.delete(key);
    this.map.set(key, value);
  }

  delete(key: string): void { this.map.delete(key); }
  clear(): void { this.map.clear(); this.playing = null; }

  totalBytes(): number {
    let t = 0;
    for (const v of this.map.values()) t += v.bytes || 0;
    return t;
  }

  /** Drop least-recently-used entries until under budget. */
  evict(): void {
    let total = this.totalBytes();
    for (const k of [...this.map.keys()]) {
      if (total <= this.budget) break;
      if (k === this.playing) continue;
      total -= this.map.get(k)?.bytes || 0;
      this.map.delete(k);
    }
  }
}
