/* Chart series colors. The canvas can't consume CSS variables directly, so we
   read them off :root at access time — that makes every figure theme-reactive:
   flip the light/dark toggle and the same `C.mean` returns the themed value on
   the next redraw. Fallbacks are the LIGHT (default) palette, used only
   before the stylesheet applies (tokens: src/theme/tokens.css). */

function cvar(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

const SERIES: Record<string, [string, string]> = {
  baseline: ["--series-baseline", "#e11d48"], // LR / floor-to-beat
  mean: ["--series-mean", "#2563eb"],         // ensemble mean
  comb: ["--series-comb", "#c2680a"],         // combiner
  muted: ["--series-muted", "#9aa6b6"],       // faint members
  cross: ["--series-cross", "#5b6879"],       // model–model r̃(k)
  guide: ["--series-guide", "#c2ccd9"],       // grid guide
  visfwhm: ["--series-visfwhm", "#2563eb"],   // VIS PSF marker
};

/* `C.mean` etc. — resolved live from the current theme's tokens. */
export const C = new Proxy({} as Record<keyof typeof SERIES, string>, {
  get: (_t, k: string) => (SERIES[k] ? cvar(SERIES[k][0], SERIES[k][1]) : ""),
});

const CAT_FALLBACK = [
  "#2563eb", "#0f9d58", "#d97706", "#7c3aed",
  "#db2777", "#0e8f96", "#dc2626", "#927608",
];

/* Categorical palette for coloring member lines by a facet (loss / depth). */
export function categorical(i: number): string {
  const j = ((i % 8) + 8) % 8;
  return cvar(`--cat-${j}`, CAT_FALLBACK[j]);
}

const LOSS_FALLBACK: Record<string, string> = {
  l1: "#2563eb", l2: "#0f9d58", l3: "#d97706", mse: "#0e8f96", berhu: "#7c3aed",
};

/* `LOSS_COLOR[loss]` — themed reconstruction-loss color. */
export const LOSS_COLOR = new Proxy({} as Record<string, string>, {
  get: (_t, k: string) => cvar(`--loss-${k}`, LOSS_FALLBACK[k] ?? "#9aa6b6"),
});

/* Euclid band colours, short → long wavelength (tokens --band-vis/-y/-j/-h). */
const BAND_TOKEN: Record<string, [string, string]> = {
  VIS: ["--band-vis", "#2563eb"],
  Y_E: ["--band-y", "#0f9d58"],
  J_E: ["--band-j", "#d97706"],
  H_E: ["--band-h", "#dc2626"],
};

/** Themed colour of a Euclid band ("VIS" | "Y_E" | "J_E" | "H_E"; also "Y",
 *  "J", "H"); unknown bands get the muted series colour. */
export function bandColor(band: string): string {
  const key = BAND_TOKEN[band] ? band : BAND_TOKEN[`${band}_E`] ? `${band}_E` : null;
  return key ? cvar(BAND_TOKEN[key][0], BAND_TOKEN[key][1]) : cvar("--series-muted", "#9aa6b6");
}

/* Perceptual value gradient (viridis) for coloring lines by a continuous
   quantity like test PSNR. Theme-independent — reads well on light and dark. */
const VIRIDIS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"];
function lerpHex(a: string, b: string, t: number): string {
  const p = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [ar, ag, ab] = p(a), [br, bg, bb] = p(b);
  const m = (x: number, y: number) => Math.round(x + (y - x) * t).toString(16).padStart(2, "0");
  return `#${m(ar, br)}${m(ag, bg)}${m(ab, bb)}`;
}
/** viridis(t): t in [0,1] → hex. */
export function viridis(t: number): string {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  return lerpHex(VIRIDIS[i], VIRIDIS[i + 1], x - i);
}
