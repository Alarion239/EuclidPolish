/* Chart series colors. The canvas can't consume CSS variables directly, so we
   read them off :root at access time — that makes every figure theme-reactive:
   flip the light/dark toggle and the same `C.mean` returns the themed value on
   the next redraw. Fallbacks are the dark palette (used before styles apply). */

function cvar(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

const SERIES: Record<string, [string, string]> = {
  baseline: ["--series-baseline", "#ff5c6c"], // LR / floor-to-beat
  mean: ["--series-mean", "#4c9ffe"],         // ensemble mean
  comb: ["--series-comb", "#ffb454"],         // combiner
  muted: ["--series-muted", "#55627a"],       // faint members
  cross: ["--series-cross", "#8794a8"],       // model–model r̃(k)
  guide: ["--series-guide", "#3a475c"],       // grid guide
  visfwhm: ["--series-visfwhm", "#4c9ffe"],   // VIS PSF marker
};

/* `C.mean` etc. — resolved live from the current theme's tokens. */
export const C = new Proxy({} as Record<keyof typeof SERIES, string>, {
  get: (_t, k: string) => (SERIES[k] ? cvar(SERIES[k][0], SERIES[k][1]) : ""),
});

const CAT_FALLBACK = [
  "#4c9ffe", "#56d68a", "#ffb454", "#b48ef2",
  "#ff77c8", "#33d6c4", "#ff6b6b", "#c7d05e",
];

/* Categorical palette for coloring member lines by a facet (loss / depth). */
export function categorical(i: number): string {
  const j = ((i % 8) + 8) % 8;
  return cvar(`--cat-${j}`, CAT_FALLBACK[j]);
}

const LOSS_FALLBACK: Record<string, string> = {
  l1: "#4c9ffe", l2: "#56d68a", l3: "#ffb454", berhu: "#b48ef2",
};

/* `LOSS_COLOR[loss]` — themed loss-norm color (l1 / l2 / l3 / berhu). */
export const LOSS_COLOR = new Proxy({} as Record<string, string>, {
  get: (_t, k: string) => cvar(`--loss-${k}`, LOSS_FALLBACK[k] ?? "#55627a"),
});
