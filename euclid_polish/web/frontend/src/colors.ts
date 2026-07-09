/* Chart series colors — canvas can't read CSS vars, so these mirror the
   --series-* / accent tokens in theme.css. Single source for data-viz color. */
export const C = {
  baseline: "#ff5c6c",   // LR / floor-to-beat
  mean: "#4c9ffe",       // ensemble mean
  comb: "#ffb454",       // combiner
  muted: "#55627a",      // faint members
  cross: "#8794a8",      // model–model r̃(k)
  guide: "#3a475c",      // grid guide
  visfwhm: "#4c9ffe",    // VIS PSF marker
};

/* Categorical palette for coloring member lines by a facet (loss / depth). */
export const CATEGORICAL = [
  "#4c9ffe", "#56d68a", "#ffb454", "#b48ef2",
  "#ff77c8", "#33d6c4", "#ff6b6b", "#c7d05e",
];

export const LOSS_COLOR: Record<string, string> = {
  l1: "#4c9ffe", l2: "#56d68a", l3: "#ffb454", berhu: "#b48ef2",
};
