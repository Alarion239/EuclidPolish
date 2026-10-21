/* What production is, in words (pure; production.test.ts). Production is
   either the production gate (the `spatial_gate_combiner` dir, a combiner
   over the ensemble members) or ONE registered universal model the registry
   names production (`production_model`: "member_238", its catalogue label
   "Universal 238"). The Models tabs name it from here:
   - `universalLabel`: "Universal 238", the catalogue's label of a universal
     model (model_catalog.spec_label);
   - `gateName`: the gate dir in words — "production gate" while it is
     production, "former production gate" while a universal model is, and
     "archived gate" for its curves once the gates are archived;
   - `productionComparison`: the Leaderboard's rows with the universal
     production first (its own score) and the gate's row renamed, with no
     real facts (the `production` spec runs the universal model now).
   No React, no DOM. */
import type { Overview } from "./api";
import { memberNumber, type Comparison, type ComparisonRow } from "./model";

/** "member_238" → "Universal 238" (the catalogue's label of its universal spec). */
export function universalLabel(name: string): string {
  return `Universal ${memberNumber(name) ?? name.replace(/^member_/, "")}`;
}

export type GateName = "production gate" | "former production gate" | "archived gate";

/** The gate dir in words: "production gate" while production is the gate;
 *  while a universal model is production, "former production gate" (still
 *  on disk, `gateAvailable`) or "archived gate" (only its curves remain). */
export function gateName(productionModel: string | null | undefined, gateAvailable = true): GateName {
  if (!productionModel) return "production gate";
  return gateAvailable ? "former production gate" : "archived gate";
}

/** "Former production gate": a row label. */
export const capitalized = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const N_BANDS = 4;
const finite = (v: number | null | undefined): number | null => (v != null && Number.isFinite(v) ? v : null);

/** The Leaderboard's comparison while a universal model is production: its
 *  row first ("Production · Universal 238", its universal score, the
 *  `production` spec's real facts), then the ensemble's rows with the gate
 *  renamed (former production / archived gate) and no real facts — a run's
 *  `production` score is of the universal model now. Unchanged while the
 *  gate is production. */
export function productionComparison(c: Comparison, o: Pick<Overview, "production_model" | "production_gate"> | null | undefined): Comparison {
  const pm = o?.production_model;
  if (!pm) return c;
  const bands = Array.from({ length: N_BANDS }, (_, i) => finite(pm.integrated?.[i]));
  const known = bands.filter((v): v is number => v != null);
  const mean = finite(pm.mean) ?? (known.length === N_BANDS ? known.reduce((a, b) => a + b, 0) / N_BANDS : null);
  const row: ComparisonRow = { id: "production", label: `Production · ${pm.label}`, integrated: mean, bands, spec: "production" };
  const gate = capitalized(gateName(pm.name, o?.production_gate?.available ?? false));
  return { ...c, rows: [row, ...c.rows.map((r) => (r.id === "gate" ? { ...r, label: gate, spec: null } : r))] };
}
