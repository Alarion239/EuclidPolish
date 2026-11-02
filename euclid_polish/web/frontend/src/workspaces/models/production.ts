/* What production is, in words (pure; production.test.ts): ONE registered
   universal model the registry names production (`production_model`:
   "member_257", its catalogue label "Universal 257").
   - `universalLabel`: "Universal 257", the catalogue's label of a universal
     model (model_catalog.spec_label).
   No React, no DOM. */
import { memberNumber } from "./model";

/** "member_238" → "Universal 238" (the catalogue's label of its universal spec). */
export function universalLabel(name: string): string {
  return `Universal ${memberNumber(name) ?? name.replace(/^member_/, "")}`;
}
