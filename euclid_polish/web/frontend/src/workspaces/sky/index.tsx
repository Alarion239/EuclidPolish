/* Sky workspace (spec §7). Phase 1: legacy adapters (atlas → JWST × Euclid,
   results → Inference, catalog-eval → Evaluation); W-SkyAtlas / W-SkyResults
   replace the tabs in phase 3. */
import { Workspace, defineTabs } from "../../app/workspace";

export const TABS = defineTabs("sky", {
  atlas: { load: () => import("./tabs/Atlas") },
  results: { load: () => import("./tabs/Results") },
  experiments: { load: () => import("./tabs/Experiments") },
  "catalog-eval": { load: () => import("./tabs/CatalogEval") },
});

export default function SkyWorkspace() {
  return <Workspace id="sky" tabs={TABS} />;
}
