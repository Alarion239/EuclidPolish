/* Sky workspace (spec §7): atlas (the celestial sphere, W-SkyAtlas),
   results, experiments and catalog-eval (W-SkyResults). Loading the
   workspace registers the atlas's inspector kinds (`tile`, `source`) and the
   results' (`realtile`, `experiment`). */
import { Workspace, defineTabs } from "../../app/workspace";
import "./atlas/inspectors/register";
import "./results/register";

export const TABS = defineTabs("sky", {
  atlas: { load: () => import("./tabs/Atlas") },
  results: { load: () => import("./tabs/Results") },
  experiments: { load: () => import("./tabs/Experiments") },
  "catalog-eval": { load: () => import("./tabs/CatalogEval") },
});

export default function SkyWorkspace() {
  return <Workspace id="sky" tabs={TABS} />;
}
