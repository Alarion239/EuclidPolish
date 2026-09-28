/* Sky workspace, `/sky/<tab>` (console regrouping): what SR does on real
   Euclid sky. Atlas (the celestial sphere: where is it?), Targets (production
   SR on each science target, and is it current?) and Compare (which model is
   safest on real bright objects?). Loading the workspace registers the
   atlas's inspector kinds (`tile`, `source`) and the results'
   (`realtile`, `experiment`).

   Targets reads its v2 keys itself (`?set=`, `?g=`, `?state=`) and, once, the
   interim store ids (`?src=`) of links copied before the merge; Compare reads
   `?exp=`, `?scope=`, `?metric=`, `?tiles=`, `?new=1` and `?defs=1`. */
import { Workspace, defineTabs } from "../../app/workspace";
import "./atlas/inspectors/register";
import "./results/register";

export const TABS = defineTabs("sky", {
  atlas: { load: () => import("./tabs/Atlas") },
  targets: { load: () => import("./tabs/Targets") },
  compare: { load: () => import("./tabs/Compare") },
});

export default function SkyWorkspace() {
  return <Workspace id="sky" tabs={TABS} />;
}
