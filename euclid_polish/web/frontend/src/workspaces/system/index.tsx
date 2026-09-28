/* System workspace, `/system/<tab>` (console regrouping): is the console
   connected, configured, on the right code and within its disk budget, and
   where did a product come from? Tabs:
     connections  FASRC, the Euclid archive session, FASRC-side credentials
     config       the one job_config.json editor
     lineage      search a product and see its lineage (was Ops › Provenance)
     code         laptop / server / FASRC commits in one sentence; the local
                  and FASRC checkouts (?side=fasrc opens on FASRC)
     storage      this laptop's disk and data roots, FASRC storage, the
                  evaluation maintenance (?side=fasrc opens on FASRC)
     appearance   theme, layout, how images are shown
   The inspector kinds `prov`, `commit` and `root` are registered by
   ./register.ts (imported here, and by the shell). */
import { Workspace, defineTabs } from "../../app/workspace";
import "./register";

export const TABS = defineTabs("system", {
  connections: { load: () => import("./tabs/Connections") },
  config: { load: () => import("./tabs/Config") },
  lineage: { load: () => import("./tabs/Lineage") },
  code: { load: () => import("./tabs/Code") },
  storage: { load: () => import("./tabs/Storage") },
  appearance: { load: () => import("./tabs/Appearance") },
});

export default function SystemWorkspace() {
  return <Workspace id="system" tabs={TABS} />;
}
