/* Ops workspace (spec §8.7): local jobs, the FASRC console, experiment
   tracking, local git and the provenance browser. The inspector kinds
   `prov`, `campaign` and `commit` are registered by ./register.ts. */
import { Workspace, defineTabs } from "../../app/workspace";
import "./register";

export const TABS = defineTabs("ops", {
  jobs: { load: () => import("./tabs/Jobs") },
  fasrc: { load: () => import("./tabs/Fasrc") },
  tracking: { load: () => import("./tabs/Tracking") },
  git: { load: () => import("./tabs/Git") },
  provenance: { load: () => import("./tabs/Provenance") },
});

export default function OpsWorkspace() {
  return <Workspace id="ops" tabs={TABS} />;
}
