/* Ops workspace (spec §8.7). Phase 1: legacy adapters (fasrc, tracking, git)
   and a minimal local-jobs list; W-Ops replaces the tabs in phase 3. */
import { Workspace, defineTabs } from "../../app/workspace";

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
