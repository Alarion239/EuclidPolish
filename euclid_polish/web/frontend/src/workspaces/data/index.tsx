/* Data workspace (spec §8.4). Phase 1: legacy adapters (records → the old
   Sky/TFRecords page, catalog, cutouts, psfs, tng); W-Data replaces the tabs
   in phase 3. */
import { Workspace, defineTabs } from "../../app/workspace";

export const TABS = defineTabs("data", {
  records: { load: () => import("./tabs/Records") },
  catalog: { load: () => import("./tabs/Catalog") },
  cutouts: { load: () => import("./tabs/Cutouts") },
  psfs: { load: () => import("./tabs/Psfs") },
  tng: { load: () => import("./tabs/Tng") },
});

export default function DataWorkspace() {
  return <Workspace id="data" tabs={TABS} />;
}
