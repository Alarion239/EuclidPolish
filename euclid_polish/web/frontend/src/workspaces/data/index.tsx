/* Data workspace (spec §8.4): the synthetic training records, the star
   catalogue, real star cutouts, the empirical PSFs and the TNG atlas. The
   inspector kinds `star`, `truth`, `psf` and `tng` are registered by
   ./register.ts. */
import { Workspace, defineTabs } from "../../app/workspace";
import "./register";

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
