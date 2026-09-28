/* Synthetic workspace, `/synthetic/<tab>` (console regrouping): what goes into
   a synthetic scene, whether each ingredient matches real Euclid Q1, whether
   the records are built from the current ingredients, and whether you can
   generate. Every ingredient tab has the same layout, top to bottom: the
   check against real data, the prior (fit / activate, `?prior=1`), then the
   "How this is produced" drawer (`?how=1`) with the real reference data and
   its FASRC steps (Fields calls its drawer "Real reference", `?ref=1`).

   Tabs: Status (can you generate), Records (what comes out), Galaxies (with
   the TNG templates), Stars, Noise, PSF (catalogue → cutouts → ePSF) and
   Fields (synthetic vs real LR). The one header beside the tabs is the
   include-training toggle (./header.tsx), shown only on the tabs whose
   numbers the training split changes. Inspector kinds: ./register.ts. */
import { Workspace, defineTabs } from "../../app/workspace";
import { SyntheticHeader } from "./header";
import "./register";
import "./realism.css";

export const TABS = defineTabs("synthetic", {
  status: { load: () => import("./tabs/Status") },
  records: { load: () => import("./tabs/Records") },
  galaxies: { load: () => import("./tabs/Galaxies") },
  stars: { load: () => import("./tabs/Stars") },
  noise: { load: () => import("./tabs/Noise") },
  psf: { load: () => import("./tabs/Psf") },
  fields: { load: () => import("./tabs/Fields") },
});

export default function SyntheticWorkspace() {
  return <Workspace id="synthetic" tabs={TABS} aside={<SyntheticHeader />} />;
}
