/* Realism workspace (spec §8.3), `/realism/<tab>`: overview (readiness of
   every prior + the synthetic_generate gate), noise, galaxies, stars, pixels
   (field statistics) and visual (synthetic–real). ONE shared header beside
   the tabs: the include-training toggle and the training-catalogue sync.
   Inspector kinds `readiness`, `noisepos` and `archivefield` are registered by
   ./register.ts. */
import { Workspace, defineTabs } from "../../app/workspace";
import { RealismHeader } from "./header";
import "./register";
import "./realism.css";

export const TABS = defineTabs("realism", {
  overview: { load: () => import("./tabs/Overview") },
  noise: { load: () => import("./tabs/Noise") },
  galaxies: { load: () => import("./tabs/Galaxies") },
  stars: { load: () => import("./tabs/Stars") },
  pixels: { load: () => import("./tabs/Pixels") },
  visual: { load: () => import("./tabs/Visual") },
});

export default function RealismWorkspace() {
  return <Workspace id="realism" tabs={TABS} aside={<RealismHeader />} />;
}
