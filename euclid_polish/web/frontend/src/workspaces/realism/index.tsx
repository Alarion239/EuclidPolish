/* Realism workspace (spec §8.3). Phase 1: legacy adapters; W-Realism replaces
   the tabs in phase 3. */
import { Workspace, defineTabs } from "../../app/workspace";

export const TABS = defineTabs("realism", {
  overview: { load: () => import("./tabs/Overview") },
  noise: { load: () => import("./tabs/Noise") },
  galaxies: { load: () => import("./tabs/Galaxies") },
  stars: { load: () => import("./tabs/Stars") },
  pixels: { load: () => import("./tabs/Pixels") },
  visual: { load: () => import("./tabs/Visual") },
});

export default function RealismWorkspace() {
  return <Workspace id="realism" tabs={TABS} />;
}
