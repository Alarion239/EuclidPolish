/* Figures workspace (spec §8.5): grid (the publication contact sheet of saved
   crops), plates (presentation plates, NEXUS comparison plates, the poster
   cutout) and results (every crop saved from a viewer). Loading the
   workspace registers the `figure:<result id>` inspector kind. */
import { Workspace, defineTabs } from "../../app/workspace";
import "./register";

export const TABS = defineTabs("figures", {
  grid: { load: () => import("./tabs/Grid") },
  plates: { load: () => import("./tabs/Plates") },
  results: { load: () => import("./tabs/Results") },
});

export default function FiguresWorkspace() {
  return <Workspace id="figures" tabs={TABS} />;
}
