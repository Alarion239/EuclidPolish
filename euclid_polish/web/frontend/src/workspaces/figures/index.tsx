/* Figures workspace (spec §8.5). Phase 1: grid → the figure-grid builder,
   plates → the old Visualization page; W-Figures replaces the tabs in phase 3. */
import { Workspace, defineTabs } from "../../app/workspace";

export const TABS = defineTabs("figures", {
  grid: { load: () => import("./tabs/Grid") },
  plates: { load: () => import("./tabs/Plates") },
  results: { load: () => import("./tabs/Results") },
});

export default function FiguresWorkspace() {
  return <Workspace id="figures" tabs={TABS} />;
}
