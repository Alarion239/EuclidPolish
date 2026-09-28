/* Figures workspace, `/figures/<tab>` (console regrouping): which figures you
   have for the paper and the poster, whether they are made from the current
   model, and how to export them. Plates (the calibration plates, NEXUS
   comparison plates and the synthetic poster scene) and Sheet (the
   publication contact sheet built from the crops saved in any viewer; it
   absorbs the old Grid and Results pages) and Studies (frozen whole-ensemble
   comparisons: charts, exports, attached fields). Loading the workspace registers
   the `figure:<result id>` inspector kind. */
import { Workspace, defineTabs } from "../../app/workspace";
import "./register";

export const TABS = defineTabs("figures", {
  plates: { load: () => import("./tabs/Plates") },
  sheet: { load: () => import("./tabs/Sheet") },
  studies: { load: () => import("./tabs/Studies") },
});

export default function FiguresWorkspace() {
  return <Workspace id="figures" tabs={TABS} />;
}
