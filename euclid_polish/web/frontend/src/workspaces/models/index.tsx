/* Models workspace, `/models/<tab>` (console regrouping, Team M): which
   model is best on synthetic truth and on real data, how the members trained,
   which combiner is production, and what SR looks like on synthetic fields.
   Tabs: leaderboard, members, train, combiner, diagnostics, images (./tabs).
   A bare `/models` returns to the last tab visited. Nothing sits beside the
   tab strip (the Images member picker is a side panel of its page), so the
   strip never reshapes. The inspector kinds `member` and `combiner` are
   registered by ./register.ts. */
import { useState } from "react";
import { useLocation } from "react-router-dom";
import { matchPage } from "../../app/manifest";
import { Workspace, defineTabs } from "../../app/workspace";
import "./register";
import "./models.css";

export const TABS = defineTabs("models", {
  leaderboard: { load: () => import("./tabs/Leaderboard") },
  members: { load: () => import("./tabs/Members") },
  train: { load: () => import("./tabs/Train") },
  combiner: { load: () => import("./tabs/Combiner") },
  diagnostics: { load: () => import("./tabs/Diagnostics") },
  images: { load: () => import("./tabs/Images") },
});

/** The last tab this workspace showed (kept while it stays mounted). */
function useLastTab(): string | null {
  const tab = matchPage(useLocation().pathname)?.tab ?? null;
  const [last, setLast] = useState<string | null>(tab);
  if (tab && tab !== last) setLast(tab); // state derived from the previous render
  return tab ?? last;
}

export default function ModelsWorkspace() {
  const lastTab = useLastTab();
  return <Workspace id="models" tabs={TABS} redirectTab={lastTab} />;
}
