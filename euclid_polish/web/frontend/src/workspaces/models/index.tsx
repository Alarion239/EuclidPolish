/* Models workspace, `/models/<tab>`: the universal models — how they score
   on synthetic truth, which knees they read, their power spectra, and
   training new ones. Tabs: universal, knees, spectra, train (./tabs).
   A bare `/models` returns to the last tab visited. Nothing sits beside the
   tab strip, so the strip never reshapes. The inspector kind `member` is
   registered by ./register.ts. */
import { useState } from "react";
import { useLocation } from "react-router-dom";
import { matchPage } from "../../app/manifest";
import { Workspace, defineTabs } from "../../app/workspace";
import "./register";
import "./models.css";

export const TABS = defineTabs("models", {
  universal: { load: () => import("./tabs/Universal") },
  knees: { load: () => import("./tabs/Knees") },
  spectra: { load: () => import("./tabs/Spectra") },
  train: { load: () => import("./tabs/Train") },
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
