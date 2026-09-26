/* Ensemble workspace (spec §8.2), `/ensemble/:mode/<tab>`. Phase 1: every tab
   but `train` renders the old Ensemble page (all its cards), `train` renders
   Train members; W-Ensemble splits the page into the real tabs in phase 3.
   The regime switch beside the tabs keeps the current tab. So does the legacy
   page's own starfull/starless switch (until W-Ensemble removes it): it
   navigates to the bare `/ensemble/<mode>`, which returns to the last tab
   visited instead of the default one. */
import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { matchPage } from "../../app/manifest";
import { pagePath } from "../../app/nav";
import { Workspace, defineTabs } from "../../app/workspace";
import { Segmented } from "../../ui";

export const TABS = defineTabs("ensemble", {
  overview: { load: () => import("./tabs/Overview") },
  members: { load: () => import("./tabs/Members") },
  curves: { load: () => import("./tabs/Curves") },
  knee: { load: () => import("./tabs/Knee") },
  diagnostics: { load: () => import("./tabs/Diagnostics") },
  combiners: { load: () => import("./tabs/Combiners") },
  disagreement: { load: () => import("./tabs/Disagreement") },
  train: { load: () => import("./tabs/Train") },
});

type Mode = "starfull" | "starless";

function RegimeSwitch() {
  const location = useLocation();
  const navigate = useNavigate();
  const m = matchPage(location.pathname);
  const mode: Mode = m?.params.mode === "starless" ? "starless" : "starfull";
  const inspect = new URLSearchParams(location.search).get("inspect");
  const go = (next: Mode) => {
    const path = pagePath("ensemble", { tab: m?.tab, params: { mode: next } });
    navigate(inspect ? `${path}?${new URLSearchParams({ inspect }).toString()}` : path);
  };
  return (
    <Segmented<Mode> size="sm" aria-label="Star regime" value={mode} onChange={go}
      options={[
        { value: "starfull", label: "starfull", title: "Reconstruct stars (default regime)" },
        { value: "starless", label: "starless", title: "Erase stars (opt-in regime)" },
      ]} />
  );
}

/** The last tab this workspace showed (kept while it stays mounted). */
function useLastTab(): string | null {
  const tab = matchPage(useLocation().pathname)?.tab ?? null;
  const [last, setLast] = useState<string | null>(tab);
  if (tab && tab !== last) setLast(tab); // state derived from the previous render
  return tab ?? last;
}

export default function EnsembleWorkspace() {
  const lastTab = useLastTab();
  return <Workspace id="ensemble" tabs={TABS} aside={<RegimeSwitch />} redirectTab={lastTab} />;
}
