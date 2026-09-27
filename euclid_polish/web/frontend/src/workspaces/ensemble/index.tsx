/* Ensemble workspace (spec §8.2), `/ensemble/:mode/<tab>`: overview,
   members, curves, knee, diagnostics, combiners, disagreement, train. The
   ONE starfull/starless switch sits beside the tabs and keeps the current
   tab (and ?inspect=); left of it a tab may put one compact control
   (./aside.ts: Disagreement's member menu); a bare `/ensemble/<mode>` returns to the last tab
   visited. The inspector kinds `member` and `combiner` are registered by
   ./register.ts. */
import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { matchPage } from "../../app/manifest";
import { pagePath } from "../../app/nav";
import { Workspace, defineTabs } from "../../app/workspace";
import { Segmented } from "../../ui";
import { TabAsideSlot } from "./aside";
import "./register";
import "./ensemble.css";

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
  // A tab's one compact control (./aside.ts) portals in left of the switch.
  const [slot, setSlot] = useState<HTMLElement | null>(null);
  return (
    <TabAsideSlot.Provider value={slot}>
      <Workspace id="ensemble" tabs={TABS} redirectTab={lastTab}
        aside={<><span ref={setSlot} className="ens-aside__slot" /><RegimeSwitch /></>} />
    </TabAsideSlot.Provider>
  );
}
