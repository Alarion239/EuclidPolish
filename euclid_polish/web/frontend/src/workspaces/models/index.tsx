/* Models workspace, `/models/:mode/<tab>` (console regrouping, Team M): which
   model is best on synthetic truth and on real data, how the members trained,
   which combiner is production, and what SR looks like on synthetic fields.
   Tabs: leaderboard, members, train, combiner, diagnostics, images (./tabs).
   The ONE starfull/starless switch sits beside the tabs and keeps the current
   tab (and ?inspect=); a bare `/models/<mode>` returns to the last tab
   visited. Nothing else sits in the tab strip (the Images member picker is a
   side panel of its page), so the strip never reshapes. The inspector kinds
   `member` and `combiner` are registered by ./register.ts. */
import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { matchPage } from "../../app/manifest";
import { pagePath } from "../../app/nav";
import { Workspace, defineTabs } from "../../app/workspace";
import { Segmented } from "../../ui";
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

type Mode = "starfull" | "starless";

function RegimeSwitch() {
  const location = useLocation();
  const navigate = useNavigate();
  const m = matchPage(location.pathname);
  const mode: Mode = m?.params.mode === "starless" ? "starless" : "starfull";
  const inspect = new URLSearchParams(location.search).get("inspect");
  const go = (next: Mode) => {
    const path = pagePath("models", { tab: m?.tab, params: { mode: next } });
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

export default function ModelsWorkspace() {
  const lastTab = useLastTab();
  return <Workspace id="models" tabs={TABS} redirectTab={lastTab} aside={<RegimeSwitch />} />;
}
