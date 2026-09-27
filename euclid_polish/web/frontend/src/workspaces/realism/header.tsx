/* The ONE Realism header (the workspace tab strip's aside): the
   include-training toggle and the single training-catalogue sync action,
   shared by every tab that has a training variant (overview, galaxies, stars,
   pixels). The toggle lives in the URL (`?training=1`, shareable) and is
   sticky for the session, so switching tabs — whose links keep only
   `?inspect=` — does not drop it. */
import { useCallback, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { create } from "zustand";
import { matchPage } from "../../app/manifest";
import { useFasrcStatus } from "../../app/status";
import { useUrlState } from "../../hooks/useUrlState";
import { Badge, IconButton, Switch, Tooltip } from "../../ui";
import { useOverview } from "./api";
import { JOB, offlinePolicy, runAction, useRealismJob } from "./jobs";

const TRAINING_TABS = new Set(["overview", "galaxies", "stars", "pixels"]);

const useTrainingPref = create<{ on: boolean; set: (on: boolean) => void }>((set) => ({
  on: false,
  set: (on) => set({ on }),
}));

/** [include training, set] — the URL param, or the session's sticky choice. */
export function useIncludeTraining(): [boolean, (on: boolean) => void] {
  const [url, setUrl] = useUrlState("training", false);
  const sticky = useTrainingPref((s) => s.on);
  const set = useCallback((on: boolean) => {
    useTrainingPref.getState().set(on);
    setUrl(on);
  }, [setUrl]);
  return [url || sticky, set];
}

/** Reset the sticky choice (tests). */
export const resetTrainingPref = () => useTrainingPref.getState().set(false);

export function RealismHeader() {
  const tab = matchPage(useLocation().pathname)?.tab ?? null;
  const [url, setUrl] = useUrlState("training", false);
  const sticky = useTrainingPref((s) => s.on);
  // URL ↔ session: a shared link turns the sticky choice on; a tab switch
  // (whose link drops ?training) gets it back.
  useEffect(() => {
    if (url && !sticky) useTrainingPref.getState().set(true);
    else if (!url && sticky && tab && TRAINING_TABS.has(tab)) setUrl(true);
  }, [url, sticky, tab, setUrl]);
  const [training, setTraining] = useIncludeTraining();
  const overview = useOverview();
  const sync = useRealismJob(JOB.trainingSync);
  const fasrc = useFasrcStatus().data;
  if (!tab || !TRAINING_TABS.has(tab)) return null;
  const info = overview.data?.training;
  const available = !!info?.available;
  const offline = fasrc ? !fasrc.ssh_connected : false;
  const syncLabel = available ? "Re-sync training catalog" : "Sync training catalog";
  const policy = info ? offlinePolicy(info.sync, offline) : { disabled: true, hint: null };
  const state = !available ? { long: "no training", short: "no train" }
    : training ? { long: "train + test + val", short: "+ train" } : { long: "test + val", short: "test+val" };
  return (
    <div className="rl-head" role="group" aria-label="Training catalog">
      <Tooltip content={available
        ? "Add the training split (sources_train.csv) to the censuses and distributions"
        : "No training catalog cached: sync sources_train.csv first"}>
        <span className="rl-head__switch">
          <Switch size="sm" id="rl-training" checked={training && available} disabled={!available}
            aria-label="Include training catalog" onChange={setTraining} />
          <label className="rl-head__label" htmlFor="rl-training">training</label>
        </span>
      </Tooltip>
      <Badge size="sm" tone={training && available ? "warn" : undefined}
        title={available ? `${info?.population_fields_with_training ?? "?"} fields with training` : "no training catalog"}>
        <span className="rl-head__long">{state.long}</span>
        <span className="rl-head__short">{state.short}</span>
      </Badge>
      <IconButton size="sm" icon={available ? "reset" : "download"} label={syncLabel}
        tooltip={policy.hint ? `${syncLabel} — ${policy.hint}` : `${syncLabel} from FASRC (then rebuild the galaxy plots)`}
        loading={sync.busy} disabled={policy.disabled}
        onClick={() => { if (info) void runAction(info.sync, syncLabel); }} />
    </div>
  );
}
