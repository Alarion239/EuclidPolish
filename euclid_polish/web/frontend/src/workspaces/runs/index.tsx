/* Runs workspace, `/runs/<tab>` (console regrouping): what is running or
   queued, locally and on FASRC, how past runs went, and which step produces
   what. Tabs:
     live     one list of the running local jobs and the live SLURM jobs, the
              selected job's monitor, the fail-stop submission queue
     history  one ledger of the finished runs (SLURM + local), filtered by
              source / step / state / campaign; the selected run's log and,
              for training runs, the wall time per 1000 steps
     steps    the FASRC step catalogue by pipeline stage, one step's card
   The inspector kinds `prov`, `campaign` and `commit` are registered by the
   System and Notebook workspaces (../system/register.ts,
   ../notebook/register.ts), which the shell imports. */
import { Workspace, defineTabs } from "../../app/workspace";

export const TABS = defineTabs("runs", {
  live: { load: () => import("./tabs/Live") },
  history: { load: () => import("./tabs/History") },
  steps: { load: () => import("./tabs/Steps") },
});

export default function RunsWorkspace() {
  return <Workspace id="runs" tabs={TABS} />;
}
