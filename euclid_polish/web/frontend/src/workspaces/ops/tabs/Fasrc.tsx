/* ops/fasrc (spec §8.7): the SLURM cluster console.
 *   live     every PENDING/RUNNING job + the selected one's monitor
 *   queue    the local fail-stop submission queue (remove / clear / resume)
 *   steps    run any pipeline step (schema-driven form, clone a past run)
 *   history  every run of every step, reconcile unresolved states
 *   logs     run logs: paged, follow, search the whole file
 *   storage  remote sizes + file browser, re-link data, pull checkpoints
 *   git      FASRC checkout vs local HEAD, git pull, conda env update
 * `?view=` selects; each view owns its own URL keys. */
import { useJobsFeed } from "../../../api/jobs";
import { useFasrcStatus } from "../../../app/status";
import { usePageActions } from "../../../app/palette";
import { useUrlState } from "../../../hooks/useUrlState";
import { Page, Segmented } from "../../../ui";
import { ConnectionBar } from "../fasrc/Connection";
import { HistoryPanel } from "../fasrc/History";
import { LiveJobs } from "../fasrc/Live";
import { LogsPanel } from "../fasrc/Logs";
import { QueuePanel } from "../fasrc/Queue";
import { RemoteGitPanel } from "../fasrc/RemoteGit";
import { StepsPanel } from "../fasrc/Steps";
import { StoragePanel } from "../fasrc/Storage";
import "../ops.css";

const VIEWS = ["live", "queue", "steps", "history", "logs", "storage", "git"] as const;
type View = typeof VIEWS[number];
const LABEL: Record<View, string> = {
  live: "Live", queue: "Queue", steps: "Steps", history: "History", logs: "Logs", storage: "Storage", git: "Git",
};
const parseView = (raw: string): View | undefined => (VIEWS as readonly string[]).includes(raw) ? raw as View : undefined;

export default function Fasrc() {
  const [view, setView] = useUrlState<View>("view", "live", { parse: parseView, replace: false });
  const status = useFasrcStatus();
  const connected = !!status.data?.ssh_connected;
  const feed = useJobsFeed();
  const liveCount = feed.slurm.length;
  const queued = feed.slurmQueue?.count ?? 0;

  usePageActions(VIEWS.map((v) => ({
    id: `fasrc-view-${v}`, label: `FASRC: ${LABEL[v]}`, group: "FASRC", keywords: ["slurm", v], run: () => setView(v),
  })));

  return (
    <Page className="ops-page">
      <div className="ops-bar" role="toolbar" aria-label="FASRC views">
        <Segmented<View> value={view} onChange={setView} aria-label="View" className="ops-bar__views"
          options={VIEWS.map((v) => ({
            value: v,
            label: v === "live" && liveCount ? `${LABEL[v]} · ${liveCount}` : v === "queue" && queued ? `${LABEL[v]} · ${queued}` : LABEL[v],
          }))} />
        <span className="ops-spacer" />
        <ConnectionBar />
      </div>
      {view === "live" && <LiveJobs onView={(v) => setView(parseView(v) ?? "live")} />}
      {view === "queue" && <QueuePanel />}
      {view === "steps" && <StepsPanel />}
      {view === "history" && <HistoryPanel fasrcConnected={connected} />}
      {view === "logs" && <LogsPanel fasrcConnected={connected} />}
      {view === "storage" && <StoragePanel fasrcConnected={connected} />}
      {view === "git" && <RemoteGitPanel fasrcConnected={connected} />}
    </Page>
  );
}
