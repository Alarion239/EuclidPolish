/* Models › Members (`/models/members`; absorbs the old Ensemble Members
   and Curves tabs): a "Pull from FASRC" banner only when finished members are
   waiting there (model.ts waitingOnFasrc: the job log vs the local
   registry), then the view switch — roster (default), curves (?view=curves)
   and archived (?view=archived; the counts are the tables' own row counts,
   not repeated on the options), and the
   "Pull from FASRC…" dialog one click away. Opening the page reads local
   files only; the FASRC check runs on request. */
import { useMemo, useState } from "react";
import { usePageActions } from "../../../app/palette";
import { useFasrcStatus } from "../../../app/status";
import { useUrlState } from "../../../hooks/useUrlState";
import { Button, Callout, Page, Segmented, Toolbar, ToolbarSpacer, Tooltip } from "../../../ui";
import { useMembers, useTrainingJobs } from "../api";
import { LoadState } from "../common";
import { memberNumber, waitingOnFasrc } from "../model";
import { Archived } from "../members/Archived";
import { Curves } from "../members/Curves";
import { PullDialog } from "../members/PullDialog";
import { Roster } from "../members/Roster";
import "../models.css";

type View = "roster" | "curves" | "archived";
const VIEWS: readonly View[] = ["roster", "curves", "archived"];

export default function Members() {
  const res = useMembers();
  const jobs = useTrainingJobs();
  const fasrc = useFasrcStatus().data;
  const [view, setView] = useUrlState<View>("view", "roster", { parse: (r) => (VIEWS.includes(r as View) ? r as View : undefined) });
  const [pullOpen, setPullOpen] = useState(false);
  const data = res.data;
  const waiting = useMemo(() => (data && jobs.data
    ? waitingOnFasrc(jobs.data.jobs, data.members, data.archived.map((t) => t.name))
    : { members: [], continued: [] }), [data, jobs.data]);
  const nWaiting = waiting.members.length + waiting.continued.length;
  usePageActions([
    { id: "mem-pull", label: "Pull members from FASRC…", group: "Members", keywords: ["download", "rsync", "checkpoints"], run: () => setPullOpen(true) },
    { id: "mem-roster", label: "Members: the roster", group: "Members", run: () => setView("roster") },
    { id: "mem-archived", label: "Members: archived members", group: "Members", run: () => setView("archived") },
    { id: "mem-refresh", label: "Refresh the members table", group: "Members", run: () => void res.reload() },
  ]);
  const offline = fasrc ? !fasrc.ssh_connected : false;
  const waitText = [
    waiting.members.length ? `${waiting.members.length} new member${waiting.members.length === 1 ? "" : "s"} finished on FASRC: ${waiting.members.map((n) => memberNumber(n)).join(", ")}` : null,
    waiting.continued.length ? `${waiting.continued.length} continued past the local checkpoint: ${waiting.continued.map((n) => memberNumber(n)).join(", ")}` : null,
  ].filter(Boolean).join("; ");

  return (
    <Page className="mdl-page">
      {nWaiting > 0 && (
        <Callout tone="info" dense action={<Button size="sm" variant="primary" icon="download" onClick={() => setPullOpen(true)}>Pull from FASRC…</Button>}>
          {waitText}.
        </Callout>
      )}
      <Toolbar label="Members view">
        <Segmented<View> size="sm" aria-label="Members view" value={view} onChange={setView} options={[
          { value: "roster", label: "Roster" },
          { value: "curves", label: "Curves" },
          { value: "archived", label: "Archived" },
        ]} />
        <ToolbarSpacer />
        {nWaiting === 0 && (
          <Tooltip content={offline ? `FASRC offline: ${fasrc?.last_error ?? "not connected"}` : "Check FASRC and pull changed members"}>
            <span><Button size="sm" icon="download" onClick={() => setPullOpen(true)}>Pull from FASRC…</Button></span>
          </Tooltip>
        )}
      </Toolbar>
      {view === "curves" ? <Curves /> : (
        <LoadState loading={res.loading} error={res.error} onRetry={res.reload}>
          {data && (view === "archived" ? <Archived rows={data.archived} /> : <Roster data={data} />)}
        </LoadState>
      )}
      {pullOpen && <PullDialog open={pullOpen} onOpenChange={setPullOpen} waiting={[...waiting.members, ...waiting.continued]} />}
    </Page>
  );
}
