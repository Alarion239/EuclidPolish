/* Notebook › Log (`/notebook/log`): the active campaign's lab notebook. The
 * campaign bar (active campaign and its commit, Back up…, Push, New
 * campaign…; Save snapshot in its menu), then the New entry card — prefilled
 * when a page's "Log to notebook" button sent an entry (`?entry=<markdown>`,
 * `?from=<page>`; both leave the URL once the entry is added) — then the
 * notebook itself, newest first, with its size and date span, the order and
 * a jump to a day. */
import { Link } from "react-router-dom";
import { useResource } from "../../../api/query";
import { pagePath } from "../../../app/nav";
import { useUrlState } from "../../../hooks/useUrlState";
import { Button, Callout, Card, CardBody, EmptyState, Page, Skeleton } from "../../../ui";
import { TRACKING_STATE_URL, type TrackingState } from "../api";
import { CampaignBar } from "../CampaignBar";
import { NotebookView } from "../NotebookView";
import "../notebook.css";

export default function Log() {
  const res = useResource<TrackingState>(TRACKING_STATE_URL, [], { ttl: 15_000, poll: 30_000 });
  const [entry, setEntry] = useUrlState("entry", "");
  const [from, setFrom] = useUrlState("from", "");
  const s = res.data;
  const active = s?.active ?? null;
  return (
    <Page className="nb-page">
      <CampaignBar state={s ?? null} />
      {res.loading && !s && <Skeleton lines={8} />}
      {res.error && !s && <Callout tone="bad" title="Could not read the notebook store">{res.error.message}</Callout>}
      {s && !active && (
        <Card><CardBody>
          <EmptyState icon="info" title="No active campaign"
            action={entry ? undefined : <Button asChild variant="ghost"><Link to={`${pagePath("notebook", { tab: "backups" })}?show=campaigns`}>Archived campaigns</Link></Button>}>
            {entry ? "Start a campaign (New campaign… above) to add this entry." : s.archived.length
              ? `${s.archived.length} saved campaigns are under Backups › Archived campaigns. Start one to collect backups, notes and FASRC jobs.`
              : "Start one to collect backups, notes and FASRC jobs."}
          </EmptyState>
        </CardBody></Card>
      )}
      {s && active && (
        <NotebookView key={entry ? `entry:${entry.length}` : "log"} text={s.log_md} editable title={`Notebook · ${active.title}`}
          initialDraft={entry} prefillFrom={from || undefined}
          onAppended={() => { setEntry(""); setFrom(""); }} />
      )}
    </Page>
  );
}
