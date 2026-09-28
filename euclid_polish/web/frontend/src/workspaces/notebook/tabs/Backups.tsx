/* Notebook › Backups (`/notebook/backups`): the active campaign's model,
 * FITS and image backups, and the archived campaigns, behind one set of
 * filter chips with their counts; every row has its ⏱ time travel (a
 * sandbox console on the backup's exact commit). URL: `show` (models | fits
 * | images | campaigns; the old tracking page's `bk` is read as a kind). */
import { useCallback, useState } from "react";
import { useResource } from "../../../api/query";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Button, Callout, EmptyState, Page, Segmented, Skeleton, Toolbar, ToolbarGroup, ToolbarSpacer,
} from "../../../ui";
import { TRACKING_STATE_URL, type TrackingState } from "../api";
import { ArchiveTable } from "../Archive";
import { BackupDialog, BackupTable } from "../Backups";
import { SHOWS, SHOW_LABEL, backupCounts, parseShow, type Show } from "../model";
import { TimeTravelDialog, type TimeTravelTarget } from "../TimeTravel";
import "../notebook.css";

export default function Backups() {
  const res = useResource<TrackingState>(TRACKING_STATE_URL, [], { ttl: 15_000, poll: 30_000 });
  const [rawShow, setShow] = useUrlState("show", "");
  const [bk, setBk] = useUrlState("bk", "");
  const show = parseShow(rawShow, bk);
  const [target, setTarget] = useState<TimeTravelTarget | null>(null);
  const [backupOpen, setBackupOpen] = useState(false);
  const onTimeTravel = useCallback((t: TimeTravelTarget) => setTarget(t), []);
  const s = res.data;
  const counts = backupCounts(s?.backups, s?.archived.length ?? 0);
  const pick = (v: Show) => { setBk(""); setShow(v === "models" ? "" : v); };
  return (
    <Page className="nb-page">
      <Toolbar label="Backups">
        <ToolbarGroup label="Show" hideLabel>
          <Segmented<Show> size="sm" value={show} onChange={pick} aria-label="Show"
            options={SHOWS.map((k) => ({ value: k, label: s ? `${SHOW_LABEL[k]} · ${counts[k]}` : SHOW_LABEL[k] }))} />
        </ToolbarGroup>
        <ToolbarSpacer />
        <Button size="sm" icon="plus" disabled={!s?.active} onClick={() => setBackupOpen(true)}>Back up…</Button>
      </Toolbar>
      {res.loading && !s && <Skeleton lines={8} />}
      {res.error && !s && <Callout tone="bad" title="Could not read the notebook store">{res.error.message}</Callout>}
      {s && show === "campaigns" && <ArchiveTable archived={s.archived} onTimeTravel={onTimeTravel} />}
      {s && show !== "campaigns" && (s.active ? (
        <BackupTable backups={s.backups} kind={show} campaign="current" trackingDir={s.tracking_dir}
          title={s.active.title} onTimeTravel={onTimeTravel} height={560} countText={null} />
      ) : (
        <EmptyState icon="database" title="No active campaign"
          action={<Button size="sm" onClick={() => pick("campaigns")}>Archived campaigns</Button>}>
          Backups belong to a campaign: the archived campaigns carry theirs.
        </EmptyState>
      ))}
      <TimeTravelDialog target={target} onClose={() => setTarget(null)} fasrcConnected={!!s?.ssh_connected} />
      <BackupDialog open={backupOpen} onOpenChange={setBackupOpen} />
    </Page>
  );
}
