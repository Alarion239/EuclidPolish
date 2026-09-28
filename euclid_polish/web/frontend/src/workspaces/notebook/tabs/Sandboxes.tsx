/* Notebook › Sandboxes (`/notebook/sandboxes`): the time-travel sandbox
 * servers (a second console on an old commit each), with open / start, stop
 * and remove, each confirmed where it stops or deletes. */
import { useResource } from "../../../api/query";
import { Callout, Caption, Page, Skeleton } from "../../../ui";
import { TRACKING_STATE_URL, type TrackingState } from "../api";
import { SandboxTable } from "../Sandboxes";
import "../notebook.css";

export default function Sandboxes() {
  const res = useResource<TrackingState>(TRACKING_STATE_URL, [], { ttl: 15_000, poll: 30_000 });
  const s = res.data;
  const running = s?.sandboxes.filter((b) => b.running).length ?? 0;
  return (
    <Page className="nb-page">
      {res.loading && !s && <Skeleton lines={4} />}
      {res.error && !s && <Callout tone="bad" title="Could not read the notebook store">{res.error.message}</Callout>}
      {s && (
        <>
          <SandboxTable sandboxes={s.sandboxes} />
          {s.sandboxes.length > 0 && (
            <Caption>{running} of {s.sandboxes.length} running. Start one from a backup's ⏱ in Backups.</Caption>
          )}
        </>
      )}
    </Page>
  );
}
