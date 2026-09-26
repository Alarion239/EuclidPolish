/* ops/jobs — phase-1 minimum: every local job of this server (running
   first) from the shared jobs feed, with cancel and the log in the
   inspector. The full job centre (filters, results, SLURM history) arrives
   with W-Ops in phase 3 (spec §8.7). */
import { useJobsFeed } from "../../../api/jobs";
import { JobList } from "../../../app/JobTray";
import { Badge, Button, Card, CardBody, CardHead, Page, PageHead } from "../../../ui";

export default function Jobs() {
  const feed = useJobsFeed();
  return (
    <Page>
      <PageHead eyebrow="ops · jobs" title="Local jobs"
        sub="Background jobs of this server since it started (newest 200 finished are kept). Click a job for its log."
        right={<Button size="sm" variant="ghost" onClick={feed.refresh}>Refresh</Button>} />
      <Card>
        <CardHead title="Jobs" sub={`${feed.jobs.length} known`}
          right={feed.running.length > 0 ? <Badge tone="info" dot>{feed.running.length} running</Badge> : undefined} />
        <CardBody>
          {feed.error && !feed.jobs.length
            ? <p className="muted">Could not list jobs: {feed.error.message}</p>
            : <JobList jobs={feed.jobs} empty="No local jobs since the server started." />}
        </CardBody>
      </Card>
    </Page>
  );
}
