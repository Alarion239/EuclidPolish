/* settings/about (spec §8.8): the server's boot commit vs the checkout HEAD
   (C3 /api/version), the committed dist build, and this bundle's runtime.
   Python / Node versions and disk usage per data root need a backend
   endpoint (phase 3, W-Settings+Home). */
import { version as reactVersion } from "react";
import { useVersion } from "../../../app/status";
import { formatDateTime, formatRelative } from "../../../format";
import { Badge, Button, Callout, Card, CardBody, CardHead, CopyButton, DefList, Page, PageHead, Skeleton } from "../../../ui";

export default function About() {
  const version = useVersion();
  const v = version.data;
  return (
    <Page>
      <PageHead eyebrow="settings · about" title="About"
        sub="Which code the server runs, which bundle this page loaded, and whether a restart is due."
        right={<Button size="sm" variant="ghost" onClick={version.reload}>Refresh</Button>} />
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: "var(--s4)" }}>
        <Card>
          <CardHead title="Server" right={v ? (v.behind ? <Badge tone="warn" dot>behind HEAD</Badge> : <Badge tone="good" dot>at HEAD</Badge>) : undefined} />
          <CardBody>
            {version.loading && <Skeleton lines={5} />}
            {version.error && <Callout tone="bad" title="Could not read /api/version">{version.error.message}</Callout>}
            {v && (
              <>
                {v.behind && (
                  <Callout tone="warn" title="Restart the server">
                    It was started at <code className="mono">{v.boot_short}</code>; the checkout is now at
                    {" "}<code className="mono">{v.head_short}</code>. New backend code (and a rebuilt
                    console) load after a restart.
                  </Callout>
                )}
                <DefList items={[
                  ["boot commit", <span className="row" style={{ gap: 6 }}><code className="mono">{v.boot_commit ?? "—"}</code>{v.boot_commit && <CopyButton value={v.boot_commit} label="Copy boot commit" />}</span>],
                  ["HEAD", <code className="mono">{v.head_commit ?? "—"}</code>],
                  ["working tree", v.dirty ? <Badge tone="warn">uncommitted changes</Badge> : <Badge tone="good">clean</Badge>],
                  ["started", v.started_at ? `${formatDateTime(v.started_at)} (${formatRelative(v.started_at)})` : "—"],
                  ["pid", v.pid != null ? <code className="mono">{v.pid}</code> : "—"],
                ]} />
              </>
            )}
          </CardBody>
        </Card>
        <Card>
          <CardHead title="Console bundle" sub="static/dist (committed build)" />
          <CardBody>
            <DefList items={[
              ["dist built", v?.dist?.built_at ? `${formatDateTime(v.dist.built_at)} (${formatRelative(v.dist.built_at)})` : "—"],
              ["index hash", v?.dist?.index_hash ? <code className="mono">{v.dist.index_hash}</code> : "—"],
              ["mode", import.meta.env.MODE],
              ["React", reactVersion],
            ]} />
          </CardBody>
        </Card>
        <Callout tone="info" title="Coming in phase 3">
          Python and Node versions and the disk usage of each local data root need a backend endpoint.
        </Callout>
      </div>
    </Page>
  );
}
