/* settings/connections — phase-1 minimum: the FASRC connection with the
   real error (C4 `last_error`) and connect / retry / disconnect. The SSH
   settings editor (/api/fasrc/config), the single Euclid session, the
   FASRC-side credentials and the TNG token arrive with W-Settings+Home in
   phase 3 (spec §8.8). */
import { useState } from "react";
import { Link } from "react-router-dom";
import { apiPost, ApiError } from "../../../api/client";
import { invalidate } from "../../../api/query";
import { FASRC_STATUS_URL, useFasrcStatus } from "../../../app/status";
import { formatDateTime } from "../../../format";
import { Badge, Button, Callout, Card, CardBody, CardHead, DefList, Page, PageHead, toast } from "../../../ui";

export default function Connections() {
  const status = useFasrcStatus();
  const [busy, setBusy] = useState<string | null>(null);
  const s = status.data;
  const connected = !!s?.ssh_connected;

  async function act(url: string, label: string) {
    setBusy(url);
    try {
      const r = await apiPost<{ ok?: boolean; error?: string }>(url, {});
      if (r.ok === false) toast.error(`${label} failed`, { description: r.error });
      else toast.success(`${label}: done`);
    } catch (e) {
      toast.error(`${label} failed`, { description: e instanceof ApiError ? e.message : String(e) });
    } finally {
      setBusy(null);
      void invalidate(FASRC_STATUS_URL);
    }
  }

  return (
    <Page>
      <PageHead eyebrow="settings · connections" title="Connections"
        sub="The FASRC SSH session every cluster action uses. Local pages keep working while it is down." />
      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <Card>
          <CardHead title="FASRC" sub="SSH ControlMaster to the login node"
            right={status.loading ? <Badge>…</Badge>
              : <Badge tone={connected ? "good" : "bad"} dot>{connected ? "connected" : "offline"}</Badge>} />
          <CardBody>
            {s?.last_error && !connected && (
              <Callout tone="bad" title="Last connection error">
                <pre className="mono" style={{ margin: 0, whiteSpace: "pre-wrap" }}>{s.last_error}</pre>
              </Callout>
            )}
            <DefList items={[
              ["state", connected ? "connected" : "not connected"],
              s?.connected_at ? ["since", formatDateTime(s.connected_at)] : null,
              s?.socket ? ["socket", <code className="mono">{s.socket}</code>] : null,
            ]} />
            <div className="row" style={{ marginTop: "var(--s3)", gap: "var(--s2)" }}>
              {connected ? (
                <Button onClick={() => act("/api/fasrc/disconnect", "Disconnect")} loading={busy === "/api/fasrc/disconnect"}>Disconnect</Button>
              ) : (
                <>
                  <Button variant="primary" onClick={() => act("/api/fasrc/connect", "Connect")} loading={busy === "/api/fasrc/connect"}>Connect</Button>
                  <Button onClick={() => act("/api/connection/retry", "Retry")} loading={busy === "/api/connection/retry"}>Retry the startup connect</Button>
                </>
              )}
            </div>
          </CardBody>
        </Card>
        <Callout tone="info" title="More connection settings arrive in phase 3">
          The SSH settings editor, the one Euclid archive session and the TNG token move here. Until
          then, log in to the Euclid archive from the <Link to="/data/catalog">Catalog</Link> page and
          set the TNG token on the <Link to="/data/tng">TNG</Link> page.
        </Callout>
      </div>
    </Page>
  );
}
