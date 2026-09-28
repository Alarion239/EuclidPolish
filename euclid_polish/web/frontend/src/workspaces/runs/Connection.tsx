/* FASRC connection chip + connect / disconnect (C4). The real `last_error`
 * sits in the chip's tooltip; the full editor is System › Connections. */
import { useState } from "react";
import { Link } from "react-router-dom";
import { apiPost } from "../../api/client";
import { invalidate } from "../../api/query";
import { refreshJobsFeed } from "../../api/jobs";
import { FASRC_STATUS_URL, useFasrcStatus } from "../../app/status";
import { Badge, Button, Tooltip, confirm, toast } from "../../ui";
import "./runs.css";

export function ConnectionBar() {
  const status = useFasrcStatus();
  const [busy, setBusy] = useState(false);
  const s = status.data;
  const connected = !!s?.ssh_connected;
  async function toggle() {
    if (connected && !(await confirm({ title: "Disconnect from FASRC?",
      message: "Closes the shared SSH session; running SLURM jobs keep running.", confirmLabel: "Disconnect" }))) return;
    setBusy(true);
    try {
      await apiPost(connected ? "/api/fasrc/disconnect" : "/api/fasrc/connect");
      toast.success(connected ? "Disconnected from FASRC" : "Connected to FASRC");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      void invalidate(FASRC_STATUS_URL);
      void invalidate("/api/fasrc/");
      void refreshJobsFeed();
    }
  }
  // Until the first status answer nothing is claimed: no "offline" flash.
  if (!s) {
    return (
      <span className="runs-conn" aria-busy="true">
        <Badge tone="neutral">FASRC …</Badge>
      </span>
    );
  }
  const tip = connected ? `Connected${s.socket ? ` · ${s.socket}` : ""}` : (s.last_error || "Not connected");
  return (
    <span className="runs-conn">
      <Tooltip content={tip}>
        <Link to="/system/connections" className="runs-conn__chip" aria-label={`FASRC ${connected ? "connected" : "offline"}`}>
          <Badge tone={connected ? "good" : "warn"} dot>{connected ? "FASRC" : "FASRC offline"}</Badge>
        </Link>
      </Tooltip>
      <Button size="sm" variant={connected ? "ghost" : "primary"} loading={busy} onClick={toggle}>
        {connected ? "Disconnect" : "Connect"}
      </Button>
    </span>
  );
}
