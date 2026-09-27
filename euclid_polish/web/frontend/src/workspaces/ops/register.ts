/* Registers the Ops inspector kinds `prov`, `campaign` and `commit`. The
   components are one lazy chunk, so importing this module is cheap: the
   workspace imports it, and the shell may too so any page can open
   `prov:<id>` / `campaign:<dir>` / `commit:<hash>` before Ops was visited. */
import { lazy } from "react";
import { registerInspector } from "../../app/inspector";

const load = () => import("./inspectors");
const ProvInspector = lazy(() => load().then((m) => ({ default: m.ProvInspector })));
const CampaignInspector = lazy(() => load().then((m) => ({ default: m.CampaignInspector })));
const CommitInspector = lazy(() => load().then((m) => ({ default: m.CommitInspector })));

export const unregisterOpsInspectors = [
  registerInspector("prov", ProvInspector, { title: (id) => `Provenance ${id}` }),
  registerInspector("campaign", CampaignInspector, { title: (id) => `Campaign ${id}` }),
  registerInspector("commit", CommitInspector, { title: (id) => `Commit ${id.slice(0, 10)}` }),
];
