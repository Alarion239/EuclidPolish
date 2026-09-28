/* Registers the System inspector kinds `prov`, `commit` and `root`. The
   components are one lazy chunk, so importing this module is cheap: the
   workspace imports it, and the shell does too (app/Shell.tsx)
   so any page can open `prov:<id>` / `commit:<hash>` / `root:<id>` before
   System was visited. */
import { lazy } from "react";
import { registerInspector } from "../../app/inspector";

const load = () => import("./inspectors");
const ProvInspector = lazy(() => load().then((m) => ({ default: m.ProvInspector })));
const CommitInspector = lazy(() => load().then((m) => ({ default: m.CommitInspector })));
const RootInspector = lazy(() => load().then((m) => ({ default: m.RootInspector })));

export const unregisterSystemInspectors = [
  registerInspector("prov", ProvInspector, { title: (id) => `Provenance ${id}` }),
  registerInspector("commit", CommitInspector, { title: (id) => `Commit ${id.slice(0, 10)}` }),
  registerInspector("root", RootInspector, { title: (id) => `Data root · ${id}` }),
];
