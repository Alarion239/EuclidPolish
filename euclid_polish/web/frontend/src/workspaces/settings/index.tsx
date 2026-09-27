/* Settings workspace (spec §8.8): config (the JobConfig editor), connections
   (FASRC SSH + settings, the one Euclid archive session, FASRC-side Euclid
   credentials, the TNG token), appearance (theme, accent, density, image
   defaults) and about (server vs HEAD, bundle, runtime, disk). */
import { Workspace, defineTabs } from "../../app/workspace";

export const TABS = defineTabs("settings", {
  config: { load: () => import("./tabs/Config") },
  connections: { load: () => import("./tabs/Connections") },
  appearance: { load: () => import("./tabs/Appearance") },
  about: { load: () => import("./tabs/About") },
});

export default function SettingsWorkspace() {
  return <Workspace id="settings" tabs={TABS} />;
}
