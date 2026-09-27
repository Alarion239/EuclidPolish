/* Home workspace (spec §8.1): no tabs — the dashboard (production numbers,
   health checks with the rail badge, running work, quick actions, sky). */
import { Workspace } from "../../app/workspace";
import Dashboard from "./Dashboard";

export default function HomeWorkspace() {
  return <Workspace id="home"><Dashboard /></Workspace>;
}
