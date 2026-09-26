/* Home workspace (spec §8.1): no tabs. A minimal dashboard in phase 1; the
   W-Settings+Home owner adds staleness alerts and the knee-integrated
   headline in phase 3. */
import { Workspace } from "../../app/workspace";
import Dashboard from "./Dashboard";

export default function HomeWorkspace() {
  return <Workspace id="home"><Dashboard /></Workspace>;
}
