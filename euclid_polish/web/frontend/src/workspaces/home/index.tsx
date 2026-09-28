/* Home workspace (`/`, console regrouping): no tabs — the production
   verdict, the Loop strip, what runs now and the latest figures. */
import { Workspace } from "../../app/workspace";
import Dashboard from "./Dashboard";

export default function HomeWorkspace() {
  return <Workspace id="home"><Dashboard /></Workspace>;
}
