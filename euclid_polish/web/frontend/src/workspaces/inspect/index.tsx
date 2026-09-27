/* Inspect workspace (spec §8.6): no tabs. The page is ./InspectPage.tsx; the
   `fits` inspector kind is registered by ./register.ts (imported here, and by
   the shell so any page can open `fits:<path>` in the side panel). */
import { Workspace } from "../../app/workspace";
import InspectPage from "./InspectPage";
import "./register";

export default function InspectWorkspace() {
  return <Workspace id="inspect"><InspectPage /></Workspace>;
}
