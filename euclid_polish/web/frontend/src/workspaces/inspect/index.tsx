/* Inspect workspace (spec §8.6): no tabs. Phase 1 renders the old FITS
   inspector page (`?fits=<project-relative path>`); W-Inspect replaces it in
   phase 3. */
import InspectPage from "../../pages/Inspect";
import { Workspace } from "../../app/workspace";

export default function InspectWorkspace() {
  return <Workspace id="inspect"><InspectPage /></Workspace>;
}
