/* Files workspace, `/files` (console regrouping; was /inspect): what is inside
   a FITS file, and where it came from. No tabs. The page is ./FilesPage.tsx
   (`?fits=`, `?path=`, `dir`, `q`, `hdu`, `slice`, `view`); the `fits`
   inspector kind is registered by ./register.ts (imported here, and by the
   shell so any page can open `fits:<path>` in the side panel). */
import { Workspace } from "../../app/workspace";
import FilesPage from "./FilesPage";
import "./register";

export default function FilesWorkspace() {
  return <Workspace id="files"><FilesPage /></Workspace>;
}
