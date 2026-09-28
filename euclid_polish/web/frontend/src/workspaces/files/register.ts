/* Registers the inspector kind `fits` (`fits:<project-relative path>`).
   The component is a lazy chunk, so importing this module is cheap: the shell
   (or any page) can import it to make `openInspector({kind: "fits", id})`
   work before Files has been visited. */
import { lazy } from "react";
import { registerInspector } from "../../app/inspector";
import { basename } from "./model";

const FitsInspector = lazy(() => import("./FitsInspector"));

export const unregisterFitsInspector = registerInspector("fits", FitsInspector, {
  // The panel names the kind ("Fits") above the title: the title is the file.
  title: (id) => basename(id),
});
