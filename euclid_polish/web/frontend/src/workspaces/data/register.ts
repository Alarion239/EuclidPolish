/* Registers the Data workspace's inspector kinds — `star:<id>`,
   `truth:<split>/<index>/<row>`, `psf:<cluster index>`, `tng:<subhalo id>`.
   The cards are one lazy chunk, so importing this module is cheap: the shell
   (or any page) can import it to make those links open before the Data
   workspace has been visited. Idempotent. */
import { lazy } from "react";
import { registerInspector } from "../../app/inspector";
import { parseClusterId, parseTruthId } from "./ids";

const load = () => import("./inspectors");
const StarInspector = lazy(() => load().then((m) => ({ default: m.StarInspector })));
const TruthInspector = lazy(() => load().then((m) => ({ default: m.TruthInspector })));
const PsfInspector = lazy(() => load().then((m) => ({ default: m.PsfInspector })));
const TngInspector = lazy(() => load().then((m) => ({ default: m.TngInspector })));

export function truthTitle(id: string): string {
  const t = parseTruthId(id);
  return t ? `Source ${t.split} ${t.index}·${t.row}` : `Source ${id}`;
}

let registered = false;

export function registerDataInspectors(): void {
  if (registered) return;
  registered = true;
  registerInspector("star", StarInspector, { title: (id) => `Star ${id}` });
  registerInspector("truth", TruthInspector, { title: truthTitle });
  registerInspector("psf", PsfInspector, {
    title: (id) => `PSF cluster ${String(parseClusterId(id) ?? id).padStart(3, "0")}`,
  });
  registerInspector("tng", TngInspector, { title: (id) => `TNG ${id}` });
}

registerDataInspectors();
