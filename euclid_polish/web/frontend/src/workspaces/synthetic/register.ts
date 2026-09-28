/* Registers the Synthetic workspace's inspector kinds. The cards are lazy
   chunks, so importing this module is cheap: the shell imports it eagerly so
   these links open before the workspace has been visited. Idempotent.
   - `readiness:<item id>` — one Status row (facts, fingerprints, its fix);
   - `noisepos:<tile>` — one measured Q1 noise position (4 band levels, the
     4×4 sub-tile grids, the depth step, a link to the atlas);
   - `archivefield:<id>` — one multipoint archive field (metadata + a compact
     viewer, links to Fields and the atlas);
   - `star:<id>`, `truth:<split>/<index>/<row>`, `psf:<cluster index>`,
     `tng:<subhalo id>` — a catalogue star, a record's truth source, a PSF
     cluster and a TNG galaxy. */
import { lazy } from "react";
import { registerInspector } from "../../app/inspector";
import { parseClusterId, parseTruthId } from "./ids";

const loadStatus = () => import("./inspectors");
const loadData = () => import("./dataInspectors");
const ReadinessInspector = lazy(() => loadStatus().then((m) => ({ default: m.ReadinessInspector })));
const NoisePositionInspector = lazy(() => loadStatus().then((m) => ({ default: m.NoisePositionInspector })));
const ArchiveFieldInspector = lazy(() => loadStatus().then((m) => ({ default: m.ArchiveFieldInspector })));
const StarInspector = lazy(() => loadData().then((m) => ({ default: m.StarInspector })));
const TruthInspector = lazy(() => loadData().then((m) => ({ default: m.TruthInspector })));
const PsfInspector = lazy(() => loadData().then((m) => ({ default: m.PsfInspector })));
const TngInspector = lazy(() => loadData().then((m) => ({ default: m.TngInspector })));

export function truthTitle(id: string): string {
  const t = parseTruthId(id);
  return t ? `Source ${t.split} ${t.index}·${t.row}` : `Source ${id}`;
}

let registered = false;

export function registerSyntheticInspectors(): void {
  if (registered) return;
  registered = true;
  registerInspector("readiness", ReadinessInspector, { title: (id) => `Status · ${id}` });
  registerInspector("noisepos", NoisePositionInspector, { title: (id) => `Noise position · tile ${id}` });
  registerInspector("archivefield", ArchiveFieldInspector, { title: (id) => `Archive field #${id}` });
  registerInspector("star", StarInspector, { title: (id) => `Star ${id}` });
  registerInspector("truth", TruthInspector, { title: truthTitle });
  registerInspector("psf", PsfInspector, {
    title: (id) => `PSF cluster ${String(parseClusterId(id) ?? id).padStart(3, "0")}`,
  });
  registerInspector("tng", TngInspector, { title: (id) => `TNG ${id}` });
}

registerSyntheticInspectors();
