/* Registers the Realism workspace's inspector kinds (lazy chunks, so
   importing this is cheap):
   - `readiness:<item id>` — one overview checklist item (facts + its fix);
   - `noisepos:<tile>` — one measured Q1 noise position (4 band levels, the
     4×4 sub-tile grids, the depth step, a link to the atlas);
   - `archivefield:<id>` — one multipoint archive field (metadata + a compact
     viewer, links to Visual and the atlas). */
import { lazy } from "react";
import { registerInspector } from "../../app/inspector";

const ReadinessInspector = lazy(() => import("./inspectors").then((m) => ({ default: m.ReadinessInspector })));
const NoisePositionInspector = lazy(() => import("./inspectors").then((m) => ({ default: m.NoisePositionInspector })));
const ArchiveFieldInspector = lazy(() => import("./inspectors").then((m) => ({ default: m.ArchiveFieldInspector })));

export const unregisterReadiness = registerInspector("readiness", ReadinessInspector, {
  title: (id) => `Readiness · ${id}`,
});
export const unregisterNoisePosition = registerInspector("noisepos", NoisePositionInspector, {
  title: (id) => `Noise position · tile ${id}`,
});
export const unregisterArchiveField = registerInspector("archivefield", ArchiveFieldInspector, {
  title: (id) => `Archive field #${id}`,
});
