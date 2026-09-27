/* The inspector kinds of the Sky results tabs, registered when the Sky
 * workspace loads (and again, idempotently, by each results tab):
 *   realtile:<source>/<id>   a real tile of the C9 store — an alias of the
 *                            atlas's `tile:` (replaced by it on open: one card,
 *                            one kind label)
 *   experiment:<id>          a real-data experiment record
 * The cards are lazy (the realtile card pulls in the image viewer); the
 * inspector panel provides the Suspense boundary. */
import { lazy, useEffect } from "react";
import { openInspector, registerInspector, useInspectorKind } from "../../../app/inspector";
import { splitRef } from "./api";

const RealTileCard = lazy(() => import("./RealTileInspector"));
const ExperimentCard = lazy(() => import("./ExperimentInspector"));

/** `realtile:` is an alias: it becomes `tile:` in place (the same card, and
 *  one kind label, "Tile", wherever the card was opened — old links and other
 *  workspaces still say `realtile:`). Without the atlas's `tile` kind (never
 *  in the console: the Sky workspace registers both) it renders the card. */
function RealTileInspector({ id }: { id: string }) {
  const tile = useInspectorKind("tile");
  useEffect(() => { if (tile) openInspector({ kind: "tile", id }, { replace: true }); }, [tile, id]);
  return tile ? null : <RealTileCard id={id} />;
}

function ExperimentInspector({ id }: { id: string }) {
  return <ExperimentCard id={id} />;
}

/** "Tile nexus/f200w-0040": the same title as the atlas's `tile:` kind (one
 *  card, one name; the panel ellipsizes a long id and keeps it in the tooltip). */
export function realtileTitle(id: string): string {
  const [, tile] = splitRef(id);
  return tile ? `Tile ${id}` : id;
}

let registered = false;

/** Idempotent. */
export function registerResultsInspectors(): void {
  if (registered) return;
  registered = true;
  registerInspector("realtile", RealTileInspector, { title: realtileTitle });
  registerInspector("experiment", ExperimentInspector, { title: (id) => `Experiment ${id}` });
}

// Importing this module registers the kinds (the Sky workspace index does,
// so the atlas can open `realtile:` / `experiment:` cards before any results
// tab loaded).
registerResultsInspectors();
