/* The inspector kinds of the Sky results tabs, registered when the Sky
 * workspace loads (and again, idempotently, by each results tab):
 *   realtile:<source>/<id>   a real tile of the C9 store (Real results card)
 *   experiment:<id>          a real-data experiment record
 * The cards are lazy (the realtile card pulls in the image viewer); the
 * inspector panel provides the Suspense boundary. */
import { lazy } from "react";
import { registerInspector } from "../../../app/inspector";
import { splitRef } from "./api";

const RealTileCard = lazy(() => import("./RealTileInspector"));
const ExperimentCard = lazy(() => import("./ExperimentInspector"));

function RealTileInspector({ id }: { id: string }) {
  return <RealTileCard id={id} />;
}

function ExperimentInspector({ id }: { id: string }) {
  return <ExperimentCard id={id} />;
}

export function realtileTitle(id: string): string {
  const [source, tile] = splitRef(id);
  const short = tile.length > 26 ? `${tile.slice(0, 25)}…` : tile;
  return tile ? `${source} · ${short}` : id;
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
