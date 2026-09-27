/* The atlas's inspector kinds, registered when this module is imported (the
 * Sky workspace imports it; it is light — the cards are lazy — so the shell
 * can import it eagerly and `tile:` / `source:` links open on any page):
 *   tile:<source>/<id>    a real tile with sky actions (the palette's
 *                         `tile:nexus/<n>` = `nexus/f200w-<NNNN>`)
 *   source:<layer>/<id>   a catalogue object / coverage feature;
 *   source:at/<ra>,<dec>  "what covers this point"
 * The cards are lazy (the tile card pulls in the image viewer); the
 * inspector panel provides the Suspense boundary. */
import { lazy } from "react";
import { registerInspector } from "../../../../app/inspector";
import { sourceTargetParts, tileTargetId } from "../layerModel";

const TileCard = lazy(() => import("./TileInspector"));
const SourceCard = lazy(() => import("./SourceInspector"));

function TileInspector({ id }: { id: string }) {
  return <TileCard id={id} />;
}

function SourceInspector({ id }: { id: string }) {
  return <SourceCard id={id} />;
}

export function sourceTitle(id: string): string {
  const parts = sourceTargetParts(id);
  if (!parts) return id;
  if (parts.layer === "at") return `Sky point ${parts.id}`;
  return `${parts.layer} · ${parts.id.length > 28 ? `${parts.id.slice(0, 27)}…` : parts.id}`;
}

let registered = false;

/** Idempotent; runs on import (below). */
export function registerAtlasInspectors(): void {
  if (registered) return;
  registered = true;
  registerInspector("tile", TileInspector, { title: (id) => `Tile ${tileTargetId(id)}` });
  registerInspector("source", SourceInspector, { title: sourceTitle });
}

registerAtlasInspectors();
