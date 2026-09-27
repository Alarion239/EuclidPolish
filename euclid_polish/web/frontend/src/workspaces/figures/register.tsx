/* The Figures inspector kind, registered when the workspace loads:
 *   figure:<saved result id>   a crop saved from a viewer (vr-<24 hex>)
 * The card is lazy; the inspector panel provides the Suspense boundary. */
import { lazy } from "react";
import { registerInspector } from "../../app/inspector";

const FigureCard = lazy(() => import("./FigureInspector"));

function FigureInspector({ id }: { id: string }) {
  return <FigureCard id={id} />;
}

export function figureTitle(id: string): string {
  return `Saved result ${id.length > 14 ? `${id.slice(0, 13)}…` : id}`;
}

let registered = false;

/** Idempotent. */
export function registerFigureInspectors(): void {
  if (registered) return;
  registered = true;
  registerInspector("figure", FigureInspector, { title: figureTitle });
}

registerFigureInspectors();
