/* Inspector kind `tile` — a real tile on the atlas (`tile:<source>/<id>`;
 * the palette's `tile:nexus/<n>` means `nexus/f200w-<NNNN>`). It renders the
 * ONE real-tile card (sky/results/RealTileInspector.tsx), the same card as
 * `realtile:` from Real results: image first, then state, actions, the
 * overlay on the sky, facts, models and metrics. */
import RealTileCard from "../../results/RealTileInspector";
import { tileTargetId } from "../layerModel";

export { jwstFilters, overlayTiers, tileFovDeg } from "./overlay";
export { outputOrigin } from "../../results/model";

export default function TileInspector({ id }: { id: string }) {
  return <RealTileCard id={tileTargetId(id)} />;
}
