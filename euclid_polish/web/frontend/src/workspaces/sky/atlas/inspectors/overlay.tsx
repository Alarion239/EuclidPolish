/* "Overlay on the sky" for a real tile: put its LR / model / JWST pixels on
 * the atlas as a FITS overlay (the atlas `img` URL param) and fly there.
 * Used by the one real-tile card (sky/results/RealTileInspector.tsx), so the
 * overlay action is there whether the card was opened from the atlas or from
 * Real results. */
import { useState } from "react";
import { Button, Field, Select } from "../../../../ui";
import { useShowOnSky } from "../engineHooks";
import { tierLabel } from "../pixelOverlays";
import "../../results/results.css";

export type OverlayTile = {
  ref: string; ra: number | null; dec: number | null; has_jwst?: boolean;
  shape?: [number, number] | null; pixscale?: number | null;
  extras?: Record<string, unknown>; image_urls?: Record<string, string>;
};

const BANDS = [{ value: "VIS", label: "VIS" }, { value: "Y_E", label: "Y" }, { value: "J_E", label: "J" }, { value: "H_E", label: "H" }];

/** A field of view that frames the tile (2.5 × its side). */
export function tileFovDeg(card: { shape?: [number, number] | null; pixscale?: number | null }): number {
  const side = Math.max(...(card.shape ?? [256, 256])) * (card.pixscale ?? 0.1);
  return Math.max(0.004, (side / 3600) * 2.5);
}

/** The JWST filters of a tile ("" = the server's default, its first). */
export function jwstFilters(card: { extras?: Record<string, unknown>; has_jwst?: boolean }): string[] {
  if (!card.has_jwst) return [];
  const bands = card.extras?.jwst_bands;
  const fromBands = Array.isArray(bands)
    ? bands.map((b) => (b && typeof b === "object" ? String((b as { filter?: unknown }).filter ?? "") : "")).filter(Boolean)
    : [];
  if (fromBands.length) return [...new Set(fromBands)];
  return typeof card.extras?.filter === "string" && card.extras.filter ? [card.extras.filter] : [""];
}

/** Tiers offered for sky overlays: LR, every model output, JWST. */
export function overlayTiers(card: { image_urls?: Record<string, string>; has_jwst?: boolean }): string[] {
  const models = Object.keys(card.image_urls ?? {}).filter((t) => t.startsWith("m:"));
  return ["lr", ...models, ...(card.has_jwst ? ["jwst"] : [])];
}

export function OverlayControl({ card }: { card: OverlayTile }) {
  const tiers = overlayTiers(card);
  const filters = jwstFilters(card);
  const [tier, setTier] = useState(tiers.find((t) => t.startsWith("m:")) ?? "lr");
  const [band, setBand] = useState("VIS");
  const [filter, setFilter] = useState(filters[0] ?? "");
  const showOnSky = useShowOnSky();
  const { ra, dec } = card;
  const add = () => {
    if (ra == null || dec == null) return;
    const b = tier === "jwst" ? (filters.length > 1 ? filter : "") : band;
    showOnSky({ overlays: [{ ref: card.ref, tier, band: b }], view: { ra, dec, fov: tileFovDeg(card) } });
  };
  return (
    <div className="res-overlay">
      <Field label="Image"><Select value={tier} onChange={setTier} options={tiers.map((t) => ({ value: t, label: tierLabel(t) }))} /></Field>
      {tier !== "jwst" && <Field label="Band"><Select value={band} onChange={setBand} options={BANDS} /></Field>}
      {tier === "jwst" && filters.length > 1 && (
        <Field label="Filter"><Select value={filter} onChange={setFilter} options={filters.map((f) => ({ value: f, label: f }))} /></Field>
      )}
      <Button onClick={add} icon="image" disabled={ra == null || dec == null}>Add to the sky</Button>
    </div>
  );
}
