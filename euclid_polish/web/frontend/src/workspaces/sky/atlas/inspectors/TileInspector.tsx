/* Inspector kind `tile` — a real tile on the atlas (`tile:<source>/<id>`;
 * the palette's `tile:nexus/<n>` means `nexus/f200w-<NNNN>`).
 *
 * Card (`GET /api/real/<source>/<id>`, C9): SR state vs the production
 * model, position, containing Q1 tile, computed models, disk; a mini viewer
 * (`real` collection: LR, the first model tier, JWST); actions: show on sky,
 * compare models (Experiments, tile preselected), run production, overlay
 * LR / SR / JWST pixels on the sky, cache a 25.6″ tile / download a JWST ×
 * Euclid pair at its centre, ESASky / SIMBAD, copy. */
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useResource } from "../../../../api/query";
import { openInspector } from "../../../../app/inspector";
import { formatBytes, formatDateTime, formatNumber } from "../../../../format";
import { Badge, Button, Callout, DefList, Field, Section, Select, Skeleton, Tooltip, type MenuItem, type Tone } from "../../../../ui";
import { ImageViewer } from "../../../../viewer";
import { buildPairInput, cacheTileAt, downloadPair, runModels, runNexusProduction } from "../actions";
import { useCompareModels, useFlyTo, useShowOnSky } from "../engineHooks";
import { tileTargetId } from "../layerModel";
import { tierLabel } from "../pixelOverlays";
import { CardActions, MoreMenu, PositionValue, positionMenuItems, splitRef } from "./common";
import "../atlas.css";

type ModelOutput = {
  state?: string; legacy?: boolean; label?: string; kind?: string; created?: string | null;
  experiment_id?: string | null; origin?: string; metrics?: { summary?: Record<string, unknown> } | null; image_url?: string;
  member_labels?: string[]; combiner_kind?: string | null;
};

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** The real-data metrics of an output, pooled over bands (C9 `summary`). */
export function metricChips(summary: Record<string, unknown> | null | undefined): string[] {
  if (!summary) return [];
  const out: string[] = [];
  const holes = num(summary.hole_pct), r = num(summary.median_R), flux = num(summary.flux_ratio);
  if (holes != null) out.push(`holes ${formatNumber(holes, { digits: 1 })}%`);
  if (r != null) out.push(`R̃ ${formatNumber(r, { digits: 2 })}`);
  if (flux != null) out.push(`flux ${formatNumber(flux, { digits: 3 })}`);
  return out;
}

/** Where an output came from (its provenance, in words). */
export function outputOrigin(m: ModelOutput): string {
  const parts: string[] = [];
  if (m.legacy) parts.push(`legacy ${m.origin ?? "record"}`);
  if (m.member_labels?.length) parts.push(`${m.member_labels.length} member${m.member_labels.length === 1 ? "" : "s"}`);
  if (m.combiner_kind) parts.push(m.combiner_kind.replace(/_/g, " "));
  return parts.join(" · ");
}

export type TileCard = {
  source: string; id: string; ref: string; label: string; ra: number; dec: number; field?: string | null;
  shape?: [number, number]; pixscale?: number; bands?: string[]; model_ready?: boolean; tiers?: string[];
  has_jwst?: boolean; polygon?: [number, number][];
  extras?: Record<string, unknown>;
  models?: Record<string, ModelOutput>;
  production_state?: "current" | "stale" | "missing" | string;
  runnable_models?: string[];
  experiments?: string[];
  disk?: { total_bytes?: number; tile_bytes?: number; output_bytes?: number; cache_bytes?: number; legacy_bytes?: number };
  q1_tile?: { tile?: string; field?: string | null; levels_e?: number[] | null; rejected?: string | null } | null;
  image_urls?: Record<string, string>;
  viewer?: { collection: string; params: Record<string, string>; id: string };
};

const STATE_TONE: Record<string, Tone> = { current: "good", stale: "warn", missing: "neutral", unavailable: "neutral" };
const BANDS = [{ value: "VIS", label: "VIS" }, { value: "Y_E", label: "Y" }, { value: "J_E", label: "J" }, { value: "H_E", label: "H" }];

export function tileFovDeg(card: Pick<TileCard, "shape" | "pixscale">): number {
  const side = Math.max(...(card.shape ?? [256, 256])) * (card.pixscale ?? 0.1);
  return Math.max(0.004, (side / 3600) * 2.5);
}

/** The JWST filters of a tile ("" = the server's default, its first). */
export function jwstFilters(card: Pick<TileCard, "extras" | "has_jwst">): string[] {
  if (!card.has_jwst) return [];
  const bands = card.extras?.jwst_bands;
  const fromBands = Array.isArray(bands)
    ? bands.map((b) => (b && typeof b === "object" ? String((b as { filter?: unknown }).filter ?? "") : "")).filter(Boolean)
    : [];
  if (fromBands.length) return [...new Set(fromBands)];
  return typeof card.extras?.filter === "string" && card.extras.filter ? [card.extras.filter] : [""];
}

/** Tiers offered for sky overlays: LR, every model output, JWST. */
export function overlayTiers(card: TileCard): string[] {
  const models = Object.keys(card.image_urls ?? {}).filter((t) => t.startsWith("m:"));
  return ["lr", ...models, ...(card.has_jwst ? ["jwst"] : [])];
}

function OverlayControl({ card }: { card: TileCard }) {
  const tiers = overlayTiers(card);
  const filters = jwstFilters(card);
  const [tier, setTier] = useState(tiers.find((t) => t.startsWith("m:")) ?? "lr");
  const [band, setBand] = useState("VIS");
  const [filter, setFilter] = useState(filters[0] ?? "");
  const showOnSky = useShowOnSky();
  const add = () => {
    const b = tier === "jwst" ? (filters.length > 1 ? filter : "") : band;
    showOnSky({ overlays: [{ ref: card.ref, tier, band: b }], view: { ra: card.ra, dec: card.dec, fov: tileFovDeg(card) } });
  };
  return (
    <div className="sky-card__overlay">
      <Field label="Overlay pixels"><Select value={tier} onChange={setTier} options={tiers.map((t) => ({ value: t, label: tierLabel(t) }))} /></Field>
      {tier !== "jwst" && <Field label="Band"><Select value={band} onChange={setBand} options={BANDS} /></Field>}
      {tier === "jwst" && filters.length > 1 && (
        <Field label="Filter"><Select value={filter} onChange={setFilter} options={filters.map((f) => ({ value: f, label: f }))} /></Field>
      )}
      <Button onClick={add} icon="image">On the sky</Button>
    </div>
  );
}

export default function TileInspector({ id }: { id: string }) {
  const ref = tileTargetId(id);
  const [source, tileId] = splitRef(ref);
  const card = useResource<TileCard>(
    tileId ? `/api/real/${encodeURIComponent(source)}/${encodeURIComponent(tileId)}` : null, [], { ttl: 30_000 },
  );
  const navigate = useNavigate();
  const fly = useFlyTo();
  const compareModels = useCompareModels();
  const c = card.data;
  const params = useMemo(() => ({ source }), [source]);
  if (!tileId) return <Callout tone="warn" title="Not a real tile">Tile ids look like <code>nexus/f200w-0012</code> or <code>archive/007</code>.</Callout>;
  if (card.loading) return <Skeleton lines={6} />;
  if (!c) {
    return (
      <Callout tone="bad" title={card.error?.status === 404 ? "Unknown tile" : "Could not load the tile"}
        action={<Button size="sm" onClick={card.reload}>Retry</Button>}>
        {card.error?.message ?? "No data."}
      </Callout>
    );
  }
  const models = Object.entries(c.models ?? {});
  const firstModel = Object.keys(c.image_urls ?? {}).find((t) => t.startsWith("m:"));
  const viewerTiers = ["lr", ...(firstModel ? [firstModel] : []), ...(c.has_jwst ? ["jwst"] : [])];
  const more: MenuItem[] = [
    { label: "Cache a 25.6″ tile at its centre…", onSelect: () => { void cacheTileAt(c.ra, c.dec); } },
    { label: "Download a JWST × Euclid pair here…", onSelect: () => { void downloadPair({ ra: c.ra, dec: c.dec }); } },
    ...(c.source === "nexus" ? [{
      label: "Run production on every stale NEXUS tile…",
      onSelect: () => { void runNexusProduction(typeof c.extras?.field_id === "string" ? c.extras.field_id : undefined); },
    }] : []),
    { label: "Open in Real results", onSelect: () => navigate(`/sky/results?inspect=${encodeURIComponent(`realtile:${c.ref}`)}`) },
    { type: "separator" },
    ...positionMenuItems(c.ra, c.dec, tileFovDeg(c)),
  ];
  const q1 = c.q1_tile;
  return (
    <div className="sky-card">
      <div className="sky-card__badges">
        <Badge tone={STATE_TONE[c.production_state ?? ""] ?? "neutral"} dot>production {c.production_state ?? "—"}</Badge>
        {c.field && <Badge>{c.field}</Badge>}
        {c.has_jwst && <Badge tone="info">JWST</Badge>}
        <Badge>{c.source}</Badge>
      </div>
      <CardActions>
        <Button size="sm" icon="globe" onClick={() => fly({ ra: c.ra, dec: c.dec, fov: tileFovDeg(c) })}>Show on sky</Button>
        <Button size="sm" variant="primary" onClick={() => compareModels([c.ref])}>Compare models…</Button>
        {c.source === "pair" && !c.model_ready ? (
          <Button size="sm" onClick={() => { void buildPairInput(c.id, c.label); }}>Build LR + run production</Button>
        ) : (
          <Button size="sm" disabled={!c.model_ready || !(c.runnable_models ?? []).includes("production")}
            onClick={() => { void runModels([c.ref], ["production"]); }}>Run production</Button>
        )}
        <MoreMenu items={more} />
      </CardActions>
      <div className="sky-card__viewer">
        <ImageViewer collection="real" params={params} initialId={tileId} tiers={viewerTiers}
          id={`sky-tile-${c.ref}`} toolbar="compact" nav={false} />
      </div>
      <OverlayControl card={c} />
      <DefList dense items={[
        ["position", <PositionValue ra={c.ra} dec={c.dec} />],
        c.shape ? ["grid", <span className="mono">{c.shape[1]} × {c.shape[0]} px · {c.pixscale ?? "—"}″/px</span>] : null,
        q1 ? ["Q1 tile", <span className="mono">{q1.tile}{q1.levels_e ? ` · VIS sky ${formatNumber(q1.levels_e[0], { digits: 1 })} e⁻` : ""}{q1.rejected ? " · rejected" : ""}</span>] : ["Q1 tile", "outside the committed Q1 polygons"],
        c.disk?.total_bytes != null ? ["disk", formatBytes(c.disk.total_bytes)] : null,
        c.experiments?.length ? ["experiments", c.experiments.length] : null,
      ]} />
      <Section title="Models" sub={models.length ? `${models.length} computed` : "none yet"} collapsible defaultOpen>
        {models.length === 0 ? <p className="muted">No SR for this tile yet — run production or compare models.</p> : (
          <ul className="sky-card__models">
            {models.map(([spec, m]) => {
              const chips = metricChips(m.metrics?.summary);
              const origin = outputOrigin(m);
              return (
                <li key={spec}>
                  <Badge size="sm" tone={STATE_TONE[m.state ?? ""] ?? "neutral"} dot>{m.state ?? "?"}</Badge>
                  <Tooltip content={[m.label, origin, m.created ? formatDateTime(m.created) : null].filter(Boolean).join(" · ")}>
                    <code className="mono" tabIndex={0}>{spec}</code>
                  </Tooltip>
                  {chips.map((c) => <span key={c} className="sky-card__metric mono">{c}</span>)}
                  {m.experiment_id && (
                    <button type="button" className="sky-card__link mono"
                      onClick={() => openInspector({ kind: "experiment", id: m.experiment_id! })}>
                      {m.experiment_id}
                    </button>
                  )}
                  {!chips.length && !m.experiment_id && <span className="muted">{origin || m.label}</span>}
                </li>
              );
            })}
          </ul>
        )}
      </Section>
    </div>
  );
}
