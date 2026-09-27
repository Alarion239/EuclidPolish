/* Inspector kind `source` — a catalogue object or coverage feature on the
 * atlas: `source:<layer>/<id>` (C9 inspect links: `lens-candidates/<id>`,
 * `stars/<row>`, `q1-tiles/<tile>`, `jwst-mast/<obs_id>`, …), plus the sky
 * point card `source:at/<ra>,<dec>` ("what covers this point").
 * The feature comes from the layer payload (the atlas's cache entry); PSF
 * clusters and catalogue objects with a reconstruction get a mini viewer
 * (SourceViewer.tsx). */
import { useMemo, type ReactNode } from "react";
import { openInspector } from "../../../../app/inspector";
import { useResource } from "../../../../api/query";
import { Badge, Button, Callout, DefList, Section, Skeleton, type MenuItem } from "../../../../ui";
import { cacheTileAt, discoverJwst, downloadPair, runNexusProduction } from "../actions";
import { useFlyTo } from "../engineHooks";
import { layerUrl } from "../layerData";
import {
  normalisePayload, sourceTargetParts, withClientLayers, type LayerPayload, type LayersResponse, type SkyFeature,
} from "../layerModel";
import { parsePointId, pointTargetId } from "../urlState";
import { CardActions, MoreMenu, PositionValue, positionMenuItems, propValue } from "./common";
import { PointCard } from "./PointCard";
import { SourceViewer } from "./SourceViewer";
import "../atlas.css";

const HIDDEN = new Set(["ra", "dec", "label", "polygons", "polygon"]);

function FeatureFacts({ f, payload }: { f: SkyFeature; payload: LayerPayload | null }) {
  const bits = payload && "flag_bits" in payload ? payload.flag_bits : undefined;
  const rows: ([string, ReactNode] | null)[] = Object.entries(f.props)
    .filter(([k]) => !HIDDEN.has(k))
    .map(([k, v]) => {
      if (k === "flags" && bits && typeof v === "number") {
        const bands = Object.entries(bits).filter(([, b]) => (v & b) !== 0).map(([band]) => band);
        return ["valid cutouts", bands.length ? bands.join(", ") : "none"];
      }
      if (k === "levels_e" && Array.isArray(v)) return ["sky level e⁻ (VIS, Y, J, H)", propValue(v)];
      return [k.replace(/_/g, " "), propValue(v)];
    });
  return <DefList dense items={[["position", <PositionValue ra={f.ra} dec={f.dec} />], ...rows]} />;
}

function LayerFeature({ layer, fid }: { layer: string; fid: string }) {
  const catalogue = useResource<LayersResponse>("/api/sky/layers", [], { ttl: 60_000 });
  const info = useMemo(() => withClientLayers(catalogue.data?.layers ?? []).find((l) => l.id === layer), [catalogue.data, layer]);
  const url = layerUrl({ id: layer, url: info?.url ?? null });
  const payload = useResource<LayerPayload>(url, [], { ttl: 60_000 });
  const feature = useMemo(() => (payload.data ? normalisePayload(payload.data).find((f) => f.key === fid) ?? null : null), [payload.data, fid]);
  const fly = useFlyTo();
  if (payload.loading || catalogue.loading) return <Skeleton lines={5} />;
  if (payload.error) {
    return (
      <Callout tone="bad" title="Could not load the layer" action={<Button size="sm" onClick={payload.reload}>Retry</Button>}>
        {payload.error.message}
      </Callout>
    );
  }
  if (!feature) return <Callout tone="warn" title="Not found">No “{fid}” in the {info?.label ?? layer} layer (it may have changed since the link was made).</Callout>;
  const f = feature;
  const fov = f.polygon || f.radius ? Math.max(0.01, f.sizeDeg * 1.6) : 0.02;
  const more: MenuItem[] = [
    { label: "Cache a 25.6″ tile here…", onSelect: () => { void cacheTileAt(f.ra, f.dec); } },
    { label: "Download a JWST × Euclid pair here…", onSelect: () => { void downloadPair({ ra: f.ra, dec: f.dec }); } },
    { type: "separator" },
    ...positionMenuItems(f.ra, f.dec, fov),
  ];
  const obsId = layer === "jwst-mast" ? String(f.props.obs_id ?? f.key) : null;
  return (
    <div className="sky-card">
      <div className="sky-card__badges">
        <Badge>{info?.label ?? layer}</Badge>
        {typeof f.props.state === "string" && <Badge tone={f.props.state === "rejected" ? "bad" : "neutral"} dot>{f.props.state}</Badge>}
        {typeof f.props.grade === "string" && <Badge tone="accent">grade {f.props.grade}</Badge>}
        {typeof f.props.field === "string" && <Badge>{f.props.field}</Badge>}
      </div>
      <h3 className="sky-card__title">{f.label}</h3>
      <CardActions>
        <Button size="sm" icon="globe" onClick={() => fly({ ra: f.ra, dec: f.dec, fov })}>Show on sky</Button>
        <Button size="sm" onClick={() => openInspector({ kind: "source", id: pointTargetId(f.ra, f.dec) })}>What covers this point</Button>
        {obsId && <Button size="sm" variant="primary" onClick={() => { void downloadPair({ obs_id: obsId }); }}>Download pair</Button>}
        {layer === "nexus-footprint" && (
          <Button size="sm" onClick={() => { void runNexusProduction(f.key); }}>Run production on stale tiles</Button>
        )}
        {layer === "q1-fields" && (
          <Button size="sm" onClick={() => { void discoverJwst({ fields: f.key, label: f.key }); }}>Discover JWST</Button>
        )}
        <MoreMenu items={more} />
      </CardActions>
      <SourceViewer layer={layer} fid={f.key} />
      <Section title="Properties" collapsible defaultOpen>
        <FeatureFacts f={f} payload={payload.data} />
      </Section>
    </div>
  );
}

export default function SourceInspector({ id }: { id: string }) {
  const parts = sourceTargetParts(id);
  if (!parts) return <Callout tone="warn" title="Not a sky source">Source ids look like <code>lens-candidates/&lt;id&gt;</code> or <code>at/268.46,65.2</code>.</Callout>;
  if (parts.layer === "at") {
    const p = parsePointId(parts.id);
    if (!p) return <Callout tone="warn" title="Bad position">“{parts.id}” is not “ra,dec” in degrees.</Callout>;
    return <PointCard ra={p.ra} dec={p.dec} />;
  }
  return <LayerFeature layer={parts.layer} fid={parts.id} />;
}
