/* "Discovered JWST observations" (JWST tools menu, palette): every MAST
 * observation the discovery found overlapping Euclid Q1 (the `jwst-mast`
 * layer), as a searchable table — row → its source card, "Pair" downloads a
 * JWST × Euclid pair, "Show" flies there. It replaces the legacy JWST × Euclid
 * page's location browser (search, instrument filter, per-location download). */
import { useMemo } from "react";
import { useResource } from "../../../api/query";
import { formatCount, formatDec, formatRA } from "../../../format";
import { Badge, Button, DataTable, Dialog, EmptyState, IconButton, Skeleton, type DataColumn } from "../../../ui";
import { discoverJwst, downloadAllPairs, downloadPair, viewRegion } from "./actions";
import { layerUrl } from "./layerData";
import { normalisePayload, type LayerPayload, type SkyFeature } from "./layerModel";

type Row = {
  key: string; obsId: string; target: string; instrument: string; filters: string; status: string;
  ra: number; dec: number; f: SkyFeature;
};

export function observationRows(features: readonly SkyFeature[]): Row[] {
  return features.map((f) => ({
    key: f.key,
    obsId: String(f.props.obs_id ?? f.key),
    target: String(f.props.target ?? ""),
    instrument: String(f.props.instrument ?? ""),
    filters: String(f.props.filters ?? ""),
    status: String(f.props.status ?? ""),
    ra: f.ra, dec: f.dec, f,
  }));
}

export function JwstObservations({ open, onOpenChange, onShow, view }: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onShow: (ra: number, dec: number) => void;
  view: { ra: number; dec: number; fov: number } | null;
}) {
  const res = useResource<LayerPayload>(open ? layerUrl({ id: "jwst-mast", url: null }) : null, [], { ttl: 60_000 });
  const rows = useMemo(() => observationRows(res.data ? normalisePayload(res.data) : []), [res.data]);
  const columns = useMemo<DataColumn<Row>[]>(() => [
    { id: "obsId", header: "Observation", cell: (r) => <code className="mono">{r.obsId}</code>, width: 200 },
    { id: "target", header: "Target" },
    { id: "instrument", header: "Instrument", width: 110 },
    { id: "filters", header: "Filters" },
    {
      id: "status", header: "Match", width: 110,
      cell: (r) => r.status ? <Badge size="sm" tone={r.status === "exact_intersection" ? "good" : "neutral"}>{r.status.replace(/_/g, " ")}</Badge> : "—",
    },
    { id: "ra", header: "RA", numeric: true, width: 104, cell: (r) => formatRA(r.ra, { digits: 1 }) },
    { id: "dec", header: "Dec", numeric: true, width: 96, cell: (r) => formatDec(r.dec, { digits: 0 }) },
    {
      id: "actions", header: "", sortable: false, filterable: false, csv: false, hideable: false, width: 112,
      cell: (r) => (
        <span className="sky-obs__actions">
          <IconButton icon="globe" size="sm" label={`Show ${r.obsId} on the sky`} onClick={() => { onShow(r.ra, r.dec); onOpenChange(false); }} />
          <Button size="sm" variant="subtle" onClick={() => { void downloadPair({ obs_id: r.obsId }); }}>Pair</Button>
        </span>
      ),
    },
  ], [onShow, onOpenChange]);
  const discoverView = () => {
    if (view) void discoverJwst({ region: viewRegion(view), label: "the current view" });
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange} size="xl" title="Discovered JWST observations"
      description={rows.length ? `${formatCount(rows.length)} MAST observations overlapping Euclid Q1` : "MAST observations overlapping Euclid Q1"}>
      {res.loading ? <Skeleton lines={6} /> : res.error ? (
        <EmptyState icon="warn" title="Could not load the discovered observations" action={<Button size="sm" onClick={res.reload}>Retry</Button>}>
          {res.error.message}
        </EmptyState>
      ) : rows.length === 0 ? (
        <EmptyState icon="globe" title="No JWST discovery yet"
          action={<Button variant="primary" size="sm" disabled={!view} onClick={discoverView}>Discover in this view</Button>}>
          Discovery queries MAST for JWST imaging and intersects it with the Q1 tiles.
        </EmptyState>
      ) : (
        <DataTable rows={rows} columns={columns} rowKey={(r) => r.key} aria-label="Discovered JWST observations"
          dense height={420} exportName="jwst-observations" urlKey="obs" defaultSort={[{ id: "target", desc: false }]}
          inspect={(r) => r.f.inspect}
          toolbar={(
            <>
              <Button size="sm" disabled={!view} onClick={discoverView}>Discover in view</Button>
              <Button size="sm" onClick={() => { void downloadAllPairs(); }}>Download every pair…</Button>
            </>
          )} />
      )}
    </Dialog>
  );
}
