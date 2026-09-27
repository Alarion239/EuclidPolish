/* The ONE real-tile card. Inspector kinds `tile:<source>/<id>` (the atlas; the
 * palette's `tile:nexus/<n>`) and `realtile:<source>/<id>` (Real results,
 * Experiments, Catalog eval) both render it, from GET /api/real/<source>/<id>
 * (C9) — the same design wherever the card was opened.
 *
 * Image first: the viewer (`real` collection with ONLY this tile's own tiers:
 * LR, each computed model output, JWST; residuals and export in its bar) is
 * the top of the card, so it gets the inspector's full height; it opens on
 * TWO frames (LR and the first model, cardViewerTiers) so they are large, and
 * "Open large" puts it in focus mode. Under it: one status line, the actions (show on sky, compare models, run models or build
 * a pair's LR, FITS download, more), pixel overlays on the sky, the facts, the
 * models table (a row shows that model in the viewer), the picked model's
 * per-band metrics, experiments, and for eval objects the catalogue-eval
 * provenance. */
import { useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { formatBytes, formatDateTime, formatNumber } from "../../../format";
import {
  Badge, Button, Callout, DataTable, DefList, Menu, Section, Skeleton, copyText, toast,
  type DataColumn, type MenuItem,
} from "../../../ui";
import { ImageViewer, type ViewerApi } from "../../../viewer";
import { VIcon } from "../../../viewer/icons";
import { buildPairInput, cacheTileAt, downloadPair, runNexusProduction } from "../atlas/actions";
import { useCompareModels, useFlyTo } from "../atlas/engineHooks";
import { MoreMenu, positionMenuItems } from "../atlas/inspectors/common";
import { OverlayControl, tileFovDeg } from "../atlas/inspectors/overlay";
import { deleteOutputs } from "./actions";
import { BANDS, splitRef, URLS, type CardModel, type EvalObjectCard, type TileCard } from "./api";
import { BandMetricsTable, CoreWeights, MetricNote, Position, StateBadge, downloadUrl } from "./common";
import { useFollowViewer } from "./follow";
import {
  bandLabel, cardViewerTiers, defaultRunSpecs, formatMetric, outputOrigin, realTileViewerParams, sortSpecs, specShort,
} from "./model";
import { RunModelsPopover } from "./RunModels";
import "./results.css";

type ModelRow = CardModel & { spec: string };

const MODEL_COLUMNS: DataColumn<ModelRow>[] = [
  { id: "spec", header: "Model", width: 72, cell: (m) => <code className="mono">{specShort(m.spec)}</code>, filterText: (m) => `${m.spec} ${m.label ?? ""}` },
  { id: "state", header: "State", width: 84, cell: (m) => <StateBadge state={m.state} title={[m.label, outputOrigin(m), m.created ? formatDateTime(m.created) : null].filter(Boolean).join(" · ") || undefined} /> },
  { id: "holes", header: "Holes %", width: 92, numeric: true, accessor: (m) => m.metrics?.summary?.hole_pct_max ?? null,
    cell: (m) => formatMetric("hole_pct", m.metrics?.summary?.hole_pct_max) },
  { id: "R", header: "R̃", headerText: "Median R", width: 56, numeric: true, accessor: (m) => m.metrics?.summary?.median_R ?? null,
    cell: (m) => formatMetric("median_R", m.metrics?.summary?.median_R) },
  { id: "legacy", header: "Origin", accessor: (m) => outputOrigin(m) || "store", hidden: true },
  { id: "created", header: "Created", accessor: (m) => m.created ?? "", cell: (m) => (m.created ? formatDateTime(m.created) : "—"), hidden: true },
];

function EvalProvenance({ id }: { id: string }) {
  const card = useResource<EvalObjectCard>(URLS.evalObject(id), [], { ttl: 30_000 });
  const c = card.data;
  if (card.loading) return <Skeleton lines={3} />;
  if (!c) return <p className="muted res-note">{card.error?.message ?? "No catalogue-eval record."}</p>;
  const members = c.members;
  const prov = c.provenance?.[0];
  return (
    <>
      <DefList dense items={[
        ["SR model", <StateBadge state={c.state ?? "unknown"} title={c.state_reason} />],
        c.state_reason ? ["why", <span className="res-note">{c.state_reason}</span>] : null,
        ["made by", members ? `${members.member_labels?.length ?? 0} members · ${members.combiner_kind ?? (members.combiner_kind === null ? "member mean" : "combiner not recorded")}` : "not recorded"],
        ["now", c.current ? `${c.current.n_members} starfull · ${c.current.combiner_kind ?? "member mean"}` : "—"],
        prov ? ["provenance", <span className="mono">{prov.id} · {prov.git ?? "?"}{prov.dirty ? " (dirty)" : ""} · {prov.created_at ? formatDateTime(prov.created_at) : ""}</span>] : null,
        c.disagreement?.pca_n ? ["disagreement", `${c.disagreement.pca_n} PCs`] : null,
      ]} />
      {!!Object.keys(c.downloads ?? {}).length && (
        <div className="res-chips">
          {Object.entries(c.downloads ?? {}).map(([tier, url]) => (
            <Button key={tier} size="sm" variant="ghost" icon="download" href={url} download>{tier}</Button>
          ))}
        </div>
      )}
    </>
  );
}

export default function RealTileInspector({ id }: { id: string }) {
  const [source, tileId] = splitRef(id);
  const card = useResource<TileCard>(tileId ? URLS.card(id) : null, [], { ttl: 20_000 });
  const navigate = useNavigate();
  const location = useLocation();
  const fly = useFlyTo();
  const compareModels = useCompareModels();
  const c = card.data;
  const specs = useMemo(() => sortSpecs(Object.keys(c?.models ?? {})), [c?.models]);
  const [picked, setPicked] = useState<string | null>(null);
  const [api, setApi] = useState<ViewerApi | null>(null);
  const focus = picked && c?.models?.[picked] ? picked : (specs.find((s) => c?.models?.[s]?.metrics?.per_band) ?? specs[0] ?? null);
  const params = useMemo(() => realTileViewerParams(source, specs), [source, specs]);
  const viewerTiers = useMemo(() => cardViewerTiers(specs, !!c?.has_jwst), [specs, c?.has_jwst]);
  // The viewer walks every tile of the source: keep it on this one, with these tiers.
  const follow = useFollowViewer(tileId, viewerTiers);
  const rows = useMemo<ModelRow[]>(() => specs.map((spec) => ({ spec, ...(c?.models?.[spec] ?? {}) })), [specs, c?.models]);

  if (!tileId) return <Callout tone="warn" title="Not a real tile">Real tiles look like <code>nexus/f200w-0012</code>, <code>archive/007</code> or <code>eval/&lt;id&gt;</code>.</Callout>;
  if (card.loading) return <Skeleton lines={8} />;
  if (!c) {
    return (
      <Callout tone="bad" title={card.error?.status === 404 ? "Unknown real tile" : "Could not load the tile"}
        action={<Button size="sm" onClick={card.reload}>Retry</Button>}>
        {card.error?.message ?? "No data."}
      </Callout>
    );
  }
  const hasPos = c.ra != null && c.dec != null;
  const ra = c.ra ?? 0, dec = c.dec ?? 0;
  const fov = tileFovDeg(c);
  const imageTiers = Object.keys(c.image_urls ?? { lr: "" });
  const downloads: MenuItem[] = imageTiers.map((tier) => ({
    type: "sub", label: tier === "lr" ? "LR" : tier === "jwst" ? "JWST" : specShort(tier.slice(2)),
    items: (tier === "jwst" ? [""] : [...BANDS]).map((band) => ({
      label: band ? bandLabel(band) : "First filter",
      onSelect: () => downloadUrl(URLS.image(c.ref, tier, band || "VIS")),
    })),
  }));
  const onResults = location.pathname === "/sky/results";
  const more: MenuItem[] = [
    ...(hasPos ? [
      { label: "Cache a 25.6″ tile at its centre…", onSelect: () => { void cacheTileAt(ra, dec); } },
      { label: "Download a JWST × Euclid pair here…", onSelect: () => { void downloadPair({ ra, dec }); } },
    ] : []),
    ...(c.source === "nexus" ? [{
      label: "Run production on every stale NEXUS tile…",
      onSelect: () => { void runNexusProduction(typeof c.extras?.field_id === "string" ? c.extras.field_id : undefined); },
    }] : []),
    ...(onResults ? [] : [{ label: "Open in Real results", onSelect: () => navigate(`/sky/results?inspect=${encodeURIComponent(`tile:${c.ref}`)}`) }]),
    { label: "Copy the tile ref", onSelect: () => { void copyText(c.ref).then((ok) => { if (ok) toast.success(`Copied ${c.ref}`); }); } },
    ...(hasPos ? [{ type: "separator" as const }, ...positionMenuItems(ra, dec, fov)] : []),
    { type: "separator" },
    { label: "Delete model outputs…", tone: "danger", disabled: !specs.length, onSelect: () => { void deleteOutputs([c.ref]).then(() => card.reload()); } },
  ];
  const showModel = (spec: string) => {
    setPicked(spec);
    // LR against that model (two large frames); JWST stays one chip away
    api?.setTiers(["lr", `m:${spec}`]);
  };
  const focusModel = focus ? c.models?.[focus] : null;
  const disk = c.disk ?? {};
  const extras = c.extras ?? {};
  const q1 = c.q1_tile;
  return (
    <div className="res-card">
      <div className="res-card__viewer">
        <ImageViewer key={`${c.ref}:${specs.join(",")}`} collection="real" params={params} initialId={c.id}
          tiers={viewerTiers} id={`realtile-${c.ref}`} toolbar="full" nav={false}
          onReady={(a) => { setApi(a); follow.onReady(a); }} onState={follow.onState} />
      </div>
      <div className="res-card__status">
        {/* the frames fill the stage (focus mode: F, Esc returns) */}
        <Button size="sm" variant="ghost" icon={<VIcon name="focus" />} className="res-card__large" disabled={!api}
          title="Show the images over the whole page (F; Esc returns)" onClick={() => api?.setFocus(true)}>Open large</Button>
        <StateBadge state={c.production_state} prefix="production" />
        <Badge size="sm">{c.source}</Badge>
        {c.field && <Badge size="sm">{c.field}</Badge>}
        {c.has_jwst && <Badge size="sm" tone="info">JWST</Badge>}
        {!c.model_ready && <Badge size="sm" tone="warn" title="The tile has no four-band LR input yet">LR incomplete</Badge>}
      </div>
      <div className="res-card__actions">
        {hasPos && <Button size="sm" icon="globe" onClick={() => fly({ ra, dec, fov })}>Show on sky</Button>}
        <Button size="sm" variant="primary" onClick={() => compareModels([c.ref])}>Compare models…</Button>
        {c.source === "pair" && !c.model_ready ? (
          <Button size="sm" onClick={() => { void buildPairInput(c.id, c.label); }}>Build LR + run production</Button>
        ) : (
          <RunModelsPopover refs={[c.ref]} defaultSpecs={defaultRunSpecs(c.models ?? {})} onDone={() => card.reload()}
            trigger={<Button size="sm" icon="activity" disabled={!c.model_ready}>Run models…</Button>} />
        )}
        <Menu label="Download image.fits" items={downloads}
          trigger={<Button size="sm" icon="download" iconRight="chevronDown">FITS</Button>} />
        <MoreMenu items={more} label="More tile actions" />
      </div>
      <DefList dense items={[
        ["position", <Position ra={c.ra} dec={c.dec} />],
        c.shape ? ["grid", <span className="mono">{c.shape[1]} × {c.shape[0]} px, {c.pixscale ?? "—"}″/px</span>] : null,
        q1 ? ["Q1 tile", <span className="mono">{q1.tile}{q1.levels_e?.length ? `, VIS sky ${formatNumber(q1.levels_e[0], { digits: 1 })} e⁻` : ""}{q1.rejected ? " (rejected)" : ""}</span>]
          : ["Q1 tile", "outside the committed Q1 polygons"],
        typeof extras.grade === "string" && extras.grade ? ["grade", extras.grade] : null,
        typeof extras.flux_ratio_sr_over_lr === "number" ? ["flux SR/LR (eval)", formatNumber(extras.flux_ratio_sr_over_lr, { digits: 3 })] : null,
        typeof extras.position_name === "string" ? ["pointing", extras.position_name] : null,
        disk.total_bytes ? ["disk", <span className="mono">{formatBytes(disk.total_bytes)}{disk.output_bytes ? `, outputs ${formatBytes(disk.output_bytes)}` : ""}{disk.cache_bytes ? `, member cache ${formatBytes(disk.cache_bytes)}` : ""}</span>] : null,
      ]} />
      {hasPos && (
        <Section title="Overlay on the sky" sub="its pixels as an atlas layer" collapsible defaultOpen={location.pathname === "/sky/atlas"}>
          <OverlayControl card={c} />
        </Section>
      )}
      <Section title="Models" sub={specs.length ? `${specs.length} computed` : "none yet"} collapsible defaultOpen>
        {specs.length ? (
          <>
            <DataTable rows={rows} columns={MODEL_COLUMNS} rowKey={(m) => m.spec} aria-label="Model outputs"
              dense height={specs.length > 8 ? 260 : "auto"} hideToolbar={specs.length <= 8}
              activeKey={focus} onRowClick={(m) => showModel(m.spec)} />
            <MetricNote />
          </>
        ) : <p className="muted res-note">No SR for this tile yet: run models, or compare models in Experiments.</p>}
      </Section>
      {focus && (
        <Section title={`Metrics of ${specShort(focus)}`} sub={focusModel?.legacy ? "legacy SR" : focusModel?.experiment_id ?? undefined}
          collapsible defaultOpen>
          {focusModel?.metrics?.per_band && Object.keys(focusModel.metrics.per_band).length ? (
            <>
              <BandMetricsTable perBand={focusModel.metrics.per_band} caption={`Per-band metrics of ${focus}`} />
              <CoreWeights weights={focusModel.metrics.gate_core_weights} />
            </>
          ) : (
            <p className="muted res-note">Not scored yet: run {specShort(focus)} in an experiment to compute holes and R.</p>
          )}
        </Section>
      )}
      {!!c.experiments?.length && (
        <Section title="Experiments" sub={String(c.experiments.length)} collapsible defaultOpen={false}>
          <div className="res-chips">
            {c.experiments.map((e) => (
              <Button key={e} size="sm" variant="ghost" onClick={() => openInspector({ kind: "experiment", id: e })}>
                <span className="mono">{e}</span>
              </Button>
            ))}
          </div>
        </Section>
      )}
      {c.source === "eval" && (
        <Section title="Catalogue evaluation" collapsible defaultOpen={false}>
          <EvalProvenance id={c.id} />
        </Section>
      )}
    </div>
  );
}
