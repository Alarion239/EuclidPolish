/* Inspector kind `realtile:<source>/<id>` — one real tile of the C9 store
 * (GET /api/real/<source>/<id>): metadata, an image viewer over LR + every
 * computed model (+ JWST; residuals from the viewer toolbar), the models and
 * their state vs the production model, per-band real-data metrics, disk,
 * experiments; actions: run models, compare, open on the sky, download
 * image.fits, delete outputs. Eval objects add their catalogue-eval
 * provenance (members.json, SR provenance sidecars). */
import { useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { formatBytes, formatDateTime, formatNumber } from "../../../format";
import {
  Badge, Button, Callout, DataTable, DefList, IconButton, Menu, Section, Skeleton, copyText, toast,
  type DataColumn, type MenuItem,
} from "../../../ui";
import { ImageViewer, type ViewerApi } from "../../../viewer";
import { deleteOutputs } from "./actions";
import { atlasHref, BANDS, experimentsHref, splitRef, URLS, type CardModel, type EvalObjectCard, type TileCard } from "./api";
import { BandMetricsTable, CoreWeights, Position, StateBadge, downloadUrl } from "./common";
import { bandLabel, formatMetric, sortSpecs, specShort } from "./model";
import { RunModelsPopover } from "./RunModels";
import "./results.css";

type ModelRow = CardModel & { spec: string };

const MODEL_COLUMNS: DataColumn<ModelRow>[] = [
  { id: "spec", header: "Model", cell: (m) => <code className="mono">{specShort(m.spec)}</code>, filterText: (m) => `${m.spec} ${m.label ?? ""}` },
  { id: "state", header: "State", cell: (m) => <StateBadge state={m.state} title={m.legacy ? `legacy ${m.origin ?? "record"}` : undefined} /> },
  { id: "holes", header: "Holes %", numeric: true, accessor: (m) => m.metrics?.summary?.hole_pct_max ?? null,
    cell: (m) => formatMetric("hole_pct", m.metrics?.summary?.hole_pct_max) },
  { id: "R", header: "R̃", numeric: true, accessor: (m) => m.metrics?.summary?.median_R ?? null,
    cell: (m) => formatMetric("median_R", m.metrics?.summary?.median_R) },
  { id: "legacy", header: "Origin", accessor: (m) => (m.legacy ? `legacy ${m.origin ?? ""}` : "store"), hidden: true },
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
        ["now", c.current ? `${c.current.n_members} STARFULL · ${c.current.combiner_kind ?? "member mean"}` : "—"],
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
  const api = useRef<ViewerApi | null>(null);
  const c = card.data;
  const specs = useMemo(() => sortSpecs(Object.keys(c?.models ?? {})), [c?.models]);
  const [picked, setPicked] = useState<string | null>(null);
  const focus = picked && c?.models?.[picked] ? picked : (specs.find((s) => c?.models?.[s]?.metrics?.per_band) ?? specs[0] ?? null);
  const params = useMemo(() => ({ source, ...(specs.length ? { models: specs.join(",") } : {}) }), [source, specs]);
  const rows = useMemo<ModelRow[]>(() => specs.map((spec) => ({ spec, ...(c?.models?.[spec] ?? {}) })), [specs, c?.models]);

  if (!tileId) return <Callout tone="warn" title="Not a real tile">Real tiles look like <code>nexus/f200w-0012</code> or <code>eval/&lt;id&gt;</code>.</Callout>;
  if (card.loading) return <Skeleton lines={8} />;
  if (!c) {
    return (
      <Callout tone="bad" title={card.error?.status === 404 ? "Unknown real tile" : "Could not load the tile"}
        action={<Button size="sm" onClick={card.reload}>Retry</Button>}>
        {card.error?.message ?? "No data."}
      </Callout>
    );
  }
  const viewerTiers = ["lr", ...specs.slice(0, 2).map((s) => `m:${s}`), ...(c.has_jwst ? ["jwst"] : [])];
  const imageTiers = Object.keys(c.image_urls ?? { lr: "" });
  const downloads: MenuItem[] = imageTiers.map((tier) => ({
    type: "sub", label: tier === "lr" ? "LR" : tier === "jwst" ? "JWST" : specShort(tier.slice(2)),
    items: (tier === "jwst" ? [""] : [...BANDS]).map((band) => ({
      label: band ? bandLabel(band) : "first filter",
      onSelect: () => downloadUrl(URLS.image(c.ref, tier, band || "VIS")),
    })),
  }));
  const more: MenuItem[] = [
    { label: "Open the atlas card", onSelect: () => openInspector({ kind: "tile", id: c.ref }) },
    { label: "Copy ref", onSelect: () => { void copyText(c.ref).then(() => toast.success(`Copied ${c.ref}`)); } },
    { type: "separator" },
    { label: "Delete model outputs…", tone: "danger", disabled: !specs.length, onSelect: () => { void deleteOutputs([c.ref]).then(() => card.reload()); } },
  ];
  const showTier = (spec: string) => {
    setPicked(spec);
    api.current?.setTiers(["lr", `m:${spec}`, ...(c.has_jwst ? ["jwst"] : [])]);
  };
  const focusModel = focus ? c.models?.[focus] : null;
  const disk = c.disk ?? {};
  const extras = c.extras ?? {};
  return (
    <div className="res-card">
      <div className="res-card__badges">
        <StateBadge state={c.production_state} prefix="production" />
        <Badge size="sm">{c.source}</Badge>
        {c.field && <Badge size="sm">{c.field}</Badge>}
        {c.has_jwst && <Badge size="sm" tone="info">JWST</Badge>}
        {!c.model_ready && <Badge size="sm" tone="warn" title="The tile has no four-band LR input yet">LR incomplete</Badge>}
      </div>
      <div className="res-card__actions">
        <RunModelsPopover refs={[c.ref]} onDone={() => card.reload()} />
        <Button size="sm" onClick={() => navigate(experimentsHref([c.ref]))}>Compare…</Button>
        {c.ra != null && c.dec != null && (
          <Button size="sm" icon="globe" onClick={() => navigate(atlasHref(c.ra!, c.dec!, c.ref))}>On sky</Button>
        )}
        <Menu label="Download image.fits" items={downloads}
          trigger={<Button size="sm" icon="download" iconRight="chevronDown">FITS</Button>} />
        <Menu label="More tile actions" items={more}
          trigger={<IconButton icon="more" label="More tile actions" size="sm" />} />
      </div>
      <div className="res-card__viewer">
        <ImageViewer key={`${c.ref}:${specs.join(",")}`} collection="real" params={params} initialId={c.id}
          tiers={viewerTiers} id={`realtile-${c.ref}`} toolbar="full" nav={false}
          onReady={(a) => { api.current = a; }} />
      </div>
      <DefList dense items={[
        ["position", <Position ra={c.ra} dec={c.dec} />],
        c.shape ? ["grid", <span className="mono">{c.shape[1]} × {c.shape[0]} px · {c.pixscale ?? "—"}″/px</span>] : null,
        c.q1_tile ? ["Q1 tile", <span className="mono">{c.q1_tile.tile}{c.q1_tile.levels_e?.length ? ` · VIS sky ${formatNumber(c.q1_tile.levels_e[0], { digits: 1 })} e⁻` : ""}{c.q1_tile.rejected ? " · rejected" : ""}</span>] : null,
        typeof extras.grade === "string" && extras.grade ? ["grade", extras.grade] : null,
        typeof extras.flux_ratio_sr_over_lr === "number" ? ["flux SR/LR (eval)", formatNumber(extras.flux_ratio_sr_over_lr, { digits: 3 })] : null,
        typeof extras.position_name === "string" ? ["pointing", extras.position_name] : null,
        disk.total_bytes ? ["disk", <span className="mono">{formatBytes(disk.total_bytes)}{disk.output_bytes ? ` · outputs ${formatBytes(disk.output_bytes)}` : ""}{disk.cache_bytes ? ` · cache ${formatBytes(disk.cache_bytes)}` : ""}</span>] : null,
      ]} />
      <Section title="Models" sub={specs.length ? `${specs.length} computed` : "none yet"} collapsible defaultOpen>
        {specs.length ? (
          <DataTable rows={rows} columns={MODEL_COLUMNS} rowKey={(m) => m.spec} aria-label="Model outputs"
            dense height={specs.length > 8 ? 260 : "auto"} hideToolbar={specs.length <= 8}
            activeKey={focus} onRowClick={(m) => showTier(m.spec)} />
        ) : <p className="muted res-note">No SR yet — run production or compare models.</p>}
      </Section>
      {focus && (
        <Section title={`Metrics · ${specShort(focus)}`} sub={focusModel?.legacy ? "legacy SR" : focusModel?.experiment_id ?? undefined}
          collapsible defaultOpen>
          {focusModel?.metrics?.per_band && Object.keys(focusModel.metrics.per_band).length ? (
            <>
              <BandMetricsTable perBand={focusModel.metrics.per_band} caption={`Per-band metrics of ${focus}`} />
              <CoreWeights weights={focusModel.metrics.gate_core_weights} />
            </>
          ) : (
            <p className="muted res-note">Not scored yet — run {specShort(focus)} in an experiment to compute holes and R.</p>
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
