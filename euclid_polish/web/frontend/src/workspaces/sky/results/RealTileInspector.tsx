/* The ONE real-tile card. Inspector kinds `tile:<source>/<id>` (the atlas; the
 * palette's `tile:nexus/<n>`) and `realtile:<source>/<id>` (old links; Sky ›
 * Targets and Compare open `tile:`) both render it, from GET
 * /api/real/<source>/<id> (C9) — the same design wherever it was opened.
 *
 * Top to bottom (console regrouping): the viewer (`real` collection with ONLY
 * this tile's own tiers — a catalogue object `eval/<id>`: the `evaluation`
 * collection's LR and SR; two large frames, LR and the model it shows) with a
 * footer giving that model's VIS Δm against the LR (warn-toned beyond
 * 0.1 mag); ONE status sentence in the Targets vocabulary (current / stale /
 * missing, "made by <model>" in words; a catalogue object reads its
 * evaluation record); the two headline numbers of the model shown (worst-band
 * holes, median R); the actions (Compare models on this tile, Open in Files,
 * FITS, show on sky, run models, more); the models table (a row shows that
 * model in the viewer), its per-band metrics, the overlay on the sky and the
 * comparisons it ran in; collapsed Details (position digits, grid, Q1 tile
 * and its sky noise, disk); and, alone at the foot, the danger zone with the
 * typed-confirmed "Delete model outputs". Opening the card runs nothing. */
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { formatBytes, formatDateTime, formatNumber } from "../../../format";
import {
  Badge, Button, Callout, DataTable, DefList, Details, FactsList, Menu, Section, Skeleton, copyText, toast,
  type DataColumn, type MenuItem,
} from "../../../ui";
import { ImageViewer, type ViewerApi } from "../../../viewer";
import { VIcon } from "../../../viewer/icons";
import { buildPairInput, cacheTileAt, downloadPair, runNexusProduction } from "../atlas/actions";
import { useCompareModels, useFlyTo } from "../atlas/engineHooks";
import { rememberTile } from "../atlas/home";
import { MoreMenu, positionMenuItems } from "../atlas/inspectors/common";
import { OverlayControl, tileFovDeg } from "../atlas/inspectors/overlay";
import { cardStatus } from "../targets/model";
import { deleteOutputs } from "./actions";
import { BANDS, splitRef, URLS, type CardModel, type EvalObjectCard, type TileCard } from "./api";
import { BandMetricsTable, CoreWeights, Position, StateBadge, downloadUrl } from "./common";
import { useFollowViewer } from "./follow";
import { useSnugStage } from "./snugStage";
import {
  bandLabel, cardDelta, cardFiles, cardHeadline, cardViewerTiers, defaultRunSpecs, evalHeadline, formatMetric, outputOrigin,
  num, realTileViewerParams, sortSpecs, specShort, specWords,
} from "./model";
import { RunModelsPopover } from "./RunModels";
import "./results.css";

type ModelRow = CardModel & { spec: string };

/** A catalogue object's frames: its LR beside the catalogue evaluation's SR. */
const EVAL_TIERS = ["LR", "SR"];

const MODEL_COLUMNS: DataColumn<ModelRow>[] = [
  { id: "spec", header: "Model", width: 110, accessor: (m) => specWords(m.spec),
    cell: (m) => <span className="res-ellipsis" title={m.label ?? m.spec}>{specWords(m.spec)}</span>, filterText: (m) => `${m.spec} ${m.label ?? ""}` },
  { id: "state", header: "State", width: 84, cell: (m) => <StateBadge state={m.state} title={[m.label, outputOrigin(m), m.created ? formatDateTime(m.created) : null].filter(Boolean).join(" · ") || undefined} /> },
  { id: "holes", header: "Holes %", width: 76, numeric: true, accessor: (m) => m.metrics?.summary?.hole_pct_max ?? null,
    cell: (m) => formatMetric("hole_pct", m.metrics?.summary?.hole_pct_max) },
  { id: "R", header: "R̃", headerText: "Median R", width: 56, numeric: true, accessor: (m) => m.metrics?.summary?.median_R ?? null,
    cell: (m) => formatMetric("median_R", m.metrics?.summary?.median_R) },
  { id: "legacy", header: "Origin", accessor: (m) => outputOrigin(m) || "store", hidden: true },
  { id: "created", header: "Created", accessor: (m) => m.created ?? "", cell: (m) => (m.created ? formatDateTime(m.created) : "—"), hidden: true },
];

/** The catalogue evaluation's own record of an `eval/` object (the SR files
 *  and provenance); its state is the card's status sentence. */
function EvalProvenance({ card }: { card: EvalObjectCard }) {
  const prov = card.provenance?.[0];
  return (
    <>
      <DefList dense items={[
        ["now", card.current ? `${card.current.n_members} starfull members · ${card.current.combiner_kind ? card.current.combiner_kind.replace(/_/g, " ") : "member mean"}` : "—"],
        prov ? ["provenance", <span className="mono">{prov.id} · {prov.git ?? "?"}{prov.dirty ? " (dirty)" : ""} · {prov.created_at ? formatDateTime(prov.created_at) : ""}</span>] : null,
        card.disagreement?.pca_n ? ["disagreement", `${card.disagreement.pca_n} principal components`] : null,
      ]} />
      {!!Object.keys(card.downloads ?? {}).length && (
        <div className="res-chips">
          {Object.entries(card.downloads ?? {}).map(([tier, url]) => (
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
  const evalCard = useResource<EvalObjectCard>(source === "eval" && tileId ? URLS.evalObject(tileId) : null, [], { ttl: 30_000 });
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
  // A catalogue object's SR lives in the catalogue evaluation (not the model
  // store): its card shows the `evaluation` collection's LR beside that SR.
  const evalViewer = source === "eval" && evalCard.data?.viewer?.collection ? evalCard.data.viewer : null;
  const viewerTiers = useMemo(() => (evalViewer ? EVAL_TIERS : cardViewerTiers(focus ? [focus] : specs, !!c?.has_jwst)),
    [evalViewer, focus, specs, c?.has_jwst]);
  // The viewer walks every tile of the source: keep it on this one, with these tiers.
  const follow = useFollowViewer(evalViewer?.id ?? tileId, viewerTiers);
  const stageRef = useRef<HTMLDivElement>(null);
  useSnugStage(stageRef, `${c?.ref ?? ""}:${evalViewer ? "evaluation" : specs.join(",")}`);
  const rows = useMemo<ModelRow[]>(() => specs.map((spec) => ({ spec, ...(c?.models?.[spec] ?? {}) })), [specs, c?.models]);
  // A metric column shows only when some output has that number (R needs a
  // bright peak: a tile with none has no R̃ anywhere).
  const hasHoles = rows.some((m) => num(m.metrics?.summary?.hole_pct_max) != null);
  const hasR = rows.some((m) => num(m.metrics?.summary?.median_R) != null);
  const modelColumns = useMemo(
    () => MODEL_COLUMNS.filter((col) => (col.id !== "holes" || hasHoles) && (col.id !== "R" || hasR)),
    [hasHoles, hasR],
  );
  // The atlas opens on the last real tile inspected (atlas/home.ts).
  useEffect(() => {
    if (c?.ref && c.ra != null && c.dec != null) rememberTile(c.ref, c.ra, c.dec, tileFovDeg(c) / 2.5);
  }, [c]);

  if (!tileId) return <Callout tone="warn" title="Not a real tile">Real tiles look like <code>nexus/f200w-0012</code>, <code>tile/&lt;id&gt;</code> or <code>eval/&lt;id&gt;</code>.</Callout>;
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
  // A catalogue object's state is its evaluation record's (the store's own
  // state stands in only when that record cannot be read).
  const status = source === "eval" && (evalCard.data || evalCard.loading)
    ? cardStatus(c, evalCard.data ?? null) : cardStatus({ ...c, source: source === "eval" ? "tile" : c.source });
  const headline = cardHeadline(c, focus);
  // A catalogue object has no holes or R: its LR and SR fluxes are its numbers.
  const evalFacts = source === "eval" && !headline ? evalHeadline(evalCard.data) : null;
  const delta = cardDelta(c, focus, status?.madeBy);
  const files = cardFiles(c, specs);
  const imageTiers = Object.keys(c.image_urls ?? { lr: "" });
  const downloads: MenuItem[] = imageTiers.map((tier) => ({
    type: "sub", label: tier === "lr" ? "LR" : tier === "jwst" ? "JWST" : specWords(tier.slice(2)),
    items: (tier === "jwst" ? [""] : [...BANDS]).map((band) => ({
      label: band ? bandLabel(band) : "First filter",
      onSelect: () => downloadUrl(URLS.image(c.ref, tier, band || "VIS")),
    })),
  }));
  const onTargets = location.pathname === "/sky/targets";
  const more: MenuItem[] = [
    ...(hasPos ? [
      { label: "Cache a 25.6″ tile at its centre…", onSelect: () => { void cacheTileAt(ra, dec); } },
      { label: "Download a JWST × Euclid pair here…", onSelect: () => { void downloadPair({ ra, dec }); } },
    ] : []),
    ...(c.source === "nexus" ? [{
      label: "Run production on every stale NEXUS tile…",
      onSelect: () => { void runNexusProduction(typeof c.extras?.field_id === "string" ? c.extras.field_id : undefined); },
    }] : []),
    ...(onTargets ? [] : [{ label: "Open in Sky › Targets", onSelect: () => navigate(`/sky/targets?inspect=${encodeURIComponent(`tile:${c.ref}`)}`) }]),
    { label: "Copy the tile ref", onSelect: () => { void copyText(c.ref).then((ok) => { if (ok) toast.success(`Copied ${c.ref}`); }); } },
    ...(hasPos ? [{ type: "separator" as const }, ...positionMenuItems(ra, dec, fov)] : []),
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
  const q1Noise = q1?.levels_e?.length
    ? `VIS ${formatNumber(q1.levels_e[0], { digits: 1 })} e⁻ (the MER noise map's per-pixel RMS, median over 25.6″)` : null;
  return (
    <div className="res-card">
      <div className="res-card__viewer">
        <div ref={stageRef} className="res-card__stage">
          {source === "eval" && evalCard.loading ? <Skeleton lines={6} /> : evalViewer ? (
            <ImageViewer key={`${c.ref}:evaluation`} collection={evalViewer.collection} initialId={evalViewer.id}
              tiers={viewerTiers} id={`realtile-${c.ref}`} toolbar="full" nav={false}
              onReady={(a) => { setApi(a); follow.onReady(a); }} onState={follow.onState} />
          ) : (
            <ImageViewer key={`${c.ref}:${specs.join(",")}`} collection="real" params={params} initialId={c.id}
              tiers={viewerTiers} id={`realtile-${c.ref}`} toolbar="full" nav={false}
              onReady={(a) => { setApi(a); follow.onReady(a); }} onState={follow.onState} />
          )}
        </div>
        <div className="res-card__foot">
          {delta ? (
            <span className="res-card__delta" data-warn={delta.warn || undefined}>
              VIS, LR vs {delta.label}: <span className="mono">{delta.text}</span>
            </span>
          ) : <span className="muted">{specs.length ? "No flux measurement for this model yet." : "LR only: no SR yet."}</span>}
          {/* the frames fill the stage (focus mode: F, Esc returns) */}
          <Button size="sm" variant="ghost" icon={<VIcon name="focus" />} className="res-card__large" disabled={!api}
            title="Show the images over the whole page (F; Esc returns)" onClick={() => api?.setFocus(true)}>Open large</Button>
        </div>
      </div>
      <p className="res-card__sentence">
        {status ? status.text : <span className="muted">Reading the catalogue evaluation…</span>}
        {!c.model_ready && <> <Badge size="sm" tone="warn" title="The tile has no four-band LR input yet">LR incomplete</Badge></>}
      </p>
      {headline ? (
        <>
          <FactsList title={`Measured on ${headline.label}`} facts={headline.facts} />
          {headline.note && <p className="muted res-note">{headline.note}</p>}
        </>
      ) : evalFacts ? (
        <FactsList title="Total VIS flux" facts={evalFacts} />
      ) : specs.length ? (
        <p className="muted res-note">{focus ? `${specWords(focus)} is not scored yet: compare models on this tile to measure its holes and R.` : ""}</p>
      ) : null}
      <div className="res-card__actions">
        <Button size="sm" variant="primary" onClick={() => compareModels([c.ref])}>Compare models on this tile…</Button>
        {files.length > 1 ? (
          <Menu label="Open in Files" items={files.map((f) => ({ label: f.label, onSelect: () => navigate(f.href) }))}
            trigger={<Button size="sm" icon="fileSearch" iconRight="chevronDown">Open in Files</Button>} />
        ) : files.length === 1 ? (
          <Button asChild size="sm" icon="fileSearch"><Link to={files[0].href}>Open in Files</Link></Button>
        ) : null}
        <Menu label="Download image.fits" items={downloads}
          trigger={<Button size="sm" icon="download" iconRight="chevronDown">FITS</Button>} />
        {hasPos && <Button size="sm" icon="globe" onClick={() => fly({ ra, dec, fov })}>Show on sky</Button>}
        {c.source === "pair" && !c.model_ready ? (
          <Button size="sm" onClick={() => { void buildPairInput(c.id, c.label); }}>Build LR + run production</Button>
        ) : (
          <RunModelsPopover refs={[c.ref]} defaultSpecs={defaultRunSpecs(c.models ?? {})} onDone={() => card.reload()}
            trigger={<Button size="sm" icon="activity" disabled={!c.model_ready}>Run models…</Button>} />
        )}
        <MoreMenu items={more} label="More tile actions" />
      </div>
      {specs.length > 0 && (
        <Section title="Models" sub={`${specs.length} computed`} collapsible defaultOpen={specs.length > 1}>
          <DataTable rows={rows} columns={modelColumns} rowKey={(m) => m.spec} aria-label="Model outputs"
            dense height={specs.length > 8 ? 260 : "auto"} hideToolbar={specs.length <= 8}
            activeKey={focus} onRowClick={(m) => showModel(m.spec)} />
          {(hasHoles || hasR) && (
            <p className="res-note">
              {hasHoles && hasR ? "Holes % is the worst band, R̃ the median enclosed-flux ratio"
                : hasHoles ? "Holes % is the worst band" : "R̃ is the median enclosed-flux ratio"}
              {hasHoles && !hasR ? " (no output has a median R)" : ""}: <Link to="/sky/compare?defs=1">what the metrics measure</Link>.
            </p>
          )}
        </Section>
      )}
      {focus && focusModel?.metrics?.per_band && Object.keys(focusModel.metrics.per_band).length > 0 && (
        <Section title={`Every metric of ${specWords(focus)}`} sub={focusModel.legacy ? "legacy SR" : undefined}
          collapsible defaultOpen={false}>
          <BandMetricsTable perBand={focusModel.metrics.per_band} caption={`Per-band metrics of ${focus}`} />
          <CoreWeights weights={focusModel.metrics.gate_core_weights} />
        </Section>
      )}
      {hasPos && (
        <Section title="Overlay on the sky" sub="its pixels as an atlas layer" collapsible defaultOpen={location.pathname === "/sky/atlas"}>
          <OverlayControl card={c} />
        </Section>
      )}
      {!!c.experiments?.length && (
        <Section title="Comparisons" sub={String(c.experiments.length)} collapsible defaultOpen={false}>
          <div className="res-chips">
            {c.experiments.map((e) => (
              <Button key={e} size="sm" variant="ghost" onClick={() => openInspector({ kind: "experiment", id: e })}>
                <span className="mono">{e}</span>
              </Button>
            ))}
          </div>
        </Section>
      )}
      {c.source === "eval" && evalCard.data && (
        <Section title="Catalogue evaluation" collapsible defaultOpen={false}>
          <EvalProvenance card={evalCard.data} />
        </Section>
      )}
      <Details summary="Details" className="res-card__details">
        <DefList dense items={[
          ["position", <Position ra={c.ra} dec={c.dec} />],
          ["source", c.source],
          c.field ? ["field", c.field] : null,
          c.shape ? ["grid", <span className="mono">{c.shape[1]} × {c.shape[0]} px, {c.pixscale ?? "—"}″/px</span>] : null,
          q1 ? ["Q1 tile", <span className="mono">{q1.tile}{q1.rejected ? " (measured unobserved)" : ""}</span>]
            : ["Q1 tile", "outside the committed Q1 polygons"],
          q1Noise ? ["Q1 sky noise", q1Noise] : null,
          typeof extras.grade === "string" && extras.grade ? ["grade", extras.grade] : null,
          typeof extras.position_name === "string" ? ["pointing", extras.position_name] : null,
          disk.total_bytes ? ["disk", <span className="mono">{formatBytes(disk.total_bytes)}{disk.output_bytes ? `, outputs ${formatBytes(disk.output_bytes)}` : ""}{disk.cache_bytes ? `, member cache ${formatBytes(disk.cache_bytes)}` : ""}</span>] : null,
          focusModel?.experiment_id ? [`${specShort(focus ?? "")} from`, <code className="mono">{focusModel.experiment_id}</code>] : null,
        ]} />
      </Details>
      {(specs.some((s) => !c.models?.[s]?.legacy) || !!disk.cache_bytes) && (
        <section className="res-danger" aria-label="Danger zone">
          <p className="res-note">Deletes every cached SR of this tile, its metrics and the member-SR cache; the LR and legacy SR files stay.</p>
          <Button size="sm" variant="danger" onClick={() => { void deleteOutputs([c.ref]).then(() => card.reload()); }}>
            Delete model outputs…
          </Button>
        </section>
      )}
    </div>
  );
}
