/* Sky › Real results (spec §7.2): one DataTable over every real tile source
 * of the C9 store (nexus, cached 25.6″ tiles, legacy fields with all their
 * sub-tiles, archive, eval, poster, pairs) — position, field, tiers, models
 * computed and the production state, real-data metrics. Row → the
 * real-tile card (`tile:`, the atlas's kind); bulk run models / compare / delete outputs; cache a
 * new 25.6″ tile; the legacy real field's model–model diagnostics beside
 * their synthetic counterparts (`diag=1`). Filters and sort live in the URL. The selection is the
 * shared `tile` selection (the atlas and Experiments see it). */
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { formatCount } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useSelected, useSelection } from "../../../state/selection";
import {
  Button, Callout, Chip, DataTable, IconButton, Menu, Page, Popover, Segmented, Tooltip, type MenuItem,
} from "../../../ui";
import { computeMetrics, deleteOutputs, refreshResults, runNexusField } from "../results/actions";
import { experimentsHref, SOURCES, URLS, type SourcesPayload, type TileList } from "../results/api";
import { CacheTilePopover } from "../results/CacheTile";
import { REAL_TILE_COLUMNS } from "../results/columns";
import { FieldDiagnostics } from "../results/FieldDiagnostics";
import { MetricDefinitions, StateBadge } from "../results/common";
import { filterByState, flattenTiles, metricsPlan, productionCounts } from "../results/model";
import "../results/register";
import { RunModelsPopover } from "../results/RunModels";
import "../results/results.css";


const TTL = { ttl: 60_000 };

const STATES = ["all", "current", "stale", "missing"] as const;

export default function Results() {
  const navigate = useNavigate();
  const [src, setSrc] = useUrlState<string[]>("src", []);
  const [state, setState] = useUrlState("state", "all");
  const [diag, setDiag] = useUrlState("diag", false);
  const [runOpen, setRunOpen] = useState(false);
  const [cacheOpen, setCacheOpen] = useState(false);
  const sources = useResource<SourcesPayload>(URLS.sources, [], TTL);
  const on = (s: string) => !src.length || src.includes(s);
  const nexus = useResource<TileList>(on("nexus") ? URLS.list("nexus") : null, [], TTL);
  const tile = useResource<TileList>(on("tile") ? URLS.list("tile") : null, [], TTL);
  const field = useResource<TileList>(on("field") ? URLS.list("field") : null, [], TTL);
  const archive = useResource<TileList>(on("archive") ? URLS.list("archive") : null, [], TTL);
  const evalObjs = useResource<TileList>(on("eval") ? URLS.list("eval") : null, [], TTL);
  const poster = useResource<TileList>(on("poster") ? URLS.list("poster") : null, [], TTL);
  const pair = useResource<TileList>(on("pair") ? URLS.list("pair") : null, [], TTL);
  const lists = [nexus, tile, field, archive, evalObjs, poster, pair];
  const all = useMemo(
    () => flattenTiles([nexus.data, tile.data, field.data, archive.data, evalObjs.data, poster.data, pair.data]),
    [nexus.data, tile.data, field.data, archive.data, evalObjs.data, poster.data, pair.data],
  );
  const rows = useMemo(() => filterByState(all, state), [all, state]);
  const counts = useMemo(() => productionCounts(all), [all]);
  const loading = lists.some((l) => l.loading) && !all.length;
  const failed = SOURCES.map((s, i) => [s, lists[i].error] as const).filter(([, e]) => e);

  const selectedAll = useSelected("tile");
  const known = useMemo(() => new Set(all.map((r) => r.ref)), [all]);
  const selected = useMemo(() => selectedAll.filter((k) => known.has(k)), [selectedAll, known]);
  const onSelected = (keys: string[]) => {
    const visible = new Set(rows.map((r) => r.ref));
    useSelection.getState().select("tile", [...selectedAll.filter((k) => !visible.has(k)), ...keys]);
  };
  const nexusSel = selected.filter((r) => r.startsWith("nexus/"));
  const nexusField = (nexus.data?.tiles.find((t) => nexusSel.includes(t.ref))?.extras?.field_id as string | undefined)
    ?? (nexus.data?.tiles[0]?.extras?.field_id as string | undefined);

  const plan = useMemo(() => {
    const sel = new Set(selected);
    return metricsPlan(all.filter((r) => sel.has(r.ref)));
  }, [all, selected]);
  const unscored = plan.reduce((n, g) => n + g.specs.length * g.refs.length, 0);
  const reload = () => { refreshResults(); for (const l of lists) void l.reload(); void sources.reload(); };
  const compare = () => navigate(experimentsHref(selected));
  usePageActions([
    { id: "results-run", label: "Run models on the selected real tiles…", group: "Real results", disabled: !selected.length, run: () => setRunOpen(true) },
    { id: "results-metrics", label: "Compute the missing metrics of the selected tiles…", group: "Real results", disabled: !unscored, run: () => { void computeMetrics(plan); } },
    { id: "results-compare", label: "Compare the selected tiles in Experiments", group: "Real results", disabled: !selected.length, run: compare },
    { id: "results-delete", label: "Delete the model outputs of the selected tiles…", group: "Real results", disabled: !selected.length, run: () => { void deleteOutputs(selected); } },
    { id: "results-cache", label: "Cache a 25.6″ real tile…", group: "Real results", keywords: ["download", "tile", "ra", "dec"], run: () => setCacheOpen(true) },
    { id: "results-stale", label: "Show tiles whose production SR is stale", group: "Real results", run: () => setState("stale") },
    { id: "results-refresh", label: "Refresh real tiles", group: "Real results", run: reload },
    { id: "results-diagnostics", label: diag ? "Hide the real-field diagnostics" : "Show the real-field diagnostics (r(d), σ, RBF occupancy)",
      group: "Real results", keywords: ["cross-correlation", "brightness", "occupancy", "inference"], run: () => setDiag(!diag) },
  ]);

  const toggleSource = (s: string) => {
    const cur = src.length ? src : [];
    setSrc(cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s]);
  };
  const byId = new Map((sources.data?.sources ?? []).map((s) => [s.id, s]));
  const more: MenuItem[] = [
    { label: `Run NEXUS field inference (production) on ${nexusSel.length || "every stale"} tile${nexusSel.length === 1 ? "" : "s"}…`,
      disabled: !nexusField, onSelect: () => { if (nexusField) void runNexusField(nexusField, nexusSel.map((r) => r.slice("nexus/".length))); } },
    { label: unscored ? `Compute metrics of ${unscored} unscored output${unscored === 1 ? "" : "s"}…` : "Compute metrics (every selected output is scored)",
      disabled: !unscored, onSelect: () => { void computeMetrics(plan); } },
    { type: "separator" },
    { label: "Clear the selection", disabled: !selectedAll.length, onSelect: () => useSelection.getState().clear("tile") },
    { label: "Delete model outputs…", tone: "danger", disabled: !selected.length, onSelect: () => { void deleteOutputs(selected); } },
  ];

  return (
    <Page className="res-page">
      <div className="res-bar" role="toolbar" aria-label="Real results">
        <div className="res-bar__group" role="group" aria-label="Sources">
          <Chip on={!src.length} onClick={() => setSrc([])}>All</Chip>
          {SOURCES.map((s) => {
            const info = byId.get(s);
            return (
              <Tooltip key={s} content={info ? `${info.label} — ${info.description ?? ""}` : s}>
                <Chip on={src.includes(s)} onClick={() => toggleSource(s)} disabled={info != null && info.count === 0 && !src.includes(s)}>
                  {s} <span className="muted">{info ? formatCount(info.count) : ""}</span>
                </Chip>
              </Tooltip>
            );
          })}
        </div>
        <Segmented size="sm" value={state} onChange={setState} aria-label="Production state"
          options={STATES.map((s) => ({ value: s, label: s === "all" ? "All" : `${s} ${counts[s]}` }))} />
        <span className="res-bar__spacer" />
        <span className="res-bar__sel" aria-live="polite">{selected.length ? `${selected.length} selected` : ""}</span>
        <RunModelsPopover refs={selected} open={runOpen} onOpenChange={setRunOpen} />
        <Button size="sm" disabled={!selected.length} onClick={compare}>Compare</Button>
        <Menu label="More actions" items={more}
          trigger={<IconButton icon="more" label="More actions" size="sm" />} />
        <CacheTilePopover open={cacheOpen} onOpenChange={setCacheOpen} />
        <Tooltip content="Real-field diagnostics: model–model r(d), σ vs brightness, RBF occupancy — vs synthetic">
          <Chip on={diag} onClick={() => setDiag(!diag)}>Diagnostics</Chip>
        </Tooltip>
        <Popover label="What the metrics measure" width={420} align="end"
          trigger={<IconButton icon="help" label="What the metrics measure" size="sm" />}>
          <div className="res-form">
            <strong>What the metrics measure</strong>
            <p className="res-note">In the table, Holes % is the worst band and R̃ the median over all bands' peaks, of the tile's headline model (production, else the first scored one).</p>
            <MetricDefinitions />
          </div>
        </Popover>
        <IconButton icon="reset" label="Refresh" size="sm" onClick={reload} />
      </div>
      {!!failed.length && (
        <Callout tone="bad" title="Some sources did not load" action={<Button size="sm" onClick={reload}>Retry</Button>}>
          {failed.map(([s, e]) => <div key={s}><strong>{s}</strong>: {e?.message}</div>)}
        </Callout>
      )}
      <div className="res-summary" aria-label="Summary">
        <span><strong>{formatCount(counts.total)}</strong> tiles</span>
        <span>production <StateBadge state="current" prefix={String(counts.current)} /> <StateBadge state="stale" prefix={String(counts.stale)} /> <StateBadge state="missing" prefix={String(counts.missing)} /></span>
      </div>
      {diag && <FieldDiagnostics onClose={() => setDiag(false)} />}
      <DataTable rows={rows} columns={REAL_TILE_COLUMNS} rowKey={(r) => r.ref} aria-label="Real tiles"
        selectable selected={selected} onSelectedChange={onSelected}
        inspect={(r) => ({ kind: "tile", id: r.ref })}
        exportName="real-tiles" urlKey="rt" loading={loading} height="max(420px, calc(100vh - 260px))"
        filterPlaceholder="Filter: field:EDF-N  models:gate  holes>5  production=stale"
        empty={src.length || state !== "all" ? "No tile matches these filters." : "No real tiles yet — cache one from the atlas or with “Cache tile…”."} />
    </Page>
  );
}
