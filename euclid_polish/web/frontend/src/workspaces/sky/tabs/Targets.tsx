/* Sky › Targets (console regrouping): what production SR does on each real
 * science target, and is it current? Organised by target, not by store:
 * NEXUS × JWST, the poster galaxy, the Q1 lens candidates and Q1 galaxies
 * (the real half of the old catalogue evaluation), cached tiles, and under
 * "more" the legacy real field and JWST pairs. Both backends (the real-tile
 * store's `production_state`, the evaluation manifest's object states) read
 * as ONE vocabulary through targets/model.ts: current / stale / missing,
 * plus "made by <model>" in words.
 *
 * Top to bottom: the toolbar (set chips with their counts, the state
 * control with its counts, "Run production on stale" — confirmed — the
 * Sources menu, the metric definitions link), one sentence per set, the
 * flux SR/LR strip, then the table sorted by flux ratio. A row (or a dot)
 * opens the tile card (`?inspect=tile:<ref>`). Opening the page runs nothing. */
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { formatCount } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useSelected, useSelection } from "../../../state/selection";
import {
  Button, Callout, Caption, Chip, DataTable, EmptyState, IconButton, Menu, Num, Page, Segmented, Skeleton, SummaryLine,
  Toolbar, ToolbarGroup, ToolbarSpacer, Tooltip, toast, type MenuItem,
} from "../../../ui";
import { computeMetrics, deleteOutputs, refreshResults, runModels } from "../results/actions";
import { experimentsHref, URLS, type EvalRuns, type SourcesPayload, type TileList } from "../results/api";
import { metricsPlan } from "../results/model";
import "../results/register";
import "../results/results.css";
import { runProductionPlan } from "../targets/actions";
import { targetColumns } from "../targets/columns";
import { FluxStrip } from "../targets/FluxStrip";
import {
  DEFAULT_SETS, SET_BY_ID, TARGET_SETS, evalTargets, failedCutouts, gradeCounts, groupedN, legacyTargetsPatch, lensGrades, parseSets, productionPlan,
  sentencePieces, setCounts, setSentence, stateCounts, tileTargets, withState, type TargetRow, type TargetSetId,
} from "../targets/model";
import { SourcesMenu } from "../targets/Sources";
import "../targets/targets.css";

const TTL = { ttl: 60_000 };
const STATES = ["all", "current", "stale", "missing"] as const;
const STATE_WORD: Record<string, string> = { all: "All", current: "Current", stale: "Stale", missing: "Missing" };

/** A saved crop of the old Catalog-eval browser links `?v.cev.id=<object>`:
 *  open that object's card instead (once), and drop the key. */
function useEvalViewerLink() {
  const [cev, setCev] = useUrlState("v.cev.id", "");
  useEffect(() => {
    if (!cev) return;
    openInspector({ kind: "tile", id: `eval/${cev}` });
    setCev("");
  }, [cev, setCev]);
}

export default function Targets() {
  const navigate = useNavigate();
  useEvalViewerLink();
  const [setRaw, setSetRaw] = useUrlState("set", "");
  const [src, setSrc] = useUrlState("src", "");
  const [state, setState] = useUrlState("state", "all");
  const [gradesRaw, setGrades] = useUrlState<string[]>("g", []);
  const [st, setSt] = useUrlState("st", "");
  const grades = useMemo(() => lensGrades(gradesRaw), [gradesRaw]);
  const [showFailed, setShowFailed] = useUrlState("failed", false);
  const [scoredOnly, setScoredOnly] = useUrlState("scored", false);
  // The table's own filter (DataTable urlKey "tg"): without one, its row
  // count repeats the state control's, so the table does not print it.
  const [tableQuery] = useUrlState("tg.q", "");
  // An old Catalog-eval / Real-results link (`?g=A,B`, `?st=stale`, store ids
  // in `?set=`): rewrite it once, on arrival (an effect: the router takes
  // no navigation from a layout effect of the first render).
  const legacy = useRef(legacyTargetsPatch({ set: setRaw, src, g: gradesRaw, st, state }));
  useEffect(() => {
    const p = legacy.current;
    legacy.current = null;
    if (!p) return;
    if (p.set != null) setSetRaw(p.set);
    if (p.src != null) setSrc(p.src);
    if (p.g) setGrades(p.g);
    if (p.state) setState(p.state);
    if (p.st != null) setSt(p.st);
  }, []);   // eslint-disable-line react-hooks/exhaustive-deps
  const [running, setRunning] = useState(false);
  const chosen = useMemo(() => parseSets(setRaw, src), [setRaw, src]);
  const active: readonly TargetSetId[] = chosen.length ? chosen : DEFAULT_SETS;
  const on = (id: TargetSetId) => active.includes(id);
  const storeOn = (store: string) => active.some((s) => SET_BY_ID[s].store === store);

  const sources = useResource<SourcesPayload>(URLS.sources, [], TTL);
  const runs = useResource<EvalRuns>(URLS.evalRuns, [], { ttl: 30_000 });
  const nexus = useResource<TileList>(storeOn("nexus") ? URLS.list("nexus") : null, [], TTL);
  const poster = useResource<TileList>(storeOn("poster") ? URLS.list("poster") : null, [], TTL);
  const tile = useResource<TileList>(storeOn("tile") ? URLS.list("tile") : null, [], TTL);
  const field = useResource<TileList>(storeOn("field") ? URLS.list("field") : null, [], TTL);
  const pair = useResource<TileList>(storeOn("pair") ? URLS.list("pair") : null, [], TTL);
  const lists = { nexus, poster, tile, field, pair } as const;
  const catalogueOn = on("lenses") || on("galaxies");

  const all = useMemo(() => {
    const out: TargetRow[] = [];
    for (const s of TARGET_SETS) {
      if (!active.includes(s.id)) continue;
      if (s.store) out.push(...tileTargets(s.id, lists[s.store as keyof typeof lists]?.data?.tiles ?? []));
    }
    if (catalogueOn) out.push(...evalTargets(runs.data?.rows ?? [], { failed: showFailed }).filter((r) => active.includes(r.set)));
    return out;
  }, [active, catalogueOn, showFailed, nexus.data, poster.data, tile.data, field.data, pair.data, runs.data]);   // eslint-disable-line react-hooks/exhaustive-deps
  const failed = useMemo(() => {
    const f = failedCutouts(runs.data?.rows, grades);
    return (on("lenses") ? f.lenses : 0) + (on("galaxies") ? f.galaxies : 0);
  }, [runs.data, active, grades]);   // eslint-disable-line react-hooks/exhaustive-deps
  const graded = useMemo(
    () => (grades.length ? all.filter((r) => r.set !== "lenses" || (r.grade != null && grades.includes(r.grade))) : all),
    [all, grades],
  );
  // Holes and R̃ exist for scored real tiles only: "Show only them" narrows the view.
  const scoredN = useMemo(() => graded.filter((r) => r.scored).length, [graded]);
  const view = useMemo(() => (scoredOnly && scoredN ? graded.filter((r) => r.scored) : graded), [graded, scoredOnly, scoredN]);
  const counts = useMemo(() => stateCounts(view), [view]);
  const rows = useMemo(() => withState(view, state), [view, state]);
  const chipCounts = useMemo(() => setCounts(sources.data, runs.data?.rows), [sources.data, runs.data]);
  const grades0 = useMemo(() => gradeCounts(runs.data?.rows ?? []), [runs.data]);
  const plan = useMemo(() => productionPlan(graded, grades0), [graded, grades0]);
  const gate = /gate/.test(String(runs.data?.current?.combiner_kind ?? "")) ? "gate" : "production model";
  // A set speaks only once its own list has arrived: until then its
  // sentence is a loading line, never "none yet" (another set may already
  // be in). A set whose list failed says nothing here (the callout names it).
  const resourceOf = (id: TargetSetId) => {
    const store = SET_BY_ID[id].store;
    return store ? lists[store as keyof typeof lists] : runs;
  };
  const readySets = active.filter((id) => resourceOf(id).data != null);
  const waitingSets = active.filter((id) => resourceOf(id).data == null && !resourceOf(id).error);
  const allReady = !waitingSets.length;
  const readyKey = readySets.join(",");
  const sentences = useMemo(
    () => readySets.map((s) => setSentence(s, graded.filter((r) => r.set === s))),
    [readyKey, graded],   // eslint-disable-line react-hooks/exhaustive-deps
  );

  const pending = [
    ...(Object.entries(lists) as [string, typeof nexus][]).filter(([store]) => storeOn(store)).map(([, l]) => l),
    ...(catalogueOn ? [runs] : []),
  ];
  const loading = pending.some((l) => l.loading) && !all.length;
  const loadErrors = [
    ...(Object.entries(lists) as [string, typeof nexus][]).filter(([store, l]) => storeOn(store) && l.error)
      .map(([store, l]) => [TARGET_SETS.find((s) => s.store === store)?.label ?? store, l.error?.message ?? ""] as const),
    ...(catalogueOn && runs.error ? [["Lens candidates and Q1 galaxies", runs.error.message] as const] : []),
  ];

  const selectedAll = useSelected("tile");
  const known = useMemo(() => new Set(all.filter((r) => r.ref).map((r) => r.key)), [all]);
  const selected = useMemo(() => selectedAll.filter((k) => known.has(k)), [selectedAll, known]);
  const onSelected = (keys: string[]) => {
    const visible = new Set(rows.map((r) => r.key));
    useSelection.getState().select("tile", [...selectedAll.filter((k) => !visible.has(k)), ...keys.filter((k) => known.has(k))]);
  };
  const selectedTiles = selected.filter((k) => !k.startsWith("eval/"));
  const scorePlan = useMemo(() => {
    const sel = new Set(selected);
    const byRef = new Map([nexus, poster, tile, field, pair].flatMap((l) => l.data?.tiles ?? []).map((t) => [t.ref, t]));
    return metricsPlan([...sel].map((k) => byRef.get(k)).filter((t): t is NonNullable<typeof t> => !!t));
  }, [selected, nexus.data, poster.data, tile.data, field.data, pair.data]);   // eslint-disable-line react-hooks/exhaustive-deps
  const unscored = scorePlan.reduce((n, g) => n + g.specs.length * g.refs.length, 0);

  const reload = () => { refreshResults(); for (const l of pending) void l.reload(); void sources.reload(); };
  const runStale = async () => {
    setRunning(true);
    try { await runProductionPlan(plan); } finally { setRunning(false); }
  };
  const writeSets = (ids: TargetSetId[]) => {
    setSetRaw(ids.join(","));
    if (src) setSrc("");
  };
  const toggleSet = (id: TargetSetId) => {
    const cur = chosen.length ? chosen : [];
    const next = cur.includes(id) ? cur.filter((s) => s !== id) : [...cur, id];
    writeSets(TARGET_SETS.map((s) => s.id).filter((s) => next.includes(s)));
  };
  const open = (r: { ref: string | null }) => {
    if (r.ref) openInspector({ kind: "tile", id: r.ref });
    else toast.info("This target has no reconstruction to open.");
  };

  usePageActions([
    { id: "targets-run", label: `Run production on ${plan.stale} stale targets…`, group: "Targets", disabled: !plan.stale || running || !allReady, run: () => { void runStale(); } },
    { id: "targets-stale", label: "Show stale targets", group: "Targets", run: () => setState("stale") },
    { id: "targets-compare", label: "Compare models on the selected targets", group: "Targets", disabled: !selected.length, run: () => navigate(experimentsHref(selected)) },
    { id: "targets-refresh", label: "Refresh targets", group: "Targets", run: reload },
  ]);

  const more = TARGET_SETS.filter((s) => s.more);
  const moreItems: MenuItem[] = more.map((s) => ({
    type: "checkbox", label: `${s.label}${chipCounts[s.id] != null ? ` · ${formatCount(chipCounts[s.id])}` : ""}`,
    checked: chosen.includes(s.id), onCheckedChange: () => toggleSet(s.id),
  }));
  const chip = (id: TargetSetId) => {
    const s = SET_BY_ID[id];
    const n = chipCounts[id];
    return (
      <Tooltip key={id} content={s.about}>
        <Chip on={chosen.includes(id)} onClick={() => toggleSet(id)}>
          {s.label}{n != null && <>{" "}<span className="tg-count">{formatCount(n)}</span></>}
        </Chip>
      </Tooltip>
    );
  };
  const selectionBar = selected.length ? (
    <>
      <Button size="sm" variant="primary" onClick={() => navigate(experimentsHref(selected))}>Compare models ({selected.length})</Button>
      <Tooltip content={selectedTiles.length < selected.length ? "Lens candidates and Q1 galaxies are re-made by the grouped analysis (Run production on stale)" : "One production comparison over the selected tiles"}>
        <span>
          <Button size="sm" disabled={!selectedTiles.length} onClick={() => { void runModels(selectedTiles, ["production"]); }}>
            Run production ({selectedTiles.length})…
          </Button>
        </span>
      </Tooltip>
      <Menu label="More selection actions" items={[
        { label: unscored ? `Compute the metrics of ${unscored} unscored output${unscored === 1 ? "" : "s"}…` : "Every selected output is scored",
          disabled: !unscored, onSelect: () => { void computeMetrics(scorePlan); } },
        { label: "Clear the selection", onSelect: () => useSelection.getState().clear("tile") },
        { type: "separator" },
        { label: `Delete the model outputs of ${selectedTiles.length} tile${selectedTiles.length === 1 ? "" : "s"}…`, tone: "danger",
          disabled: !selectedTiles.length, onSelect: () => { void deleteOutputs(selectedTiles).then(reload); } },
      ]} trigger={<IconButton icon="more" size="sm" label="More selection actions" />} />
    </>
  ) : null;

  const manySets = active.length > 1;
  // The Holes / R̃ columns only when every row in view has them (no blank columns).
  const allScored = rows.length > 0 && rows.every((r) => r.scored);
  const lensRows = rows.some((r) => r.set === "lenses");
  const columns = useMemo(() => targetColumns({
    scored: allScored, manySets, lenses: lensRows,
  }), [allScored, manySets, lensRows]);
  const empty = !loading && !all.length;

  return (
    <Page className="res-page tg-page">
      <Toolbar label="Targets">
        <ToolbarGroup label="Target sets" hideLabel>
          <Chip on={!chosen.length} onClick={() => writeSets([])} title="Every science target (not the sets under More)">All</Chip>
          {TARGET_SETS.filter((s) => !s.more || chosen.includes(s.id)).map((s) => chip(s.id))}
          <Menu label="More target sets" items={moreItems}
            trigger={<Button size="sm" variant="ghost" iconRight="chevronDown">More</Button>} />
        </ToolbarGroup>
        <Segmented size="sm" value={state} onChange={setState} aria-label="Production SR state"
          options={STATES.map((s) => ({ value: s, label: allReady ? <>{STATE_WORD[s]} <span className="tg-count">{formatCount(counts[s])}</span></> : STATE_WORD[s] }))} />
        <ToolbarSpacer />
        <Button size="sm" variant="primary" icon="activity" loading={running} disabled={!plan.stale || !allReady}
          onClick={() => { void runStale(); }}>
          Run production on stale{plan.stale && allReady ? ` (${formatCount(plan.stale)})` : ""}…
        </Button>
        <SourcesMenu defaultN={groupedN(grades0)} />
        <Button asChild size="sm" variant="ghost" icon="help">
          <Link to="/sky/compare?defs=1">Metric definitions</Link>
        </Button>
        <IconButton icon="reset" label="Refresh" size="sm" onClick={reload} />
      </Toolbar>

      {!!loadErrors.length && (
        <Callout tone="bad" title="Some targets did not load" action={<Button size="sm" onClick={reload}>Retry</Button>}>
          {loadErrors.map(([what, why]) => <div key={what}><strong>{what}</strong>: {why}</div>)}
        </Callout>
      )}

      {loading ? <Skeleton lines={6} /> : empty ? (
        <EmptyState icon="globe" title="No targets in these sets yet">
          {active.map((s) => SET_BY_ID[s].empty).join(" ")}
        </EmptyState>
      ) : (
        <>
          <div className="tg-sentences">
            {active.map((id) => {
              const s = sentences.find((x) => x.set === id);
              if (s) {
                return (
                  <SummaryLine key={id} className="tg-sentence">
                    {sentencePieces(s, gate).map((p, i) => (p.num ? <Num key={i} tone={p.warn ? "warn" : undefined}>{p.text}</Num> : <span key={i}>{p.text}</span>))}
                  </SummaryLine>
                );
              }
              return waitingSets.includes(id) ? (
                <p key={id} className="tg-sentence tg-sentence--loading" aria-busy="true">
                  <span>{SET_BY_ID[id].label}: loading…</span>
                  <Skeleton width={180} height={12} />
                </p>
              ) : null;
            })}
            {failed > 0 && !(scoredOnly && scoredN) && (
              <Caption>
                {formatCount(failed)} catalogue cutout{failed === 1 ? "" : "s"} failed to download and {failed === 1 ? "has" : "have"} no reconstruction.{" "}
                <button type="button" className="tg-link" onClick={() => setShowFailed(!showFailed)}>
                  {showFailed ? "Hide them" : "Show them"}
                </button>
              </Caption>
            )}
            {scoredN > 0 && scoredN < graded.length && (
              <Caption>
                {scoredOnly
                  ? <>Showing the {formatCount(scoredN)} scored tile{scoredN === 1 ? "" : "s"}, with their holes and R̃.{" "}</>
                  : <>Holes and R̃ are measured on {formatCount(scoredN)} scored tile{scoredN === 1 ? "" : "s"}.{" "}</>}
                <button type="button" className="tg-link" onClick={() => setScoredOnly(!scoredOnly)}>
                  {scoredOnly ? "Show every target" : "Show only them"}
                </button>
              </Caption>
            )}
          </div>
          {!!grades.length && (
            <div className="tg-filters" role="group" aria-label="Lens grades">
              {grades.map((g) => (
                <Chip key={g} on onRemove={() => setGrades(grades.filter((x) => x !== g))}>Grade {g}</Chip>
              ))}
            </div>
          )}
          <FluxStrip sets={scoredOnly ? readySets.filter((set) => view.some((r) => r.set === set)) : readySets} rows={view} onPick={open} />
          <DataTable rows={rows} columns={columns} rowKey={(r) => r.key} aria-label="Targets"
            selectable selected={selected} onSelectedChange={onSelected}
            inspect={(r) => (r.ref ? { kind: "tile", id: r.ref } : null)}
            defaultSort={[{ id: "flux", desc: false }]} toolbar={selectionBar}
            exportName="targets" urlKey="tg" className={tableQuery.trim() ? undefined : "tg-table--whole"} height="max(420px, calc(100vh - 280px))"
            filterPlaceholder="Filter: field:EDF-N  state:stale  flux<0.8  grade:A"
            empty={state !== "all" ? `No ${state} targets in these sets.` : "No targets in these sets."} />
        </>
      )}
    </Page>
  );
}
