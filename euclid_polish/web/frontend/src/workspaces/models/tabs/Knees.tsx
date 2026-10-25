/* Models › Knees (`/models/knees`): knee attribution — which asinh knees a
   multi-knee model relies on, for which pixels (spec
   docs/superpowers/specs/2026-10-15-knee-attribution-design.md). Two views
   (?view=): the attribution below, and the pixel types (`pixel-types`:
   ../attribution/PixelTypes — which kinds of pixel a model reads through
   which knee, and two models compared on the same pixels). From the top:
   - the run panel (collapsible; open when there is no run yet): models,
     synthetic test fields, real tiles, samples per class, seed; Run asks
     first with the cost, then the job's progress;
   - the toolbar (URL state): the run (?run=, newest by default), the models
     overlaid (?km=, up to four), the output band (?band=), the set
     (?set= synthetic | real) and the split (?split=): the gradient summed
     over the receptive field (`gradient`) or over the 3×3 LR pixels under
     the sample (`local`: every gradient view below swaps to it; a record
     before the local split notes it and draws the receptive field's), or
     the ablation (the class bars only; the other views keep the
     receptive-field gradient);
   - one summary line: which knee leads the faintest and the brightest pixels;
   - "Which knees matter for which pixels": knee share vs pixel brightness, a
     line per knee in a sequential ramp (faint knee → bright knee), a panel
     per model on shared axes, aggregated over all the run's fields; the
     input band (?input= all | VIS | Y | J | H) draws that band's own knee
     split here and in the class bars (all: the σ-weighted split; disabled
     under the ablation split, which ablates every input band at once);
   - by pixel class: stacked bars of the split for the synthetic truth
     classes and the real brightness classes;
   - the cross-band matrix per model;
   - the field explorer (the field image with its sampled pixels).
   Shares are noise-weighted (record version 2); an older record gets a
   warning. Opening the page reads only (GETs); the one job starts after a
   confirm. */
import { useMemo, useState, type ReactNode } from "react";
import Plot, { Legend, useLegend, type LegendItem, type Series } from "../../../charts/Plot";
import { useJob } from "../../../api/jobs";
import { invalidate } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { formatCount, formatDateTime } from "../../../format";
import { useResolvedTheme } from "../../../state/prefs";
import { useUrlState } from "../../../hooks/useUrlState";
import { logTicks } from "../../../ticks";
import {
  Button, Callout, Caption, Details, EmptyState, JobProgress, MultiSelect, Num, Page, Segmented, Select, Skeleton, SummaryLine, Toolbar,
  ToolbarGroup,
} from "../../../ui";
import { LoadState } from "../common";
import { JOB, useOnJobEnd } from "../jobs";
import { useKneeRun, useKneeRuns, type KneeRunRecord } from "../kneesApi";
import {
  BAND_LABEL, FIELD_SETS, KNEE_BANDS, KNEE_INPUTS, KNEE_RAMP, MAX_OVERLAY, SHARE_KINDS, brightnessExtent, brightnessLines, brightnessSamples,
  edgeLeads, effectiveInput, effectiveSplit, electronText, errorRows, isPerElectron, kneeLabel, kneeRamp, kneeUnion, modelShort,
  normaliseRecord, normaliseRuns, overlayModels, parseBand, parseInput, parseSet, parseShare, runText, setsWithData, sharePct, slug,
  type FieldSet, type KneeBand, type KneeInput, type ShareKind,
} from "../knees";
import { ClassBars, type DrawnModel } from "../attribution/ClassBars";
import { CrossBand } from "../attribution/CrossBand";
import { FieldExplorer } from "../attribution/FieldExplorer";
import { PixelTypes } from "../attribution/PixelTypes";
import { EMPTY_DRAFT, RunPanel, type RunDraft } from "../attribution/RunPanel";
import "../models.css";
import "../attribution/knees.css";

const SET_LABEL: Record<FieldSet, string> = { synthetic: "Synthetic", real: "Real" };
type KneeView = "attribution" | "pixel-types";
const parseView = (v: string): KneeView => (v === "pixel-types" ? "pixel-types" : "attribution");
const VIEW_TITLE: Record<KneeView, string> = {
  attribution: "Each knee's share of the sensitivity, by pixel class and brightness, per run",
  "pixel-types": "Which kinds of pixel a model reads through which knee (stamps, groups, types), and two models compared",
};
const INPUT_LABEL: Record<KneeInput, string> = { all: "All (σ-weighted)", VIS: "VIS", Y: "Y", J: "J", H: "H" };
const INPUT_TITLE: Record<KneeInput, string> = {
  all: "Every input band, each band's terms weighted by its noise σ", VIS: "The VIS input's own split over the knees",
  Y: "The Y input's own split over the knees", J: "The J input's own split over the knees", H: "The H input's own split over the knees",
};
const ABLATION_HINT = "The ablation removes a knee in every input band at once";
const SPLIT_LABEL: Record<ShareKind, string> = { gradient: "Gradient (receptive field)", local: "Gradient (local 3×3)", ablation: "Ablation" };
const SPLIT_TITLE: Record<ShareKind, string> = {
  gradient: "The chain-rule split summed over the receptive field (mostly the surrounding sky): what the prediction depends on overall",
  local: "The chain-rule split summed over the 3×3 LR pixels under the sample: how the pixel's own neighbourhood is read",
  ablation: "Each knee's share of the output change when its input channels are replaced by their window mean (the class bars)",
};
/** What each gradient split sums over, for the captions. */
const SCOPE_TEXT = {
  gradient: "Summed over the whole receptive field, mostly the surrounding sky, it says what the prediction depends on overall; the "
    + "local 3×3 split says how the pixel's own neighbourhood is read.",
  local: "Summed only over the 3×3 LR pixels under the sample, it says how the pixel's own neighbourhood is read; the receptive-field "
    + "split says what the prediction depends on overall (mostly the surrounding sky).",
};
const POLL = 4000;
const FULL_DOMAIN: [number, number] = [1e-3, 1e7];

/* ── the main chart ────────────────────────────────────────────────────── */

function BrightnessPanels({ models, set, band, input, kind, color, union }: {
  models: readonly DrawnModel[]; set: FieldSet; band: KneeBand; input: KneeInput; kind: ShareKind; color: (q: number) => string;
  union: readonly number[];
}) {
  const lg = useLegend();
  const panels = useMemo(() => models.map((m) => {
    const agg = m.result.aggregate[set];
    const lines = brightnessLines(agg, band, m.result.knees, input, kind);
    const series: Series[] = lines.filter((l) => l.x.length).map((l) => ({
      x: l.x, y: l.y, color: color(l.knee), width: 1.8, dots: true, name: kneeLabel(l.knee), key: `k:${+l.knee.toPrecision(6)}`,
    }));
    return { m, lines, series, n: brightnessSamples(agg, band) };
  }), [models, set, band, input, kind, color]);
  const ext = panels.map((p) => brightnessExtent(p.lines)).filter(Boolean) as [number, number][];
  const xDomain: [number, number] = ext.length
    ? [Math.min(...ext.map((e) => e[0])) / 1.6, Math.max(...ext.map((e) => e[1])) * 1.6] : FULL_DOMAIN;
  const items: LegendItem[] = union.map((q) => ({ label: kneeLabel(q), color: color(q), key: `k:${+q.toPrecision(6)}`, line: true }));
  const drawn = panels.filter((p) => p.series.length);
  if (!drawn.length) {
    return (
      <EmptyState compact icon="activity" title={`No ${set} samples in ${BAND_LABEL[band]}`}>
        This run sampled no {set} pixel in {BAND_LABEL[band]} for the models shown.
      </EmptyState>
    );
  }
  const many = drawn.length > 1;
  const local = kind === "local";
  const via = `${local ? ", local 3×3" : ""}${input === "all" ? "" : `, ${input} input`}`;
  return (
    <>
      <Legend items={items} {...lg.legendProps} />
      <div className={many ? "mdl-charts" : undefined}>
        {drawn.map((p) => (
          <div key={p.m.spec} className="mdl-chart kn-chart" data-model={p.m.spec}>
            <h3 className="mdl-chart__title">
              {modelShort(p.m.spec)}{p.m.result.depth != null ? ` · depth ${p.m.result.depth}` : ""} · {p.m.result.knees.length} knees ·{" "}
              n = {formatCount(p.n)}
            </h3>
            <Plot {...lg.plotProps} xScale="log" xDomain={xDomain} xTicks={logTicks(xDomain, { maxTicks: 7 })} yDomain={[0, 1]}
              xLabel={`pixel brightness in ${BAND_LABEL[band]} [e⁻]`}
              yLabel={`${local ? "local " : ""}knee share${input === "all" ? "" : ` of the ${input} input`}`}
              series={p.series} aspect={many ? 0.6 : 0.45} syncKey="knee-brightness"
              xFormat={(v) => electronText(v)} yFormat={(v) => sharePct(v)}
              exportName={`knee-share-${slug(p.m.spec)}-${band}-${set}${local ? "-local" : ""}${input === "all" ? "" : `-in-${input}`}`}
              aria-label={`Knee share vs pixel brightness, ${modelShort(p.m.spec)}, ${BAND_LABEL[band]}, ${set}${via}`} />
          </div>
        ))}
      </div>
    </>
  );
}

/* ── the tab ───────────────────────────────────────────────────────────── */

export default function Knees() {
  const job = useJob(JOB.kneeAttribution);
  const jobRunning = job.busy || job.job?.status === "running";
  const runsRes = useKneeRuns(jobRunning ? POLL : undefined);
  const runs = useMemo(() => normaliseRuns(runsRes.data), [runsRes.data]);
  const [runParam, setRunParam] = useUrlState("run", "");
  const [picked, setPicked] = useUrlState<string[]>("km", []);
  const [band, setBand] = useUrlState<KneeBand>("band", "VIS", { parse: parseBand });
  const [setParam, setSet] = useUrlState<FieldSet>("set", "synthetic", { parse: parseSet });
  const [splitParam, setSplit] = useUrlState<ShareKind>("split", "gradient", { parse: parseShare });
  const [inputParam, setInput] = useUrlState<KneeInput>("input", "all", { parse: parseInput });
  const [view, setView] = useUrlState<KneeView>("view", "attribution", { parse: parseView });
  const [panelOpen, setPanelOpen] = useState<boolean | null>(null);
  const [draft, setDraft] = useState<RunDraft>(EMPTY_DRAFT);

  const runId = runParam && (!runsRes.data || runs.some((r) => r.id === runParam)) ? runParam : runs[0]?.id ?? null;
  const summary = runs.find((r) => r.id === runId) ?? null;
  // the pixel types read their own figures: no run record (tens of MB for a mixed run)
  const recRes = useKneeRun(view === "pixel-types" ? null : runId, summary?.status === "running" || jobRunning ? POLL : undefined);
  const rec: KneeRunRecord | null = useMemo(() => normaliseRecord(recRes.data), [recRes.data]);
  const status = rec?.status ?? summary?.status ?? null;

  useOnJobEnd(job.job, () => { void invalidate("/api/knee-attribution"); });
  const onStarted = () => {
    setRunParam("");
    setPanelOpen(false);
    void invalidate("/api/knee-attribution");
  };

  const noRuns = !!runsRes.data && runs.length === 0;
  const open = panelOpen ?? noRuns;
  usePageActions([
    { id: "knees-run", label: "Knees: set up a knee-attribution run…", group: "Models", keywords: ["knee", "attribution", "multi-knee", "gradient"],
      run: () => setPanelOpen(true) },
    { id: "knees-pixel-types", label: "Knees: pixel types", group: "Models", keywords: ["knee", "pixel", "types", "stamps", "compare"],
      run: () => setView("pixel-types") },
  ]);

  const shown = useMemo(() => overlayModels(rec, picked), [rec, picked]);
  const drawn: DrawnModel[] = useMemo(() => (rec ? shown.map((spec) => ({ spec, result: rec.models[spec] })) : []), [rec, shown]);
  const sets = useMemo(() => setsWithData(rec, shown), [rec, shown]);
  const set: FieldSet = sets.includes(setParam) || !sets.length ? setParam : sets[0];
  const union = useMemo(() => kneeUnion(drawn.map((d) => d.result.knees)), [drawn]);
  const theme = useResolvedTheme();
  const color = useMemo(() => kneeRamp(union, KNEE_RAMP[theme]), [union, theme]);
  // the split and the input split drawn (the URL's picks unless an older
  // record or the ablation split rules them out), over every aggregate the
  // page draws
  const aggs = useMemo(() => drawn.flatMap((d) => sets.map((s) => d.result.aggregate[s])), [drawn, sets]);
  const splitView = useMemo(() => effectiveSplit(splitParam, aggs), [splitParam, aggs]);
  const split = splitView.split;
  const inputView = useMemo(() => effectiveInput(inputParam, split, aggs), [inputParam, split, aggs]);
  const input = inputView.input;
  const leads = useMemo(() => {
    const first = drawn[0];
    return first ? edgeLeads(brightnessLines(first.result.aggregate[set], band, first.result.knees, input, split)) : null;
  }, [drawn, set, band, input, split]);
  const scope = split === "local" ? "local" : "gradient";
  const errors = rec ? errorRows(rec.errors) : [];
  const nFields = drawn[0]?.result.fields.filter((f) => f.kind === set).length ?? 0;

  const runPanel = (
    <RunPanel open={open} onOpenChange={setPanelOpen} draft={draft} setDraft={setDraft} busy={jobRunning} onStarted={onStarted} />
  );
  const jobs = (job.job || job.error) && <div className="mdl-jobs"><JobProgress job={job.job} error={job.error} /></div>;

  const viewSwitch = (
    <div className="mdl-row kn-view">
      <Segmented<KneeView> aria-label="Knees view" value={view} onChange={setView}
        options={(["attribution", "pixel-types"] as const).map((v) => ({
          value: v, label: v === "attribution" ? "Attribution" : "Pixel types", title: VIEW_TITLE[v],
        }))} />
    </div>
  );
  if (view === "pixel-types") {
    return (
      <Page className="mdl-page kn-page">
        {viewSwitch}
        {runPanel}
        {jobs}
        <LoadState loading={runsRes.loading} error={runsRes.error} onRetry={runsRes.reload}>
          <PixelTypes runs={runs} />
        </LoadState>
      </Page>
    );
  }

  return (
    <Page className="mdl-page kn-page">
      {viewSwitch}
      {runPanel}
      {jobs}
      <LoadState loading={runsRes.loading} error={runsRes.error} onRetry={runsRes.reload}
        empty={noRuns && (
          <EmptyState icon="activity" title="No knee attribution runs yet"
            action={open ? undefined : <Button variant="primary" onClick={() => setPanelOpen(true)}>Set up a run</Button>}>
            A run splits each sampled output pixel's sensitivity over the model's input knees (the exact chain-rule split, plus an
            ablation check), for multi-knee models on synthetic test fields and real tiles. Each input band's terms are weighted by
            its noise σ, so a share is the output's response to a 1σ change of the input. Pick the models and fields in the run
            panel above; it runs locally, never on FASRC.
          </EmptyState>
        )}>
        {runs.length > 0 && (
          <div className="mdl-stack">
            <Toolbar label="Knee attribution view">
              <ToolbarGroup label="Run">
                <Select size="sm" aria-label="Run" value={runId ?? ""} onChange={(v) => setRunParam(v === runs[0]?.id ? "" : v)}
                  options={runs.map((r) => ({ value: r.id, label: runText(r, formatDateTime(r.created) === "—" ? r.id : formatDateTime(r.created)) }))} />
              </ToolbarGroup>
              {rec && Object.keys(rec.models).length > 1 && (
                <ToolbarGroup label="Models">
                  <MultiSelect size="sm" aria-label="Models shown" value={shown}
                    onChange={(v) => setPicked(v.length > MAX_OVERLAY ? v.slice(v.length - MAX_OVERLAY) : v)}
                    options={Object.keys(rec.models).map((spec) => ({ value: spec, label: modelShort(spec), hint: `${rec.models[spec].knees.length} knees` }))} />
                </ToolbarGroup>
              )}
              <ToolbarGroup label="Band">
                <Segmented<KneeBand> size="sm" aria-label="Band" value={band} onChange={setBand}
                  options={KNEE_BANDS.map((b) => ({ value: b, label: BAND_LABEL[b] }))} />
              </ToolbarGroup>
              <ToolbarGroup label="Fields">
                <Segmented<FieldSet> size="sm" aria-label="Field set" value={set} onChange={setSet}
                  options={FIELD_SETS.map((s) => ({ value: s, label: SET_LABEL[s], disabled: !!rec && !sets.includes(s),
                    title: rec && !sets.includes(s) ? `This run has no ${s} fields` : undefined }))} />
              </ToolbarGroup>
              <ToolbarGroup label="Split">
                <Segmented<ShareKind> size="sm" aria-label="Split" value={split} onChange={setSplit}
                  options={SHARE_KINDS.map((k) => ({ value: k, label: SPLIT_LABEL[k], title: SPLIT_TITLE[k] }))} />
              </ToolbarGroup>
            </Toolbar>
            <RunBody rec={rec} loading={recRes.loading} error={recRes.error} reload={recRes.reload} status={status} errors={errors}>
              {rec && drawn.length > 0 && (
                <>
                  {splitView.missing && (
                    <p className="mdl-note mdl-warn">
                      This run predates the local split; re-run to see it. Showing the receptive-field gradient.
                    </p>
                  )}
                  {leads?.faint && (
                    <SummaryLine>
                      For <Num>{modelShort(drawn[0].spec)}</Num> in {BAND_LABEL[band]} ({set} fields
                      {split === "local" ? ", local 3×3" : ""}{input === "all" ? "" : `, ${input} input`}), the faintest pixels
                      (~{electronText(leads.faint.x)}) lean on the <Num>{kneeLabel(leads.faint.knee)}</Num> knee
                      ({sharePct(leads.faint.share)})
                      {leads.bright && <>, the brightest (~{electronText(leads.bright.x)}) on the <Num>{kneeLabel(leads.bright.knee)}</Num> knee
                        ({sharePct(leads.bright.share)})</>}.
                    </SummaryLine>
                  )}
                  <section className="mdl-stack mdl-stack--tight" aria-labelledby="kn-main-title">
                    <div className="mdl-row">
                      <h2 id="kn-main-title" className="mdl-h2">Which knees matter for which pixels</h2>
                      <ToolbarGroup label="Input band">
                        <span className="kn-input" title={inputView.disabled ? ABLATION_HINT : undefined}>
                          <Segmented<KneeInput> size="sm" aria-label="Input band" value={input} onChange={setInput}
                            disabled={inputView.disabled}
                            options={KNEE_INPUTS.map((v) => ({
                              value: v, label: INPUT_LABEL[v], title: inputView.disabled ? ABLATION_HINT : INPUT_TITLE[v],
                            }))} />
                        </span>
                        {inputView.disabled && <span className="mdl-faint">ablation: every input band at once</span>}
                      </ToolbarGroup>
                    </div>
                    {inputView.missing && (
                      <p className="mdl-note mdl-warn">This run predates the per-input split; re-run to see it. Showing all input bands.</p>
                    )}
                    <BrightnessPanels models={drawn} set={set} band={band} input={input} kind={split} color={color} union={union} />
                    <Caption>
                      {input === "all" ? (
                        <>
                          The share of the model's sensitivity that flows through each input knee (the exact chain-rule split of ∂y/∂x
                          over the knees' stretched channels). Each input band's terms are weighted by its noise σ, so a share is the
                          output's response to a 1σ change of the input.
                        </>
                      ) : (
                        <>
                          The {input} input's own split over the knees (its chain-rule terms alone, so σ cancels). The knees are in
                          electrons, so the same knee sits at very different noise levels per band: 1 e⁻ ≈ 1σ in NISP but ≈ 0.06σ in VIS.
                        </>
                      )}{" "}
                      {SCOPE_TEXT[scope]} Averaged over the sampled pixels of each brightness bin (half a decade); brightness is the{" "}
                      {set === "synthetic" ? "truth" : "model's SR"} in {BAND_LABEL[band]}. Over all {formatCount(nFields)} {set} fields of
                      the run; empty bins are skipped.
                      {split === "ablation" && " The ablation has no brightness curves: this chart is the receptive-field gradient."}
                      {rec.knee_grid_note ? ` ${rec.knee_grid_note}` : ""}
                    </Caption>
                  </section>
                  <section className="mdl-stack mdl-stack--tight" aria-labelledby="kn-class-title">
                    <h2 id="kn-class-title" className="mdl-h2">By pixel class</h2>
                    <div className="kn-panels">
                      {FIELD_SETS.filter((s) => sets.includes(s)).map((s) => (
                        <ClassBars key={s} set={s} models={drawn} band={band} kind={split} input={input} color={color} />
                      ))}
                    </div>
                  </section>
                  <section className="mdl-stack mdl-stack--tight" aria-labelledby="kn-cross-title">
                    <h2 id="kn-cross-title" className="mdl-h2">Cross-band</h2>
                    <div className="kn-crossgrid">
                      {drawn.map((d) => <CrossBand key={d.spec} agg={d.result.aggregate[set]} title={modelShort(d.spec)} kind={split} />)}
                    </div>
                    <Caption>
                      Rows: the output band; columns: the input band. Each cell is the mean share of the output pixel's sensitivity that
                      flows through that input band ({set} fields, all knees), each band's terms weighted by its noise σ: the output's
                      response to a 1σ change of that input. {split === "local"
                        ? "Local 3×3: summed only over the LR pixels under the sample."
                        : "Receptive field: summed over everything the prediction sees."}
                    </Caption>
                  </section>
                  <section className="mdl-stack mdl-stack--tight" aria-labelledby="kn-field-title">
                    <h2 id="kn-field-title" className="mdl-h2">Field explorer</h2>
                    <FieldExplorer runId={rec.id} models={drawn} band={band} set={set} kind={split} color={color} />
                  </section>
                </>
              )}
            </RunBody>
          </div>
        )}
      </LoadState>
    </Page>
  );
}

/** The run's state around its figures: loading, unreadable, running (the
 *  figures fill in as models finish), failed (the errors), partial errors. */
function RunBody({ rec, loading, error, reload, status, errors, children }: {
  rec: KneeRunRecord | null; loading: boolean; error: { message: string; status: number } | null; reload: () => void;
  status: string | null; errors: { where: string; reason: string }[]; children: ReactNode;
}) {
  if (!rec) {
    if (loading) return <Skeleton lines={6} />;
    if (error) {
      return (
        <EmptyState icon="warn" title="Could not read this run" action={<Button size="sm" icon="reset" onClick={reload}>Retry</Button>}>
          <span className="mdl-mono">{error.message}</span>
        </EmptyState>
      );
    }
    return null;
  }
  const nModels = Object.keys(rec.models).length;
  const errorList = (
    <ul className="kn-errors">
      {errors.slice(0, 6).map((e) => <li key={e.where}><strong>{e.where}</strong>: <span className="mdl-mono">{e.reason}</span></li>)}
    </ul>
  );
  return (
    <>
      {status === "running" && (
        <Callout tone="info" dense>
          This run is still being written: {nModels} of {rec.params.models.length} models done so far; the figures fill in as each finishes.
        </Callout>
      )}
      {status === "cancelled" && nModels > 0 && (
        <Callout tone="info" dense>
          This run was cancelled: it kept the {nModels} of {rec.params.models.length} models it finished.
        </Callout>
      )}
      {nModels > 0 && isPerElectron(rec) && (
        <Callout tone="warn" dense>
          Per-electron weights (record v1): band and knee shares lean toward NISP; re-run for noise-weighted shares.
        </Callout>
      )}
      {status === "failed" && (
        <Callout tone="bad" title="This run failed">
          {errors.length ? errorList : "It recorded no model and no reason."}
          {errors.length > 6 && <Details summary={`${errors.length - 6} more errors`}>
            <ul className="kn-errors">{errors.slice(6).map((e) => <li key={e.where}><strong>{e.where}</strong>: {e.reason}</li>)}</ul>
          </Details>}
        </Callout>
      )}
      {status !== "failed" && errors.length > 0 && (
        <Callout tone="warn" title={`${errors.length} ${errors.length === 1 ? "part" : "parts"} of this run failed`}>
          {errorList}
          {errors.length > 6 && <Details summary={`${errors.length - 6} more errors`}>
            <ul className="kn-errors">{errors.slice(6).map((e) => <li key={e.where}><strong>{e.where}</strong>: {e.reason}</li>)}</ul>
          </Details>}
        </Callout>
      )}
      {status !== "failed" && nModels === 0 && status !== "running" && (
        <EmptyState compact icon="activity" title="No model results in this run">
          {status === "cancelled" ? "The run was cancelled before a model finished." : "The run finished without a model result."}
        </EmptyState>
      )}
      {children}
    </>
  );
}
