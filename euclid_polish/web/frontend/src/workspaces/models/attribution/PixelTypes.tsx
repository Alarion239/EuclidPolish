/* Models › Knees › Pixel types (`?view=pixel-types`; spec
   docs/superpowers/specs/2026-10-15-knee-attribution-design.md › Pixel
   types): which kinds of pixel a model reads through which knee, from a
   finished knee-attribution run. Two sections:
   - one model's analysis: the toolbar picks the analysis (?pa=), the field
     set (?pk=), the band (?band=) and, for the groups, whether pixels are
     grouped by their knee or by the network's own preference over the
     chain rule (?pg=); the figure chips (?pf=) pick the figure the server
     drew, the band facts come from its summary.json; below, Analyse… runs the
     analysis of a finished run (a confirmed local job);
   - two models compared on the same pixels: the comparison (?pc=), its
     figure (?pcf=), and Compare… for a new pair (two models of runs with one
     seed; what pairs few pixels is noted).
   Opening the view only GETs; each job starts after a confirm. */
import { useMemo, useState } from "react";
import { useJob } from "../../../api/jobs";
import { invalidate } from "../../../api/query";
import { formatCount, formatDateTime } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Button, Callout, Caption, EmptyState, Field, JobProgress, Num, PngFigure, Segmented, Select, Skeleton, SummaryLine, Toolbar, ToolbarGroup,
} from "../../../ui";
import { JOB, startPixelCompare, startPixelTypes, useOnJobEnd } from "../jobs";
import { kneeUrl, usePixelReports, usePixelSummary, type KneeRunSummary, type PixelAnalysis, type PixelComparison } from "../kneesApi";
import { BAND_LABEL, FIELD_SETS, KNEE_BANDS, modelShort, parseBand, parseSet, runText, type FieldSet, type KneeBand } from "../knees";
import {
  analyseEstimate, analysisLabel, bandFacts, COMPARE_FIGURES, compareFigure, comparisonLabel, comparisonSpecs, modelRefs,
  normalisePixelReports, pairCheck, parseBasis, parseCompareFigure, parsePixelFigure, PIXEL_FIGURES, pixelFigure,
  type CompareFigure, type GroupBasis, type PixelFigure,
} from "../pixelTypes";

const SET_LABEL: Record<FieldSet, string> = { synthetic: "Synthetic", real: "Real" };
const BASIS_LABEL: Record<GroupBasis, string> = { learned: "Network's preference", raw: "Knee" };
const BASIS_TITLE: Record<GroupBasis, string> = {
  learned: "Group by the knee's excess over the chain-rule baseline: what the network itself prefers",
  raw: "Group by the effective knee itself (includes the stretch's own derivative)",
};
const POLL = 4000;
const when = (iso: string | null | undefined, fallback: string) => {
  const t = formatDateTime(iso ?? "");
  return t === "—" ? fallback : t;
};

export function PixelTypes({ runs }: { runs: readonly KneeRunSummary[] }) {
  const job = useJob(JOB.kneePixelTypes);
  const running = job.busy || job.job?.status === "running";
  const res = usePixelReports(true, running ? POLL : undefined);
  const reports = useMemo(() => normalisePixelReports(res.data), [res.data]);
  useOnJobEnd(job.job, () => { void invalidate(kneeUrl.pixelTypes()); });
  const finished = useMemo(() => runs.filter((r) => r.status !== "running" && r.status !== "failed" && r.models.length), [runs]);

  return (
    <div className="mdl-stack kn-pt">
      {(job.job || job.error) && <div className="mdl-jobs"><JobProgress job={job.job} error={job.error} /></div>}
      {res.loading && !res.data ? <Skeleton lines={6} /> : res.error ? (
        <EmptyState icon="warn" title="Could not read the pixel types"
          action={<Button size="sm" icon="reset" onClick={res.reload}>Retry</Button>}>
          <span className="mdl-mono">{res.error.message}</span>
        </EmptyState>
      ) : (
        <>
          <AnalysisSection analyses={reports.analyses} finished={finished} busy={running} />
          <CompareSection comparisons={reports.comparisons} runs={runs} busy={running} />
        </>
      )}
    </div>
  );
}

/* ── one model's analysis ──────────────────────────────────────────────── */

function AnalysisSection({ analyses, finished, busy }: {
  analyses: readonly PixelAnalysis[]; finished: readonly KneeRunSummary[]; busy: boolean;
}) {
  const [folderParam, setFolder] = useUrlState("pa", "");
  const [kindParam, setKind] = useUrlState<FieldSet>("pk", "synthetic", { parse: parseSet });
  const [band, setBand] = useUrlState<KneeBand>("band", "VIS", { parse: parseBand });
  const [figKey, setFig] = useUrlState<PixelFigure>("pf", "groups", { parse: parsePixelFigure });
  const [basis, setBasis] = useUrlState<GroupBasis>("pg", "learned", { parse: parseBasis });
  const a = analyses.find((x) => x.folder === folderParam) ?? analyses[0] ?? null;
  const kind: FieldSet = a && !a.kinds.includes(kindParam) && a.kinds.length ? a.kinds[0] : kindParam;
  const fig = pixelFigure(figKey);
  const file = fig.file(kind, band, basis);
  const has = (name: string) => !!a?.files.includes(name);
  const summaryRes = usePixelSummary(a?.folder ?? null, a?.created);
  const facts = useMemo(() => bandFacts(summaryRes.data, kind, band), [summaryRes.data, kind, band]);

  return (
    <section className="mdl-stack mdl-stack--tight" aria-labelledby="kn-pt-title">
      <h2 id="kn-pt-title" className="mdl-h2">Which pixels a model reads through which knee</h2>
      {a ? (
        <>
          <Toolbar label="Pixel types view">
            <ToolbarGroup label="Analysis">
              <Select size="sm" aria-label="Analysis" value={a.folder}
                onChange={(v) => setFolder(v === analyses[0]?.folder ? "" : v)}
                options={analyses.map((x) => ({ value: x.folder, label: analysisLabel(x, when(x.created, x.run_id)) }))} />
            </ToolbarGroup>
            <ToolbarGroup label="Fields">
              <Segmented<FieldSet> size="sm" aria-label="Pixel types fields" value={kind} onChange={setKind}
                options={FIELD_SETS.map((s) => ({ value: s, label: SET_LABEL[s], disabled: !a.kinds.includes(s),
                  title: a.kinds.includes(s) ? undefined : `This analysis has no ${s} fields` }))} />
            </ToolbarGroup>
            <ToolbarGroup label="Band">
              <Segmented<KneeBand> size="sm" aria-label="Pixel types band" value={band} onChange={setBand} disabled={!fig.perBand}
                options={KNEE_BANDS.map((b) => ({ value: b, label: BAND_LABEL[b],
                  disabled: fig.perBand && !has(fig.file(kind, b, basis)) }))} />
            </ToolbarGroup>
            {fig.basis && (
              <ToolbarGroup label="Group by">
                <Segmented<GroupBasis> size="sm" aria-label="Group by" value={basis} onChange={setBasis}
                  options={(["learned", "raw"] as const).map((v) => ({ value: v, label: BASIS_LABEL[v], title: BASIS_TITLE[v] }))} />
              </ToolbarGroup>
            )}
          </Toolbar>
          <Segmented<PixelFigure> size="sm" aria-label="Pixel types figure" value={fig.key} onChange={setFig} className="kn-pt__figs"
            options={PIXEL_FIGURES.map((f) => {
              const present = f.perBand ? KNEE_BANDS.some((b) => has(f.file(kind, b, basis))) : has(f.file(kind, band, basis));
              return { value: f.key, label: f.label, title: present ? f.title : `Not drawn for the ${kind} fields`, disabled: !present };
            })} />
          {facts && (
            <SummaryLine>
              <Num>{modelShort(a.model)}</Num>, {kind} fields, {BAND_LABEL[band]}: <Num>{formatCount(facts.n)}</Num> sampled pixels
              {facts.sigma != null && <>, noise σ = <Num>{facts.sigma.toPrecision(3)}</Num> e⁻ per pixel</>}
              {facts.blank ? <> ({(facts.blank * 100).toFixed(1)}% next to a blanked pixel, left out)</> : null}
              {facts.types ? <>; {facts.types} knee types</> : null}.
            </SummaryLine>
          )}
          {has(file) ? (
            <figure className="kn-pt__figure" aria-label={`${fig.title}, ${modelShort(a.model)}, ${kind}${fig.perBand ? `, ${BAND_LABEL[band]}` : ""}`}>
              <PngFigure srcFor={() => kneeUrl.pixelFile(a.folder, file, a.created)} alt={fig.title} minHeight={240} />
            </figure>
          ) : (
            <EmptyState compact icon="activity" title="Not drawn">
              This analysis has no {fig.label.toLowerCase()} figure for the {kind} fields{fig.perBand ? ` in ${BAND_LABEL[band]}` : ""}{" "}
              (too few pixels in every group).
            </EmptyState>
          )}
          <Caption>
            {fig.caption} Units: the band's noise σ per pixel; brightness is the truth (synthetic) or the LR above its sky,
            smoothed 3×3 (real). Pixels next to a blanked LR pixel are left out (there the lowest knee wins mechanically).
          </Caption>
        </>
      ) : (
        <EmptyState icon="activity" title="No pixel-types analysis yet">
          Analyse a finished run below: each sampled pixel's own-band knee profile becomes an effective knee, grouped by
          brightness, by what surrounds it and into knee types, each with its median stamps. A run with mixed sampling (the run
          panel's Sampling) gives true prevalences and pairs its pixels with any other run of the same seed.
        </EmptyState>
      )}
      <AnalyseRow finished={finished} analyses={analyses} busy={busy} />
    </section>
  );
}

function AnalyseRow({ finished, analyses, busy }: {
  finished: readonly KneeRunSummary[]; analyses: readonly PixelAnalysis[]; busy: boolean;
}) {
  const [picked, setPicked] = useState("");
  const run = finished.find((r) => r.id === picked) ?? finished[0] ?? null;
  if (!run) {
    return <p className="mdl-note">No finished knee-attribution run to analyse yet: start one in the run panel above.</p>;
  }
  const done = run.models.filter((spec) => analyses.some((x) => x.run_id === run.id && x.model === spec)).length;
  const estimate = analyseEstimate(run.models.length);
  const go = () => { void startPixelTypes({ run_id: run.id }, estimate); };
  return (
    <div className="kn-pt__start">
      <Field label="Analyse a run">
        <Select size="sm" aria-label="Run to analyse" value={run.id} onChange={setPicked}
          options={finished.map((r) => ({ value: r.id, label: runText(r, when(r.created, r.id)),
            hint: r.sampling && r.sampling !== "class" ? `${r.sampling} sampling` : "per-class sampling" }))} />
      </Field>
      <Button icon="activity" loading={busy} disabled={busy} onClick={go}>
        {done ? "Re-analyse…" : "Analyse…"}
      </Button>
      <span className="mdl-note">
        {estimate}{done ? ` · ${done} of ${run.models.length} already analysed` : ""}
        {run.sampling === "class" || !run.sampling ? " · per-class sampling: prevalences come from the weights, few pixels pair" : ""}
      </span>
    </div>
  );
}

/* ── two models on the same pixels ─────────────────────────────────────── */

function CompareSection({ comparisons, runs, busy }: {
  comparisons: readonly PixelComparison[]; runs: readonly KneeRunSummary[]; busy: boolean;
}) {
  const [folderParam, setFolder] = useUrlState("pc", "");
  const [kindParam, setKind] = useUrlState<FieldSet>("pck", "synthetic", { parse: parseSet });
  const [band, setBand] = useUrlState<KneeBand>("band", "VIS", { parse: parseBand });
  const [figKey, setFig] = useUrlState<CompareFigure>("pcf", "curves", { parse: parseCompareFigure });
  const c = comparisons.find((x) => x.folder === folderParam) ?? comparisons[0] ?? null;
  const kind: FieldSet = c && !c.kinds.includes(kindParam) && c.kinds.length ? c.kinds[0] : kindParam;
  const fig = compareFigure(figKey);
  const file = fig.file(kind, band, "learned");
  const has = (name: string) => !!c?.files.includes(name);
  const [specA, specB] = c ? comparisonSpecs(c) : ["", ""];

  return (
    <section className="mdl-stack mdl-stack--tight" aria-labelledby="kn-pc-title">
      <h2 id="kn-pc-title" className="mdl-h2">Two models on the same pixels</h2>
      {c ? (
        <>
          <Toolbar label="Comparison view">
            <ToolbarGroup label="Comparison">
              <Select size="sm" aria-label="Comparison" value={c.folder} onChange={(v) => setFolder(v === comparisons[0]?.folder ? "" : v)}
                options={comparisons.map((x) => ({ value: x.folder, label: comparisonLabel(x, when(x.created, x.folder)) }))} />
            </ToolbarGroup>
            <ToolbarGroup label="Fields">
              <Segmented<FieldSet> size="sm" aria-label="Comparison fields" value={kind} onChange={setKind}
                options={FIELD_SETS.map((s) => ({ value: s, label: SET_LABEL[s], disabled: !c.kinds.includes(s) }))} />
            </ToolbarGroup>
            <ToolbarGroup label="Band">
              <Segmented<KneeBand> size="sm" aria-label="Comparison band" value={band} onChange={setBand} disabled={!fig.perBand}
                options={KNEE_BANDS.map((b) => ({ value: b, label: BAND_LABEL[b], disabled: fig.perBand && !has(fig.file(kind, b, "learned")) }))} />
            </ToolbarGroup>
          </Toolbar>
          <Segmented<CompareFigure> size="sm" aria-label="Comparison figure" value={fig.key} onChange={setFig} className="kn-pt__figs"
            options={COMPARE_FIGURES.map((f) => {
              const present = f.perBand ? KNEE_BANDS.some((b) => has(f.file(kind, b, "learned"))) : has(f.file(kind, band, "learned"));
              return { value: f.key, label: f.label, title: present ? f.title : `Not drawn for the ${kind} fields`, disabled: !present };
            })} />
          <SummaryLine>
            B = <Num>{modelShort(specB)}</Num> against A = <Num>{modelShort(specA)}</Num>
            {c.tag ? <> ({c.tag})</> : null}, {kind} fields.
          </SummaryLine>
          {has(file) ? (
            <figure className="kn-pt__figure" aria-label={`${fig.title}, ${modelShort(specB)} vs ${modelShort(specA)}, ${kind}`}>
              <PngFigure srcFor={() => kneeUrl.pixelFile(c.folder, file, c.created)} alt={fig.title} minHeight={240} />
            </figure>
          ) : (
            <EmptyState compact icon="activity" title="Not drawn">
              This comparison has no {fig.label.toLowerCase()} figure for the {kind} fields{fig.perBand ? ` in ${BAND_LABEL[band]}` : ""}.
            </EmptyState>
          )}
          <Caption>{fig.caption}</Caption>
        </>
      ) : (
        <EmptyState compact icon="activity" title="No comparison yet">
          Compare two models below: the pixels both runs sampled (one seed draws the same uniform pixels in every run) are
          grouped by how differently the two models read them.
        </EmptyState>
      )}
      <CompareRow runs={runs} busy={busy} />
    </section>
  );
}

function CompareRow({ runs, busy }: { runs: readonly KneeRunSummary[]; busy: boolean }) {
  const refs = useMemo(() => modelRefs(runs, (r) => when(r.created, r.id)), [runs]);
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  if (refs.length < 2) {
    return <p className="mdl-note">A comparison needs two models: run the attribution on two models (or one model twice).</p>;
  }
  const refA = refs.find((r) => r.value === a) ?? refs[1] ?? null;
  const refB = refs.find((r) => r.value === b) ?? refs[0] ?? null;
  const check = pairCheck(refA, refB);
  const go = () => {
    if (!check.ok || !refA || !refB) return;
    void startPixelCompare({ a: refA.value, b: refB.value }, `${modelShort(refB.spec)} against ${modelShort(refA.spec)}`);
  };
  const options = refs.map((r) => ({ value: r.value, label: r.label, hint: `${r.sampling} sampling` }));
  return (
    <div className="kn-pt__start">
      <Field label="A (reference)">
        <Select size="sm" aria-label="Model A" value={refA?.value ?? ""} onChange={setA} options={options} />
      </Field>
      <Field label="B (compared)">
        <Select size="sm" aria-label="Model B" value={refB?.value ?? ""} onChange={setB} options={options} />
      </Field>
      <Button icon="activity" loading={busy} disabled={busy || !check.ok} onClick={go}>Compare…</Button>
      {check.error ? <span className="mdl-note mdl-warn" role="alert">{check.error}</span>
        : check.notes.length > 0 && <Callout tone="info" dense>{check.notes.join(" ")}</Callout>}
    </div>
  );
}
