/* Models › Diagnostics, the real-field section (?d=real-field; absorbs the
 * old Sky › Results field diagnostics, ?diag=1): on the cached legacy 10×10
 * real field, what the ensemble does WITHOUT an HR reference — the
 * model–model angular cross-correlation r(d) (?fd=cross) and the member σ
 * vs brightness (?fd=brightness) — each beside its synthetic STARFULL twin on
 * shared axes, with the caption naming the two ensembles ("14 real vs 30
 * synthetic members"). The RBF occupancy view is gone with the RBF. The data
 * loads only while the section is shown; Recompute asks first. */
import { useEffect, useMemo, type ReactNode } from "react";
import Plot, { type Series } from "../../../charts/Plot";
import { C } from "../../../colors";
import { isTerminal, useJob } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { formatCount, formatDeg } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { Button, Caption, EmptyState, JobProgress, Segmented, Skeleton, Tooltip } from "../../../ui";
import { url, type Evals } from "../api";
import { JOB, refreshFieldDiagnostics } from "../jobs";
import {
  brightnessPair, crossCurves, CROSS_X_DOMAIN, evenTicks, membersCaption,
  type FieldDiagnostics as Diag, type FieldStatus, type HeatPair,
} from "./realField";

type View = "cross" | "brightness";
const VIEWS: { value: View; label: string; title: string }[] = [
  { value: "cross", label: "r(d)", title: "Model–model angular cross-correlation (1 = identical Fourier structure)" },
  { value: "brightness", label: "σ vs brightness", title: "Member spread vs mean brightness (asinh space): ensemble variation, not error" },
];
const parseView = (raw: string): View | undefined => (raw === "cross" || raw === "brightness" ? raw : undefined);

const TTL = { ttl: 60_000 };
const CROSS_TICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10].map((v) => ({ v, label: String(v) }));
const CROSS_Y: [number, number] = [0, 1.05];

function Side({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="mdl-side">
      <h3 className="mdl-chart__title">{title}</h3>
      {children}
    </div>
  );
}

function Missing({ loading, error }: { loading: boolean; error?: string | null }) {
  if (loading) return <Skeleton height={260} />;
  return <p className="mdl-note">{error ? `Synthetic evaluation unavailable: ${error}` : "The synthetic evaluation has no analogous plot."}</p>;
}

function crossSeries(c: NonNullable<ReturnType<typeof crossCurves>>, color: string): Series[] {
  return [
    ...c.pairs.map((p) => ({ ...p, color: C.muted, width: 0.8, alpha: 0.22, key: "pairs" })),
    { ...c.median, color, width: 2.4, dash: [6, 3], name: "median rᵢⱼ(d)", key: "median" },
  ];
}

function CrossPlot({ series, xLabel, name }: { series: Series[]; xLabel: string; name: string }) {
  return (
    <Plot xScale="log" xDomain={CROSS_X_DOMAIN} yDomain={CROSS_Y} xTicks={CROSS_TICKS} yTicks={evenTicks(0, 1, 2)}
      xLabel={xLabel} yLabel="rᵢⱼ(d)" series={series} height={320} zoomAxes="xy"
      guides={[{ axis: "y", v: 1, color: C.guide, dash: [2, 3] }]}
      legend={[{ label: "model pairs", color: C.muted, key: "pairs" }, { label: "median rᵢⱼ(d)", color: series[series.length - 1]?.color ?? C.mean, key: "median", dash: true }]}
      legendToggle exportName={name} aria-label={`Cross-correlation r(d), ${name}`} />
  );
}

function HeatPlot({ pair, z, label, name }: { pair: HeatPair; z: number[][]; label: string; name: string }) {
  return (
    <Plot xDomain={pair.xDomain} yDomain={pair.yDomain} xTicks={evenTicks(...pair.xDomain)} yTicks={evenTicks(...pair.yDomain)}
      xLabel={pair.xLabel} yLabel={pair.yLabel} series={[]} height={300}
      heat={{ z, xEdges: pair.xEdges, yEdges: pair.yEdges, colorLabel: label }}
      exportName={name} aria-label={`${pair.yLabel} vs ${pair.xLabel}, ${name}`} />
  );
}

function Body({ view, diag, syn, synLoading, synError, fieldId }: {
  view: View; diag: Diag; syn: Evals | null; synLoading: boolean; synError: string | null; fieldId: string;
}) {
  const power = diag.model_power;
  const real = useMemo(() => crossCurves(power.k, power.r_pairs, power.r_cross, true), [power]);
  const synCross = useMemo(() => (syn?.ps?.theta ? crossCurves(syn.ps.theta, syn.ps.r_pairs ?? [], syn.ps.r_cross ?? [], false) : null), [syn]);
  const bright = useMemo(() => brightnessPair(diag, syn), [diag, syn]);
  const n = diag.member_labels.length;

  if (view === "cross") {
    return (
      <>
        <div className="mdl-pair">
          <Side title="Real field">
            {real ? <CrossPlot series={crossSeries(real, C.mean)} name={`field-${fieldId}-rd`}
              xLabel={`d [″] · ${power.pixel_scale_arcsec.toFixed(2)}″ px`} />
              : <p className="mdl-note">No cross-correlation measured.</p>}
          </Side>
          <Side title="Synthetic starfull twin">
            {synCross ? <CrossPlot series={crossSeries(synCross, C.comb)} name="synthetic-rd" xLabel="d [″]" />
              : <Missing loading={synLoading} error={synError} />}
          </Side>
        </div>
        <Caption>{formatCount(n * (n - 1) / 2)} real member pairs · no HR reference · the last measured bin is held to 10″</Caption>
      </>
    );
  }
  if (!bright) return <EmptyState compact title="No σ-vs-brightness histogram in these diagnostics" />;
  return (
    <div className="mdl-pair">
      <Side title="Real field"><HeatPlot pair={bright} z={bright.real} label="sampled pixels" name={`field-${fieldId}-sigma`} /></Side>
      <Side title="Synthetic starfull twin">
        {bright.synthetic ? <HeatPlot pair={bright} z={bright.synthetic} label="synthetic pixels" name="synthetic-sigma" />
          : <Missing loading={synLoading} error={synError} />}
      </Side>
    </div>
  );
}

export function FieldSection() {
  const [view, setView] = useUrlState<View>("fd", "cross", { parse: parseView });
  const field = useResource<FieldStatus>(url.fieldStatus(), [], TTL);
  const diag = useResource<{ diagnostics: Diag | null }>(url.fieldDiagnostics(), [], TTL);
  const syn = useResource<Evals>(url.evals("starfull"), ["starfull"], { ttl: 5 * 60_000 });
  const job = useJob(JOB.fieldDiagnostics);
  const f = field.data?.field ?? null;
  const d = diag.data?.diagnostics ?? null;
  const current = VIEWS.find((v) => v.value === view) ?? VIEWS[0];
  const reload = () => { void field.reload(); void diag.reload(); };
  const status = job.job?.status;
  useEffect(() => {
    if (status && isTerminal(status)) { void invalidate("/api/inference/"); }
  }, [status]);
  const recompute = () => { if (f) void refreshFieldDiagnostics(f.field_id); };

  let body: ReactNode;
  if (field.loading || diag.loading) body = <Skeleton height={320} />;
  else if (field.error || diag.error) {
    body = (
      <EmptyState compact icon="warn" title="Could not load the field diagnostics" action={<Button size="sm" onClick={reload}>Retry</Button>}>
        {(field.error ?? diag.error)?.message}
      </EmptyState>
    );
  } else if (!f) {
    body = <EmptyState compact title="No cached real field">The diagnostics come from the legacy 10×10 real field (Sky › Targets › Legacy field).</EmptyState>;
  } else if (!d) {
    body = (
      <EmptyState compact title="No diagnostics for this field yet" action={<Button size="sm" onClick={recompute}>Recompute…</Button>}>
        Recomputing derives the model–model and σ plots once.
      </EmptyState>
    );
  } else {
    body = <Body view={current.value} diag={d} syn={syn.data ?? null} synLoading={syn.loading}
      synError={syn.error?.message ?? null} fieldId={f.field_id} />;
  }
  const synMembers = syn.data?.n_members ?? null;
  return (
    <div className="mdl-stack mdl-stack--tight">
      <div className="mdl-row">
        <Segmented<View> size="sm" value={current.value} onChange={setView} aria-label="Real-field diagnostic"
          options={VIEWS.map((v) => ({ value: v.value, label: v.label, title: v.title }))} />
        <span className="mdl-muted">{current.title}</span>
        <span className="mdl-grow" />
        <Tooltip content="Apply the newest starfull combiners to the field and rewrite its diagnostics">
          <span><Button size="sm" icon="reset" disabled={!f} loading={job.busy} onClick={recompute}>Recompute…</Button></span>
        </Tooltip>
      </div>
      <JobProgress job={job.job} error={job.error} />
      {body}
      {f && (
        <Caption>
          {membersCaption(f, synMembers)}:
          {" "}the two panels are not the same ensemble · field {f.field_id} at {formatDeg(f.ra, 4)} {formatDeg(f.dec, 4, { signed: true })}
        </Caption>
      )}
    </div>
  );
}
