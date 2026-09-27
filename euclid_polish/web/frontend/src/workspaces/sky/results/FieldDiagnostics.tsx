/* Real-field diagnostics (the legacy Inference page's comparison, spec §7.2
 * follow-up): on the cached 10×10 real field, what the ensemble does WITHOUT
 * an HR reference — model–model angular cross-correlation, member σ vs
 * brightness and the RBF combiners' pixel occupancy — each beside its
 * synthetic STARFULL counterpart on shared axes. The view is in the URL
 * (`fd`); the data loads only while the section is shown. */
import { useMemo, type ReactNode } from "react";
import Plot, { type Series } from "../../../charts/Plot";
import { C } from "../../../colors";
import { useResource } from "../../../api/query";
import { formatCount, formatDeg } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { Button, EmptyState, IconButton, Section, Segmented, Skeleton, Tooltip } from "../../../ui";
import type { Evals } from "../../ensemble/api";
import { refreshFieldDiagnostics } from "./actions";
import { URLS } from "./api";
import {
  brightnessPair, crossCurves, CROSS_X_DOMAIN, evenTicks, occupancyViews,
  type FieldDiagnostics as Diag, type FieldStatus, type HeatPair,
} from "./diagnostics";
import { autoTicks } from "../../plotTicks";

type View = "cross" | "brightness" | "occupancy";
const VIEWS: { value: View; label: string; title: string }[] = [
  { value: "cross", label: "r(d)", title: "Model–model angular cross-correlation (1 = identical Fourier structure)" },
  { value: "brightness", label: "σ vs brightness", title: "Member spread vs mean brightness (asinh space) — ensemble variation, not error" },
  { value: "occupancy", label: "RBF occupancy", title: "Where the RBF combiners' pixels fall on their feature axes" },
];

const TTL = { ttl: 60_000 };
const CROSS_TICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10].map((v) => ({ v, label: String(v) }));
const CROSS_Y: [number, number] = [0, 1.05];

function Side({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="res-diag__side">
      <div className="res-diag__eyebrow">{title}</div>
      {children}
    </div>
  );
}

function Missing({ loading, error }: { loading: boolean; error?: string | null }) {
  if (loading) return <Skeleton height={260} />;
  return <p className="muted res-note">{error ? `Synthetic evaluation unavailable: ${error}` : "The synthetic evaluation has no analogous plot."}</p>;
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
  const occ = useMemo(() => occupancyViews(diag, syn), [diag, syn]);
  const n = diag.member_labels.length;

  if (view === "cross") {
    return (
      <>
        <p className="muted res-note">{formatCount(n * (n - 1) / 2)} member pairs · no HR reference · last measured bin held to 10″</p>
        <div className="res-diag__pair">
          <Side title="Real field">
            {real ? <CrossPlot series={crossSeries(real, C.mean)} name={`field-${fieldId}-rd`}
              xLabel={`d [″] · ${power.pixel_scale_arcsec.toFixed(2)}″ px`} />
              : <p className="muted res-note">No cross-correlation measured.</p>}
          </Side>
          <Side title="Synthetic STARFULL">
            {synCross ? <CrossPlot series={crossSeries(synCross, C.comb)} name="synthetic-rd" xLabel="d [″]" />
              : <Missing loading={synLoading} error={synError} />}
          </Side>
        </div>
      </>
    );
  }
  if (view === "brightness") {
    if (!bright) return <EmptyState compact title="No σ-vs-brightness histogram in these diagnostics" />;
    return (
      <div className="res-diag__pair">
        <Side title="Real field"><HeatPlot pair={bright} z={bright.real} label="sampled pixels" name={`field-${fieldId}-sigma`} /></Side>
        <Side title="Synthetic STARFULL">
          {bright.synthetic ? <HeatPlot pair={bright} z={bright.synthetic} label="synthetic pixels" name="synthetic-sigma" />
            : <Missing loading={synLoading} error={synError} />}
        </Side>
      </div>
    );
  }
  if (!occ.length) {
    return <EmptyState compact title="No RBF combiner on this field">The occupancy is recorded for the RBF combiners only; recompute after fitting one.</EmptyState>;
  }
  return (
    <>
      {occ.map((o) => (
        <div key={o.kind} className="res-diag__block">
          <div className="res-diag__eyebrow">{o.label} · {formatCount(o.pixels)} real pixels, four bands</div>
          {o.mode === "histogram" ? (
            <Plot xDomain={o.xDomain} yDomain={[0, Math.max(1, ...o.y)]} xTicks={evenTicks(...o.xDomain)}
              yTicks={autoTicks([0, Math.max(1, ...o.y)])}
              xLabel={o.xLabel} yLabel="log10(pixels + 1)" height={240}
              series={[{ x: o.x, y: o.y, color: C.mean, width: 2, name: o.label }]}
              exportName={`field-${fieldId}-${o.kind}`} aria-label={`${o.label} occupancy`} />
          ) : (
            <div className="res-diag__pair">
              <Side title="Real field"><HeatPlot pair={o.heat} z={o.heat.real} label="real pixels" name={`field-${fieldId}-${o.kind}`} /></Side>
              <Side title="Synthetic STARFULL">
                {o.heat.synthetic ? <HeatPlot pair={o.heat} z={o.heat.synthetic} label="synthetic pixels" name={`synthetic-${o.kind}`} />
                  : <Missing loading={synLoading} error={synError} />}
              </Side>
            </div>
          )}
        </div>
      ))}
    </>
  );
}

export function FieldDiagnostics({ onClose }: { onClose: () => void }) {
  const [view, setView] = useUrlState<View>("fd", "cross");
  const field = useResource<FieldStatus>(URLS.fieldStatus, [], TTL);
  const diag = useResource<{ diagnostics: Diag | null }>(URLS.fieldDiagnostics, [], TTL);
  const syn = useResource<Evals>(URLS.syntheticEvals, [], { ttl: 5 * 60_000 });
  const f = field.data?.field ?? null;
  const d = diag.data?.diagnostics ?? null;
  const current = VIEWS.find((v) => v.value === view) ?? VIEWS[0];
  const reload = () => { void field.reload(); void diag.reload(); };

  const right = (
    <span className="res-chips">
      <Segmented size="sm" value={current.value} onChange={setView} aria-label="Diagnostic"
        options={VIEWS.map((v) => ({ value: v.value, label: v.label, title: v.title }))} />
      <Tooltip content="Apply the newest STARFULL combiners to the field and rewrite its diagnostics">
        <Button size="sm" icon="reset" disabled={!f} onClick={() => { if (f) void refreshFieldDiagnostics(f.field_id, reload); }}>Recompute</Button>
      </Tooltip>
      <IconButton icon="close" size="sm" label="Hide field diagnostics" onClick={onClose} />
    </span>
  );
  const sub = f ? `${f.field_id} · ${formatDeg(f.ra, 4)} ${formatDeg(f.dec, 4, { signed: true })} · ${f.member_labels?.length ?? 0} members` : undefined;

  let body: ReactNode;
  if (field.loading || diag.loading) body = <Skeleton height={320} />;
  else if (field.error || diag.error) {
    body = (
      <EmptyState compact icon="warn" title="Could not load the field diagnostics"
        action={<Button size="sm" onClick={reload}>Retry</Button>}>
        {(field.error ?? diag.error)?.message}
      </EmptyState>
    );
  } else if (!f) {
    body = <EmptyState compact title="No cached real field">Diagnostics come from the legacy 10×10 field (source “field”).</EmptyState>;
  } else if (!d) {
    body = (
      <EmptyState compact title="No diagnostics for this field yet"
        action={<Button size="sm" onClick={() => void refreshFieldDiagnostics(f.field_id, reload)}>Recompute</Button>}>
        Recomputing derives the model–model, σ and occupancy plots once.
      </EmptyState>
    );
  } else {
    body = <Body view={current.value} diag={d} syn={syn.data ?? null} synLoading={syn.loading}
      synError={syn.error?.message ?? null} fieldId={f.field_id} />;
  }

  return (
    <Section title="Field diagnostics" sub={sub} right={right} className="res-diag" id="field-diagnostics">
      <p className="muted res-note" title={current.title}>{current.title}</p>
      {body}
    </Section>
  );
}
