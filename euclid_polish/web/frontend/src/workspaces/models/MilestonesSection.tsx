/* Models › Universal › Milestones of a long run (GET
   /ensemble/universal/milestones.json; helpers/milestones.py): the
   checkpoints a long run kept along the way (0.2 … 15M steps), each scored
   like a universal model (synthetic ∫PSNR on the universal test fields) and
   measured against JWST on the NEXUS field (½-coherence resolution of the
   Fourier ring correlation, per band, at each surface brightness of the
   knee scan; lower = sharper) beside the reference models measured in the
   same reports. Four charts and a table: ∫PSNR vs steps (references as
   lines), the resolution curves vs surface brightness (one per milestone,
   references dashed), the resolution at one brightness vs steps, and the
   learned output knee vs steps. Read-only (GETs). */
import { useMemo } from "react";
import Plot, { type Guide, type Series } from "../../charts/Plot";
import { categorical, viridis } from "../../colors";
import { useResource } from "../../api/query";
import { formatDate } from "../../format";
import { useUrlState } from "../../hooks/useUrlState";
import { logTicks } from "../../ticks";
import { Caption, Segmented, Select, SummaryLine, Num, Table, Toolbar, ToolbarGroup } from "../../ui";

type ScanPoint = { multiple: number; sb: number; arcsec: number };
type MilestoneRow = {
  step: number; spec: string; mean: number | null; integrated: (number | null)[] | null; hours: number | null;
  learned_knees: (number | null)[] | null; scored_at?: string | null;
};
type MemberMilestones = {
  member: string; steps: MilestoneRow[];
  resolution: Record<string, { jwst_filter?: string | null; noise_sb?: number | null; models: Record<string, ScanPoint[]> }>;
};
export type MilestonesPayload = { members: MemberMilestones[] };
/** A reference line of the ∫PSNR chart (production, another universal model). */
export type MilestoneReference = { label: string; mean: number | null; integrated?: (number | null)[] | null };

const BANDS = ["VIS", "Y_E", "J_E", "H_E"] as const;
type Band = (typeof BANDS)[number];
const BAND_SHORT: Record<Band, string> = { VIS: "VIS", Y_E: "Y", J_E: "J", H_E: "H" };
type Metric = "mean" | Band;
const MULTIPLES = ["0.1", "0.3", "1", "3", "10", "30", "100", "300", "1000"];
const parseBand = (v: string): Band => ((BANDS as readonly string[]).includes(v) ? (v as Band) : "H_E");
const parseMetric = (v: string): Metric => (v === "mean" || (BANDS as readonly string[]).includes(v) ? (v as Metric) : "mean");

export const milestonesUrl = "/ensemble/universal/milestones.json";
const stepText = (s: number) => (s >= 1e6 ? `${+(s / 1e6).toPrecision(3)}M` : `${Math.round(s / 1e3)}k`);
const refName = (spec: string) => (spec === "production" ? "production" : spec.replace(/^universal:member_/, "#").replace(/^member:member_/, "#"));

/** A model's resolution (arcsec) at a noise multiple (null when not measured). */
function at(scan: readonly ScanPoint[] | undefined, multiple: number): number | null {
  const hit = scan?.find((p) => Math.abs(p.multiple - multiple) < 1e-9);
  return hit ? hit.arcsec : null;
}

/** "3.1% sharper" / "2.0% blurrier" (never a signed %: lower arcsec = sharper). */
export function sharpnessText(model: number | null, reference: number | null): string {
  if (model == null || reference == null || reference <= 0) return "—";
  const d = (reference - model) / reference;
  if (Math.abs(d) < 0.0005) return "same";
  return `${(Math.abs(d) * 100).toFixed(1)}% ${d > 0 ? "sharper" : "blurrier"}`;
}

export function MilestonesSection({ references }: { references: readonly MilestoneReference[] }) {
  const res = useResource<MilestonesPayload>(milestonesUrl, [], { ttl: 60_000 });
  const [memberParam, setMember] = useUrlState("ms", "");
  const [metric, setMetric] = useUrlState<Metric>("msm", "mean", { parse: parseMetric });
  const [band, setBand] = useUrlState<Band>("msb", "H_E", { parse: parseBand });
  const [multiple, setMultiple] = useUrlState("msx", "1");
  const members = res.data?.members ?? [];
  const m = members.find((x) => x.member === memberParam) ?? members[members.length - 1] ?? null;
  const steps = useMemo(() => [...(m?.steps ?? [])].sort((a, b) => a.step - b.step), [m]);
  if (!m || !steps.length) return null;

  const scans = m.resolution[band]?.models ?? {};
  const msSpecs = steps.map((r) => r.spec).filter((s) => scans[s]);
  const refSpecs = Object.keys(scans).filter((s) => !s.startsWith("milestone:"));
  const mult = Number(multiple) || 1;
  const stepColor = (i: number) => viridis(steps.length > 1 ? (i / (steps.length - 1)) * 0.9 : 0.5);
  const xSteps: [number, number] = [steps[0].step / 1.6, steps[steps.length - 1].step * 1.6];
  const stepTicks = logTicks(xSteps).map((t) => ({ ...t, label: stepText(t.v) }));
  const bandIndex = metric === "mean" ? -1 : BANDS.indexOf(metric);
  const psnrOf = (r: { mean: number | null; integrated?: (number | null)[] | null }) =>
    (bandIndex < 0 ? r.mean : r.integrated?.[bandIndex] ?? null);
  const psnr = steps.map(psnrOf);
  const refPsnr = references.map((r) => ({ label: r.label, v: psnrOf(r) })).filter((r) => r.v != null);
  const allPsnr = [...psnr, ...refPsnr.map((r) => r.v)].filter((v): v is number => v != null);
  const psnrDom: [number, number] = allPsnr.length ? [Math.min(...allPsnr) - 0.3, Math.max(...allPsnr) + 0.3] : [55, 65];
  const psnrGuides: Guide[] = refPsnr.map((r, i) => ({ axis: "y", v: r.v as number, dash: [4, 3], color: categorical(i + 1), label: r.label }));

  // resolution curves vs surface brightness (x: the noise multiple, log)
  const curve = (spec: string, color: string, dash?: number[], name?: string): Series => {
    const pts = [...(scans[spec] ?? [])].sort((a, b) => a.multiple - b.multiple);
    return { x: pts.map((p) => Math.log10(p.multiple)), y: pts.map((p) => p.arcsec), color, dash, width: dash ? 1.4 : 2, dots: !dash, name };
  };
  const curves: Series[] = [
    ...msSpecs.map((s) => curve(s, stepColor(steps.findIndex((r) => r.spec === s)), undefined, `${stepText(Number(s.split("@")[1]))} steps`)),
    ...refSpecs.map((s, i) => curve(s, categorical(i + 1), [5, 3], refName(s))),
  ];
  const allX = curves.flatMap((s) => s.x);
  const allY = curves.flatMap((s) => s.y.filter((v): v is number => v != null));
  const xDom: [number, number] = allX.length ? [Math.min(...allX) - 0.15, Math.max(...allX) + 0.15] : [-1, 3];
  const yDom: [number, number] = allY.length ? [Math.max(0, Math.min(...allY) * 0.9), Math.max(...allY) * 1.05] : [0, 2];
  const xMultTicks = logTicks(xDom, { space: "log10" }).map((t) => ({ ...t, label: `${t.label}σ` }));

  // resolution at one brightness vs steps
  const atStep = steps.map((r) => at(scans[r.spec], mult));
  const refAt = refSpecs.map((s, i) => ({ spec: s, v: at(scans[s], mult), color: categorical(i + 1) })).filter((r) => r.v != null);
  const atAll = [...atStep, ...refAt.map((r) => r.v)].filter((v): v is number => v != null);
  const atDom: [number, number] = atAll.length ? [Math.min(...atAll) * 0.97, Math.max(...atAll) * 1.03] : [0, 2];
  const prodAt = at(scans.production, mult);
  const sbAt = (scans[msSpecs[msSpecs.length - 1]] ?? []).find((p) => Math.abs(p.multiple - mult) < 1e-9)?.sb;

  // the learned output knee vs steps
  const knees: Series[] = BANDS.map((b, i) => ({
    x: steps.map((r) => r.step), y: steps.map((r) => r.learned_knees?.[i] ?? null), color: categorical(i), width: 1.8, dots: true, name: BAND_SHORT[b],
  }));
  const kAll = knees.flatMap((s) => s.y.filter((v): v is number => v != null && v > 0));
  const kDom: [number, number] = kAll.length ? [Math.min(...kAll) / 1.6, Math.max(...kAll) * 1.6] : [0.01, 1];

  const last = steps[steps.length - 1];
  const prod = references.find((r) => r.label.startsWith("production"));
  const columns = [
    { id: "step", header: "Step", cell: (r: MilestoneRow) => stepText(r.step), align: "right" as const },
    { id: "hours", header: "Train h", cell: (r: MilestoneRow) => (r.hours == null ? "—" : r.hours.toFixed(1)), align: "right" as const },
    { id: "mean", header: "∫PSNR", cell: (r: MilestoneRow) => (r.mean == null ? "—" : r.mean.toFixed(2)), align: "right" as const },
    ...BANDS.map((b, i) => ({ id: `psnr-${b}`, header: BAND_SHORT[b], align: "right" as const,
      cell: (r: MilestoneRow) => (r.integrated?.[i] == null ? "—" : (r.integrated[i] as number).toFixed(2)) })),
    { id: "vs", header: "vs production", align: "right" as const,
      cell: (r: MilestoneRow) => (r.mean == null || prod?.mean == null ? "—" : `${r.mean - prod.mean >= 0 ? "+" : "−"}${Math.abs(r.mean - prod.mean).toFixed(2)} dB`) },
    ...BANDS.map((b) => ({ id: `res-${b}`, header: `${BAND_SHORT[b]} at ${multiple}σ`, align: "right" as const,
      cell: (r: MilestoneRow) => {
        const own = at(m.resolution[b]?.models[r.spec], mult);
        return own == null ? "—" : `${own.toFixed(3)}″ · ${sharpnessText(own, at(m.resolution[b]?.models.production, mult))}`;
      } })),
  ];

  return (
    <section className="mdl-stack mdl-stack--tight" aria-labelledby="uni-ms-title">
      <h2 id="uni-ms-title" className="mdl-h2">Milestones of a long run</h2>
      <SummaryLine>
        <Num>#{m.member.replace("member_", "")}</Num> at <Num>{stepText(last.step)}</Num> steps ({last.hours?.toFixed(1)} h):
        ∫PSNR <Num>{last.mean?.toFixed(2)}</Num> dB
        {prod?.mean != null && last.mean != null && <> ({last.mean - prod.mean >= 0 ? "+" : "−"}{Math.abs(last.mean - prod.mean).toFixed(2)} dB vs production)</>}
        {prodAt != null && atStep[atStep.length - 1] != null && <>; {BAND_SHORT[band]} at {multiple}σ: {sharpnessText(atStep[atStep.length - 1], prodAt)} than production on JWST</>}.
      </SummaryLine>
      <Toolbar label="Milestones view">
        {members.length > 1 && (
          <ToolbarGroup label="Run">
            <Select size="sm" aria-label="Long run" value={m.member} onChange={setMember}
              options={members.map((x) => ({ value: x.member, label: `#${x.member.replace("member_", "")}` }))} />
          </ToolbarGroup>
        )}
        <ToolbarGroup label="∫PSNR">
          <Segmented<Metric> size="sm" aria-label="Milestone PSNR band" value={metric} onChange={setMetric}
            options={[{ value: "mean", label: "Mean" }, ...BANDS.map((b) => ({ value: b, label: BAND_SHORT[b] }))]} />
        </ToolbarGroup>
        <ToolbarGroup label="JWST band">
          <Segmented<Band> size="sm" aria-label="Milestone resolution band" value={band} onChange={setBand}
            options={BANDS.map((b) => ({ value: b, label: BAND_SHORT[b], disabled: !m.resolution[b] }))} />
        </ToolbarGroup>
        <ToolbarGroup label="Brightness">
          <Select size="sm" aria-label="Surface brightness (noise multiple)" value={multiple} onChange={setMultiple}
            options={MULTIPLES.map((v) => ({ value: v, label: `${v}σ` }))} />
        </ToolbarGroup>
      </Toolbar>
      <div className="mdl-charts">
        <figure className="mdl-chart">
          <h3 className="mdl-chart__title">Synthetic ∫PSNR ({metric === "mean" ? "mean of the bands" : BAND_SHORT[metric as Band]}) vs training steps</h3>
          <Plot xScale="log" xDomain={xSteps} xTicks={stepTicks} yDomain={psnrDom} guides={psnrGuides}
            series={[{ x: steps.map((r) => r.step), y: psnr, color: categorical(0), width: 2.2, dots: true, name: `#${m.member.replace("member_", "")}` }]}
            xLabel="training steps" yLabel="∫PSNR [dB]" aspect={0.6} exportName={`milestones-psnr-${metric}`}
            xFormat={(v) => stepText(v)} aria-label="Milestone integrated PSNR vs training steps" />
        </figure>
        <figure className="mdl-chart">
          <h3 className="mdl-chart__title">JWST resolution in {BAND_SHORT[band]} ({m.resolution[band]?.jwst_filter ?? "NIRCam"}) vs surface brightness</h3>
          <Plot xDomain={xDom} xTicks={xMultTicks} yDomain={yDom} series={curves} legend="auto"
            guides={[{ axis: "x", v: 0, dash: [2, 3], label: "Euclid noise" }]}
            xLabel="surface brightness (× Euclid's noise per SR pixel)" yLabel="½-coherence resolution [″] (lower = sharper)"
            aspect={0.6} exportName={`milestones-resolution-${band}`} aria-label={`Resolution vs surface brightness, ${BAND_SHORT[band]}`} />
        </figure>
        <figure className="mdl-chart">
          <h3 className="mdl-chart__title">{BAND_SHORT[band]} resolution at {multiple}σ{sbAt != null ? ` (${sbAt.toFixed(1)} mag/″²)` : ""} vs training steps</h3>
          <Plot xScale="log" xDomain={xSteps} xTicks={stepTicks} yDomain={atDom}
            guides={refAt.map((r) => ({ axis: "y" as const, v: r.v as number, dash: [4, 3], color: r.color, label: refName(r.spec) }))}
            series={[{ x: steps.map((r) => r.step), y: atStep, color: categorical(0), width: 2.2, dots: true, name: `#${m.member.replace("member_", "")}` }]}
            xLabel="training steps" yLabel="resolution [″] (lower = sharper)" aspect={0.6} xFormat={(v) => stepText(v)}
            exportName={`milestones-resolution-${band}-${multiple}`} aria-label={`Resolution at ${multiple} sigma vs training steps, ${BAND_SHORT[band]}`} />
        </figure>
        <figure className="mdl-chart">
          <h3 className="mdl-chart__title">Learned output knee per band vs training steps</h3>
          <Plot xScale="log" xDomain={xSteps} xTicks={stepTicks} yScale="log" yDomain={kDom} series={knees} legend="auto"
            xLabel="training steps" yLabel="output knee [e⁻]" aspect={0.6} xFormat={(v) => stepText(v)}
            exportName="milestones-output-knee" aria-label="Learned output knee vs training steps" />
        </figure>
      </div>
      <Table className="mdl-table-mini" aria-label="Milestones" columns={columns} rows={steps} rowKey={(r) => String(r.step)} />
      <Caption>
        Each milestone is the run's state kept at that step, scored like any universal model: ∫PSNR on the universal models'
        {" "}synthetic test fields (dashed: the references named), and measured against JWST/NIRCam on the NEXUS field — the
        period where the Fourier ring correlation with JWST falls to ½, at each surface brightness (a multiple of Euclid's noise
        per SR pixel); lower is sharper. “sharper/blurrier” compares with production at that brightness.
        {last.scored_at ? ` Newest score ${formatDate(last.scored_at)}.` : ""}
      </Caption>
    </section>
  );
}
