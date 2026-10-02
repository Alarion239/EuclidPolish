/* Inspector kind `combiner` (`combiner:<variant dir>`, e.g.
   `combiner:spatial_gate_linear`; an older id with the regime in front still
   opens): one combiner variant — its fit (knobs, held-out selection),
   membership vs the active members, held-out curve, test / knee-integrated
   PSNR, and the promote / compare actions. */
import { useNavigate } from "react-router-dom";
import Plot from "../../charts/Plot";
import { C } from "../../colors";
import { useJob } from "../../api/jobs";
import type { InspectorProps } from "../../app/inspector";
import { pagePath } from "../../app/nav";
import { formatDateTime } from "../../format";
import { Badge, Button, DefList, EmptyState, JobProgress, JsonTree, Section, Skeleton, confirm } from "../../ui";
import { BAND_SHORT, REGIME, useCombiners } from "./api";
import { JOB, useOnJobEnd } from "./jobs";
import { combinerVariant, db, memberNumber, variantLabel } from "./model";
import "./models.css";

const nums = (labels: string[]) => labels.map((l) => memberNumber(l) ?? l).join(", ");

export default function CombinerInspector({ id }: InspectorProps) {
  const name = combinerVariant(id);
  const res = useCombiners();
  const navigate = useNavigate();
  const promote = useJob(JOB.promote);
  const compare = useJob(JOB.compare);
  useOnJobEnd(promote.job);
  useOnJobEnd(compare.job);
  if (res.loading) return <Skeleton lines={6} />;
  const v = res.data?.variants.find((x) => x.name === name || x.spec === name);
  if (res.error || !v) {
    return <EmptyState icon="warn" title={`No combiner ${name}`}><span className="mdl-mono">{res.error?.message ?? "not among the combiner variants"}</span></EmptyState>;
  }
  const f = v.fit as Record<string, unknown>;
  const hist = v.history.filter((h) => h.loss != null);
  const ys = hist.map((h) => h.loss as number);
  const kneeVals = (v.knee?.integrated ?? []).filter((x): x is number => Number.isFinite(x));
  const kneeMean = kneeVals.length ? kneeVals.reduce((a, b) => a + b, 0) / kneeVals.length : null;
  const doPromote = async () => {
    const mismatch = !v.membership.current;
    if (await confirm({
      title: `Promote ${v.name} to production?`,
      message: `The production gate is backed up to spatial_gate_backup_<UTC time> first.${mismatch ? " It was fitted for other members than the active ones." : ""}`,
      tone: mismatch ? "danger" : "default", confirmLabel: "Promote", ...(mismatch ? { requireText: "promote" } : {}),
    })) await promote.run("/ensemble/combiners/promote", { mode: REGIME, variant: v.name, ...(mismatch ? { force: "1" } : {}) });
  };
  const doCompare = async () => {
    if (await confirm({ title: `Compare ${variantLabel(v.name)} with production?`, message: "Scores both on the cached test cubes and blackout copies.", confirmLabel: "Compare" })) {
      await compare.run("/ensemble/combiners/compare", { mode: REGIME, gates: [res.data?.production ?? "spatial_gate_combiner", v.name].filter((x, i, a) => a.indexOf(x) === i).join(",") });
    }
  };
  return (
    <div className="mdl-insp">
      <div className="mdl-insp__head">
        <span className="mdl-insp__title">{variantLabel(v.name)}</span>
        {v.production && <Badge tone="accent">production</Badge>}
        {v.backup && <Badge>backup</Badge>}
        {!v.membership.current && <Badge tone="warn">other members</Badge>}
      </div>
      <div className="mdl-row">
        {v.kind === "gate" && !v.production && <Button size="sm" variant="primary" loading={promote.busy} onClick={() => void doPromote()}>Promote…</Button>}
        {v.kind === "gate" && v.applies_to_test_cubes && <Button size="sm" loading={compare.busy} onClick={() => void doCompare()}>Compare</Button>}
        <Button size="sm" variant="ghost" onClick={() => navigate(pagePath("models", { tab: "combiner" }))}>Combiner</Button>
      </div>
      <JobProgress job={promote.job} error={promote.error} />
      <JobProgress job={compare.job} error={compare.error} />
      <DefList dense items={[
        ["directory", <code key="d">{v.name}</code>],
        ["members", `${v.n_reads}${v.pruned ? ` of ${v.n_members} (pruned)` : ""}${v.membership.current ? " · the active set" : ""}`],
        !v.membership.current ? ["missing now", nums(v.membership.missing) || "none"] : null,
        !v.membership.current ? ["not read", nums(v.membership.extra) || "none"] : null,
        ["mix · LR · width", `${v.mix_space ?? "—"} · ${v.use_lr ? "LR input" : "no LR"} · ${v.width ?? "—"}`],
        ["fitted", v.fitted_at ? formatDateTime(v.fitted_at) : "—"],
        f.steps != null ? ["steps", `${f.steps_run ?? f.steps}${f.complete === false ? " (stopped early)" : ""}`] : null,
        f.learning_rate != null ? ["lr · batch · crop", `${f.learning_rate} · ${f.batch_size ?? "—"} · ${f.crop ?? "—"}`] : null,
        Array.isArray(f.loss_knees_e) ? ["loss knees", `${(f.loss_knees_e as number[]).length} (${(f.loss_knees_e as number[]).map((x) => +x.toPrecision(2)).join(", ")} e⁻)`] : null,
        f.promoted_from ? ["promoted from", String(f.promoted_from)] : null,
        ["held-out loss", v.selected?.loss != null ? `${v.selected.loss.toFixed(4)} (baseline ${v.baseline?.loss?.toFixed(4) ?? "—"})` : "—"],
        v.eval?.psnr != null ? ["test PSNR (eval)", `${db(v.eval.psnr, 3)} dB`] : null,
        v.test?.band_psnr ? ["test PSNR (compare)", v.test.band_psnr.map((x, i) => `${BAND_SHORT[["VIS", "Y_E", "J_E", "H_E"][i]]} ${db(x)}`).join(" · ")] : null,
        kneeMean != null ? ["∫PSNR", `${db(kneeMean, 3)} dB mean · from ${v.knee?.source}${v.knee?.stale ? " (stale)" : ""}`] : null,
      ]} />
      {hist.length > 1 && (() => {
        const histX: [number, number] = [0, Math.max(...hist.map((h) => h.step), 1)];
        const histY: [number, number] = [Math.min(...ys) * 0.98, Math.max(...ys) * 1.02];
        return (
        <Section title="Held-out loss">
          <Plot xDomain={histX} yDomain={histY}
            series={[{ x: hist.map((h) => h.step), y: ys, color: C.comb, width: 2, dots: true, name: "held-out loss" }]}
            xLabel="fit step" yLabel="loss (1 = best member)" aspect={0.6} exportName={`${v.name}-history`} aria-label={`${v.name} held-out loss`} />
        </Section>
        );
      })()}
      <Section title="Fit meta" collapsible defaultOpen={false}><JsonTree data={v.fit} expandDepth={1} /></Section>
    </div>
  );
}
