/* Models › Knees › By pixel class: one stacked bar per class (and per model
   overlaid), the segments the knees' shares in the knee colours, classes in
   the spec's order with their sample count; the gradient split (σ-weighted
   over every input band, or one input band's own split) or the ablation
   split (the caller's toggles), and how far the all-input gradient and the
   ablation agree. Synthetic truth classes and real brightness classes are
   separate panels (the caller renders one per set). */
import { formatCount } from "../../../format";
import { Caption } from "../../../ui";
import type { KneeModelResult } from "../kneesApi";
import {
  agreement, classOrder, classRows, kneeLabel, modelShort, sharePct, type ClassRow, type FieldSet, type KneeInput, type ShareKind,
} from "../knees";

export type DrawnModel = { spec: string; result: KneeModelResult };

const SET_TITLE: Record<FieldSet, string> = {
  synthetic: "Synthetic fields · truth classes",
  real: "Real fields · brightness classes",
};

/** A stacked bar of shares (each segment its knee's colour; hover names it). */
export function ShareStack({ shares, knees, color, label }: {
  shares: readonly number[]; knees: readonly number[]; color: (q: number) => string; label: string;
}) {
  const total = shares.reduce((s, v) => s + (v > 0 ? v : 0), 0);
  const text = shares.map((v, k) => `${kneeLabel(knees[k] ?? NaN)} ${sharePct(v)}`).join(", ");
  return (
    <span className="kn-stack" role="img" aria-label={`${label}: ${text}`}>
      {total > 0 && shares.map((v, k) => (v > 0 ? (
        <span key={k} className="kn-stack__seg" title={`${kneeLabel(knees[k] ?? NaN)}: ${sharePct(v, 1)}`}
          style={{ width: `${(100 * v) / total}%`, background: color(knees[k] ?? NaN) }} />
      ) : null))}
    </span>
  );
}

export function ClassBars({ set, models, band, kind, input = "all", color }: {
  set: FieldSet; models: readonly DrawnModel[]; band: string; kind: ShareKind; input?: KneeInput; color: (q: number) => string;
}) {
  const perModel = models.map((m) => ({ ...m, rows: classRows(m.result.aggregate[set], band, set, kind, input) }));
  const known = classOrder(set);
  const present = new Set(perModel.flatMap((m) => m.rows.map((r) => r.cls)));
  const classes = [...known.filter((c) => present.has(c)), ...[...present].filter((c) => !known.includes(c)).sort()];
  const many = models.length > 1;
  if (!classes.length) return null;
  return (
    <section className="kn-panel" aria-label={SET_TITLE[set]}>
      <h3 className="kn-panel__title">{SET_TITLE[set]}</h3>
      <div className="kn-classes" data-set={set}>
        <div className="kn-classes__head" aria-hidden="true">
          <span>Class</span><span />{many && <span />}<span>n</span><span title="gradient vs ablation agreement">agree</span>
        </div>
        {classes.map((cls) => (
          <div key={cls} className="kn-class" data-class={cls} data-many={many || undefined}>
            <span className="kn-class__name">{cls}</span>
            <div className="kn-class__bars">
              {perModel.map((m) => {
                const row: ClassRow | undefined = m.rows.find((r) => r.cls === cls);
                const who = modelShort(m.spec);
                if (!row) {
                  return (
                    <div key={m.spec} className="kn-barrow" data-many={many || undefined}>
                      {many && <span className="kn-barrow__who">{who}</span>}
                      <span className="kn-stack kn-stack--none">not sampled</span><span /><span />
                    </div>
                  );
                }
                const agree = agreement(row.gradient, row.ablation);
                return (
                  <div key={m.spec} className="kn-barrow" data-many={many || undefined}>
                    {many && <span className="kn-barrow__who">{who}</span>}
                    <ShareStack shares={row.shares} knees={m.result.knees} color={color} label={`${cls}${many ? `, ${who}` : ""}`} />
                    <span className="kn-barrow__n mdl-tnum">{formatCount(row.n)}</span>
                    <span className="kn-barrow__agree mdl-tnum" title="gradient vs ablation agreement">{agree == null ? "—" : sharePct(agree)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <Caption>
        {kind === "ablation"
          ? "Ablation: each knee's share of the output change when its input channels (in every input band) are replaced by their window mean, "
          : input === "all"
            ? "Gradient: each knee's share of the summed |∂y/∂x| through its input channels (the exact chain-rule split, each input "
              + "band's terms weighted by its noise σ), "
            : `Gradient, ${input} input: each knee's share of that input band's own |∂y/∂x| (its chain-rule terms alone), `}
        averaged over the sampled pixels of each class{set === "synthetic" ? " (truth classes)" : " (by the model's own SR brightness)"}.
        n: sampled pixels. Agree: 1 − ½ Σ|gradient − ablation|, both over all input bands (100% = the same split).
      </Caption>
    </section>
  );
}
