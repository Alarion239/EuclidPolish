/* Ensemble — the first fully-ported tab. Renders the power-spectrum r(k) figure
   (with the LR baseline) entirely on the frontend from /ensemble/evals.json,
   in the observatory-console language. The exemplar the other tabs follow. */
import { useEffect, useMemo, useState } from "react";
import { getJSON } from "../api";
import { useThemeValue } from "../theme";
import { C, LOSS_COLOR } from "../colors";
import Plot, { Legend, type Series, type Guide, type Tick } from "../charts/Plot";
import {
  Badge, Button, Card, CardBody, CardHead, Empty, Field, Segmented, Select,
  Spinner, Stat, Toolbar, ToolbarLabel, ToolbarSep,
} from "../ui";

type Mode = "starfull" | "starless";
type ColorBy = "uniform" | "loss";

type Member = { label: string; loss?: string; blocks?: number; psnr?: number | null };
type PS = {
  theta: (number | null)[];
  r: (number | null)[]; r_lr?: (number | null)[]; r_comb?: (number | null)[];
  r_members?: (number | null)[][]; r_pairs?: (number | null)[][]; r_cross?: (number | null)[];
};
type Evals = {
  ps: PS | null; guides?: { theta_min?: number; lr_scale?: number; vis_fwhm?: number };
  members?: Member[]; n_fields?: number; n_members?: number;
  combiner?: { available?: boolean; psnr?: number | null; ensemble_mean_psnr?: number | null } | null;
};

const XTICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5];
const hasData = (a?: (number | null)[]) => !!a && a.some((v) => v != null && isFinite(v as number));

export default function EnsemblePage() {
  const theme = useThemeValue();
  const [mode, setMode] = useState<Mode>("starfull");
  const [colorBy, setColorBy] = useState<ColorBy>("uniform");
  const [data, setData] = useState<Evals | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let live = true;
    setLoading(true);
    getJSON<Evals>(`/ensemble/evals.json?mode=${mode}`).then((d) => {
      if (live) { setData(d); setLoading(false); }
    });
    return () => { live = false; };
  }, [mode]);

  const ps = data?.ps ?? null;
  const members = data?.members ?? [];

  const chart = useMemo(() => {
    if (!ps || !hasData(ps.theta)) return null;
    const theta = ps.theta.map((v) => (v == null ? NaN : v)) as number[];
    const xs = theta.filter((v) => isFinite(v));
    const g = data?.guides ?? {};
    const xDomain: [number, number] = [g.theta_min ?? 0.05, Math.max(...xs)];
    const yDomain: [number, number] = [0, 1.05];
    const xTicks: Tick[] = XTICKS.filter((v) => v >= xDomain[0] && v <= xDomain[1])
      .map((v) => ({ v, label: String(v) }));
    const yTicks: Tick[] = [0, 0.25, 0.5, 0.75, 1].map((v) => ({ v, label: String(v) }));

    const memberColor = (i: number) =>
      colorBy === "loss" ? (LOSS_COLOR[members[i]?.loss ?? "l1"] ?? C.muted) : C.muted;

    const series: Series[] = [];
    // faint model–model pairs
    for (const pair of ps.r_pairs ?? [])
      series.push({ x: theta, y: pair, color: C.muted, width: 0.7, alpha: 0.18 });
    // per-member r(k)
    (ps.r_members ?? []).forEach((row, i) =>
      series.push({ x: theta, y: row, color: memberColor(i), width: 1, alpha: colorBy === "loss" ? 0.6 : 0.4 }));
    // model–model median (truth-free)
    if (hasData(ps.r_cross)) series.push({ x: theta, y: ps.r_cross!, color: C.cross, width: 2, dash: [6, 3] });
    // LR baseline (floor to beat)
    if (hasData(ps.r_lr)) series.push({ x: theta, y: ps.r_lr!, color: C.baseline, width: 2.5, dash: [7, 4] });
    // ensemble mean
    series.push({ x: theta, y: ps.r, color: C.mean, width: 2.6, dots: true });
    // combiner
    if (hasData(ps.r_comb)) series.push({ x: theta, y: ps.r_comb!, color: C.comb, width: 2.2, dots: true });

    const guides: Guide[] = [
      { axis: "y", v: 1, color: C.guide, dash: [2, 3] },
      { axis: "x", v: g.lr_scale ?? 0.1, color: C.guide, width: 1.3, dash: [6, 3] },
      { axis: "x", v: g.vis_fwhm ?? 0.16, color: C.visfwhm, alpha: 0.5, width: 1.5, dash: [5, 2] },
    ];

    const legend = [
      ...(hasData(ps.r_lr) ? [{ label: "LR baseline (no SR)", color: C.baseline, dash: true }] : []),
      { label: "ensemble mean", color: C.mean },
      ...(hasData(ps.r_comb) ? [{ label: "combiner", color: C.comb }] : []),
      { label: "individual models", color: C.muted },
      { label: "model–model r̃(k) (no HR)", color: C.cross, dash: true },
      { label: "LR sampling (0.1″)", color: C.guide, dash: true },
      { label: "VIS PSF FWHM", color: C.visfwhm, dash: true },
    ];
    return { series, guides, xDomain, yDomain, xTicks, yTicks, legend };
    // `theme` participates so the series/legend colors (read live from the
    // theme's CSS tokens via C/LOSS_COLOR) rebuild when the toggle flips.
  }, [ps, members, colorBy, data, theme]);

  const comb = data?.combiner;

  return (
    <div className="page">
      <header className="page__head">
        <div>
          <div className="eyebrow">model · ensemble</div>
          <h1 className="page__title">Ensemble evaluation</h1>
          <div className="page__sub">
            Cross-correlation r(k) of each reconstruction with HR — how much true
            structure survives at every angular scale. Higher is better; the LR
            baseline is the floor to beat.
          </div>
        </div>
        <div className="page__spacer" />
        <Segmented<Mode>
          value={mode}
          onChange={setMode}
          options={[{ value: "starfull", label: "starfull" }, { value: "starless", label: "starless" }]}
        />
      </header>

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <Card>
          <CardHead
            title="Power spectrum · r(k) vs HR"
            sub={data ? `${data.n_fields ?? 0} fields · ${data.n_members ?? 0} members · VIS` : "VIS band"}
            right={
              <Toolbar>
                <ToolbarLabel>color</ToolbarLabel>
                <Select<ColorBy>
                  value={colorBy}
                  onChange={setColorBy}
                  options={[{ value: "uniform", label: "uniform" }, { value: "loss", label: "by loss" }]}
                />
              </Toolbar>
            }
          />
          <CardBody>
            {loading ? (
              <Empty><Spinner /> loading…</Empty>
            ) : !chart ? (
              <Empty>
                No evaluation cached for <b>{mode}</b>. Run “Evaluate on test set” on the{" "}
                <a href="/ensemble">classic Ensemble page</a>, then return here.
              </Empty>
            ) : (
              <>
                <Plot
                  title="cross-correlation  r(k) vs HR   (1 = perfect)"
                  xScale="log"
                  xDomain={chart.xDomain}
                  yDomain={chart.yDomain}
                  xTicks={chart.xTicks}
                  yTicks={chart.yTicks}
                  xLabel="angular scale θ = 1/2k [arcsec]"
                  yLabel="r(k)  [VIS]"
                  series={chart.series}
                  guides={chart.guides}
                  aspect={0.46}
                />
                <Legend items={chart.legend} />
              </>
            )}
          </CardBody>
        </Card>

        {chart && (
          <div className="row" style={{ gap: "var(--s6)" }}>
            <Card style={{ flex: "1 1 160px" }}>
              <CardBody><Stat k="fields scored" v={data?.n_fields ?? "—"} /></CardBody>
            </Card>
            <Card style={{ flex: "1 1 160px" }}>
              <CardBody><Stat k="members" v={data?.n_members ?? "—"} /></CardBody>
            </Card>
            {comb?.available && (
              <Card style={{ flex: "1 1 220px" }}>
                <CardBody>
                  <div className="row" style={{ alignItems: "flex-end", gap: "var(--s4)" }}>
                    <Stat k="combiner PSNR" v={comb.psnr != null ? `${comb.psnr.toFixed(2)} dB` : "—"} />
                    {comb.psnr != null && comb.ensemble_mean_psnr != null && (
                      <Badge tone={comb.psnr >= comb.ensemble_mean_psnr ? "good" : "warn"}>
                        {comb.psnr >= comb.ensemble_mean_psnr ? "+" : ""}
                        {(comb.psnr - comb.ensemble_mean_psnr).toFixed(2)} vs mean
                      </Badge>
                    )}
                  </div>
                </CardBody>
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
