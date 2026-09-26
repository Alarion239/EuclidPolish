/* Line and radial profiles across every visible tier. Shift-drag on a frame
 * draws a line; a click (with this panel open) or shift-click sets a radial
 * profile centre. The geometry lives on the tier it was drawn on and is
 * mapped onto every other tier through the sky (WCS) or by normalised
 * position, so LR, SR and JWST sample the same sky; distances are in arcsec
 * when every tier has a pixel scale. "per arcsec²" divides by the pixel area
 * (e⁻/px → e⁻/″²), which makes 0.1″ and 0.05″ tiers directly comparable.
 * Tiers in different units (e⁻ beside a JWST MJy/sr tier) are never drawn on
 * one y-axis: each unit gets its own plot, labelled with the band(s) sampled
 * and the unit. */
import { useMemo, useState } from "react";
import Plot from "../charts/Plot";
import { categorical } from "../colors";
import { linearTicks, paddedDomain } from "../ticks";
import { Button, Checkbox, Select } from "../ui";
import { useController, useSettings, useViewer } from "./hooks";
import { unitLabel } from "./readout";
import { lineProfile, radialProfile } from "./stats";

const RADII = [1, 2, 3, 5, 8];

export function ProfilePanel() {
  const ctrl = useController();
  const profile = useViewer((s) => s.profile);
  const shown = useViewer((s) => s.shown);
  const settings = useSettings();
  const [bandPick, setBandPick] = useState<string | null>(null);
  const [perArea, setPerArea] = useState(false);
  const [rmax, setRmax] = useState(3);
  const keys = ctrl.frameKeys().filter((k) => shown[k] && k !== "morph");
  const bandsAll = Array.from(new Set(keys.flatMap((k) => {
    const r = shown[k]!.rec;
    return r.bands.length === r.c ? r.bands : (ctrl.s.meta?.band_names.slice(0, r.c) ?? []);
  })));
  const band = bandPick && bandsAll.includes(bandPick) ? bandPick : bandsAll.includes(settings.color) ? settings.color : bandsAll[0];
  const allPix = keys.every((k) => (shown[k]!.rec.pixscale || 0) > 0);

  const series = useMemo(() => {
    if (!profile) return [];
    const out: { name: string; x: number[]; y: (number | null)[]; unit: string; band: string; color: string }[] = [];
    keys.forEach((k, i) => {
      const s = shown[k]!;
      const rec = s.rec;
      const names = rec.bands.length === rec.c ? rec.bands : (ctrl.s.meta?.band_names.slice(0, rec.c) ?? []);
      const b = Math.max(0, names.indexOf(band));
      // A tier without the chosen band (a one-filter JWST cube) samples its own channel.
      const sampled = names[b] ?? band;
      const ps = rec.pixscale > 0 ? rec.pixscale : 0;
      const area = perArea && ps ? ps * ps : 1;
      const unit = (s.kind === "cube" ? unitLabel(rec.unit || ctrl.tierMeta(k)?.unit || "") : unitLabel(rec.unit)) + (perArea && ps && s.kind === "cube" ? "/″²" : "");
      const scale = (v: number | null) => (v == null ? null : v / area);
      if (profile.kind === "line") {
        const a = ctrl.mapPoint(profile.tier, k, profile.p0.x, profile.p0.y);
        const c = ctrl.mapPoint(profile.tier, k, profile.p1.x, profile.p1.y);
        if (!a || !c) return;
        const p = lineProfile(rec, b, a, c);
        out.push({ name: ctrl.tierLabel(k), x: p.d.map((d) => (allPix ? d * ps : d)), y: p.v.map(scale), unit, band: sampled, color: categorical(i) });
      } else {
        const c = ctrl.mapPoint(profile.tier, k, profile.c.x, profile.c.y);
        if (!c) return;
        const r = allPix ? rmax / ps : rmax * 10;
        const p = radialProfile(rec, b, c, r, 1);
        out.push({ name: ctrl.tierLabel(k), x: p.r.map((d) => (allPix ? d * ps : d)), y: p.v.map(scale), unit, band: sampled, color: categorical(i) });
      }
    });
    return out;
  }, [ctrl, profile, keys, shown, band, perArea, allPix, rmax]);

  if (!profile) {
    return (
      <div className="cv-panel cv-profile">
        <p className="cv-panel__empty">Shift-drag on a frame for a line profile; click a point (or shift-click) for a radial profile.</p>
      </div>
    );
  }
  const xs = series.flatMap((s) => s.x);
  const xDomain: [number, number] = xs.length ? [Math.min(...xs), Math.max(...xs) || 1] : [0, 1];
  // One plot per unit (a shared x-axis range keeps them aligned).
  const groups: { unit: string; series: typeof series }[] = [];
  for (const s of series) {
    const g = groups.find((x) => x.unit === s.unit);
    if (g) g.series.push(s); else groups.push({ unit: s.unit, series: [s] });
  }
  const xLabel = profile.kind === "line" ? (allPix ? "distance along the line [″]" : "distance [px]") : (allPix ? "radius [″]" : "radius [px]");
  return (
    <div className="cv-panel cv-profile">
      <div className="cv-panel__head">
        <span className="cv-panel__title">{profile.kind === "line" ? "Line profile" : "Radial profile"}</span>
        {bandsAll.length > 1 && <Select value={band} onChange={setBandPick} aria-label="Profile band" options={bandsAll.map((b) => ({ value: b, label: b }))} />}
        {profile.kind === "radial" && (
          <Select value={String(rmax)} onChange={(v) => setRmax(Number(v))} aria-label="Radius"
            options={RADII.map((r) => ({ value: String(r), label: allPix ? `r ≤ ${r}″` : `r ≤ ${r * 10} px` }))} />
        )}
        {allPix && <Checkbox checked={perArea} onChange={setPerArea}>per arcsec²</Checkbox>}
        <Button size="sm" variant="ghost" onClick={() => ctrl.setProfile(null)}>clear</Button>
      </div>
      {groups.length > 1 && <p className="cv-panel__note">Tiers in different units are plotted on separate axes.</p>}
      {groups.length ? groups.map((g) => {
        const ys = g.series.flatMap((s) => s.y.filter((v): v is number => v != null && Number.isFinite(v)));
        const yDomain = paddedDomain(ys.length ? ys : [0, 1], { pad: 0.06 }) as [number, number];
        const bands = Array.from(new Set(g.series.map((s) => s.band)));
        const multi = groups.length > 1;
        return (
          <Plot key={g.unit || "none"} xDomain={xDomain} yDomain={yDomain} height={multi ? 180 : 220}
            xTicks={linearTicks(xDomain, { count: 6 })} yTicks={linearTicks(yDomain, { count: 5 })}
            xLabel={xLabel} yLabel={`${bands.join(" / ")}${g.unit ? ` [${g.unit}]` : ""}`}
            series={g.series.map((s) => ({ x: s.x, y: s.y, color: s.color, name: s.name, dots: s.x.length < 40 }))}
            legend="auto" exportName={`profile_${ctrl.collection}_${ctrl.s.index}${multi && g.unit ? `_${g.unit.replace(/[^\w]+/g, "")}` : ""}`}
            aria-label={multi ? `Profile · ${g.unit || "no unit"}` : "Profile"} />
        );
      }) : <p className="cv-panel__empty">The profile lies outside the loaded tiers.</p>}
    </div>
  );
}
