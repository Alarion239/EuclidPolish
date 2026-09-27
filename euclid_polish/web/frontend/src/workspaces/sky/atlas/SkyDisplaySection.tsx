/* The Display panel's "Sky" section (registered while the atlas is mounted):
 * colour of the current background HiPS (colormap, stretch, cuts, reverse
 * for FITS tiles; gamma, saturation, brightness, contrast for all), the
 * FITS pixel overlays' stretch, the coordinate grid and the marker size. */
import { useEffect, useState } from "react";
import { useUrlState } from "../../../hooks/useUrlState";
import { ALADIN_COLORMAPS, ALADIN_STRETCHES, DEFAULT_BASE, baseSurvey } from "../../../sky/surveys";
import { Button, Field, NumberField, Select, Slider, Switch } from "../../../ui";
import { useSkyDisplay } from "./store";

const cmapOptions = ALADIN_COLORMAPS.map((c) => ({ value: c, label: c }));
const stretchOptions = ALADIN_STRETCHES.map((s) => ({ value: s, label: s }));
const fmt2 = (v: number) => String(Number(v.toFixed(2)));

/** A pair of cut fields committed together (only a valid min < max applies). */
function Cuts({ min, max, onCommit, unit }: { min?: number | null; max?: number | null; unit?: string; onCommit: (lo: number | null, hi: number | null) => void }) {
  const [lo, setLo] = useState(min != null ? String(min) : "");
  const [hi, setHi] = useState(max != null ? String(max) : "");
  useEffect(() => { setLo(min != null ? String(min) : ""); setHi(max != null ? String(max) : ""); }, [min, max]);
  const commit = (a: string, b: string) => {
    const x = Number(a), y = Number(b);
    if (a.trim() === "" && b.trim() === "") onCommit(null, null);
    else if (a.trim() !== "" && b.trim() !== "" && Number.isFinite(x) && Number.isFinite(y) && x < y) onCommit(x, y);
  };
  return (
    <div className="sky-display__pair">
      <NumberField label="Min cut" value={lo} step="any" unit={unit} onChange={(v) => { setLo(v); commit(v, hi); }} />
      <NumberField label="Max cut" value={hi} step="any" unit={unit} onChange={(v) => { setHi(v); commit(lo, v); }} />
    </div>
  );
}

export function SkyDisplaySection() {
  const [baseId] = useUrlState("base", DEFAULT_BASE);
  const survey = baseSurvey(baseId);
  const d = useSkyDisplay();
  const own = d.byBase[survey.id] ?? {};
  const color = { ...survey.color, ...own };
  const fits = survey.format === "fits";
  const setBase = (patch: Parameters<typeof d.setBase>[1]) => d.setBase(survey.id, patch);
  return (
    <div className="sky-display">
      <div className="sky-display__head">
        <strong>{survey.label}</strong>
        <Button size="sm" variant="ghost" disabled={!Object.keys(own).length} onClick={() => d.resetBase(survey.id)}>Reset</Button>
      </div>
      {survey.url == null ? <p className="muted">No background imagery (black sky).</p> : (
        <div className="sky-display__grid">
          {fits && (
            <>
              <Field label="Colormap"><Select value={color.colormap ?? "grayscale"} options={cmapOptions} onChange={(v) => setBase({ colormap: v })} /></Field>
              <Field label="Stretch"><Select value={color.stretch ?? "linear"} options={stretchOptions} onChange={(v) => setBase({ stretch: v })} /></Field>
              <Cuts min={color.minCut} max={color.maxCut} onCommit={(lo, hi) => setBase({ minCut: lo ?? undefined, maxCut: hi ?? undefined })} />
              <Switch checked={!!color.reversed} onChange={(v) => setBase({ reversed: v })}>Reverse colormap</Switch>
            </>
          )}
          <Field label="Gamma"><Slider value={color.gamma ?? 1} min={0.1} max={10} scale="log" showValue format={fmt2} onChange={(v) => setBase({ gamma: v })} aria-label="Gamma" /></Field>
          <Field label="Saturation"><Slider value={color.saturation ?? 0} min={-1} max={1} step={0.05} showValue format={fmt2} onChange={(v) => setBase({ saturation: v })} aria-label="Saturation" /></Field>
          <Field label="Brightness"><Slider value={color.brightness ?? 0} min={-1} max={1} step={0.05} showValue format={fmt2} onChange={(v) => setBase({ brightness: v })} aria-label="Brightness" /></Field>
          <Field label="Contrast"><Slider value={color.contrast ?? 0} min={-1} max={1} step={0.05} showValue format={fmt2} onChange={(v) => setBase({ contrast: v })} aria-label="Contrast" /></Field>
        </div>
      )}
      <div className="sky-display__head"><strong>Pixel overlays</strong><span className="muted">LR / SR / JWST FITS</span></div>
      <div className="sky-display__grid">
        <Switch checked={d.overlay.follow} onChange={(v) => d.set({ overlay: { ...d.overlay, follow: v } })}>
          Follow the viewer colours
        </Switch>
        {!d.overlay.follow && (
          <>
            <Field label="Colormap"><Select value={d.overlay.colormap} options={cmapOptions} onChange={(v) => d.set({ overlay: { ...d.overlay, colormap: v } })} /></Field>
            <Field label="Stretch"><Select value={d.overlay.stretch} options={stretchOptions} onChange={(v) => d.set({ overlay: { ...d.overlay, stretch: v } })} /></Field>
            <Cuts min={d.overlay.minCut} max={d.overlay.maxCut} unit="e⁻"
              onCommit={(lo, hi) => d.set({ overlay: { ...d.overlay, minCut: lo, maxCut: hi } })} />
          </>
        )}
      </div>
      <div className="sky-display__grid">
        <Switch checked={d.grid} onChange={(v) => d.set({ grid: v })}>Coordinate grid</Switch>
        <Field label="Marker size"><Slider value={d.markerScale} min={0.5} max={3} step={0.1} showValue format={(v) => `×${fmt2(v)}`} onChange={(v) => d.set({ markerScale: v })} aria-label="Marker size" /></Field>
      </div>
    </div>
  );
}
