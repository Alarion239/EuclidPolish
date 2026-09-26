/* The viewer toolbar. Row 1 keeps the old controls: tier chips (multi-select),
 * layout, the BHR FWHM slider, JWST band chips, colour chips (Q–Y keys), the
 * knee + brightness sliders per transfer group, the movie sliders. Row 2 has
 * the inspection tools: pan / lens, zoom, compare (blink / swipe), residual
 * tiers, histogram and profile panels, stretch + colormap, link to the
 * Display panel. */
import { useState } from "react";
import { openDisplayPanel } from "../app/shellStore";
import { COLORMAPS, GAIN_RANGE, KNEE_RANGE, STRETCHES, useDisplay, type Colormap, type Stretch } from "../state/display";
import { Button, IconButton, Popover, Segmented, Select } from "../ui";
import { COLOR_KEYS, COLOR_MODES_EXTRA } from "./controller";
import { useController, useSettings, useViewer } from "./hooks";
import { VIcon } from "./icons";
import { RESIDUAL_OPS, parseResidualKey, type ResidualOp } from "./residual";
import type { ToolbarMode } from "./types";

function LogRange({ label, min, max, value, format, onChange }: {
  label: string; min: number; max: number; value: number; format: (v: number) => string; onChange: (v: number) => void;
}) {
  const to = (v: number) => Math.round((1000 * (Math.log(v) - Math.log(min))) / (Math.log(max) - Math.log(min)));
  const from = (p: number) => Math.exp(Math.log(min) + (p / 1000) * (Math.log(max) - Math.log(min)));
  return (
    <label className="cv-slider">
      <span>{label}</span>
      <input type="range" className="cv-range" min={0} max={1000} value={to(Math.min(max, Math.max(min, value)))}
        onChange={(e) => onChange(from(Number(e.target.value)))} aria-label={label} aria-valuetext={format(value)} />
      <span className="cv-val">{format(value)}</span>
    </label>
  );
}

function LinearRange({ label, min, max, step, value, format, onChange }: {
  label: string; min: number; max: number; step: number; value: number; format: (v: number) => string; onChange: (v: number) => void;
}) {
  return (
    <label className="cv-slider">
      <span>{label}</span>
      <input type="range" className="cv-range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} aria-label={label} aria-valuetext={format(value)} />
      <span className="cv-val">{format(value)}</span>
    </label>
  );
}

function Chip({ active, disabled, title, onClick, children }: {
  active: boolean; disabled?: boolean; title?: string; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button type="button" className={`cv-chip${active ? " active" : ""}${disabled ? " cv-disabled" : ""}`}
      aria-pressed={active} aria-disabled={disabled || undefined} title={title} onClick={onClick}>{children}</button>
  );
}

const GROUP_LABEL: Record<string, string> = { jwst: "JWST", euclid: "Euclid", default: "" };
const STRETCH_LABEL: Record<Stretch, string> = {
  "asinh-abs": "asinh (absolute)", linear: "linear", log: "log", sqrt: "sqrt", "asinh-auto": "asinh (auto)", zscale: "zscale",
};
const OP_LABEL: Record<ResidualOp, string> = { diff: "A − B", ratio: "log₂ A/B", chi: "(A − B)/σ" };

function ResidualPicker() {
  const ctrl = useController();
  const meta = useViewer((s) => s.meta);
  const residuals = useViewer((s) => s.residuals);
  const candidates = (meta?.tiers ?? []).filter((t) => !t.disabled && t.key !== "morph" && !/^pca\d+$/.test(t.key));
  const [a, setA] = useState(candidates.find((t) => t.key.toLowerCase() === "sr")?.key ?? candidates[0]?.key ?? "");
  const [b, setB] = useState(candidates.find((t) => ["hr", "lr", "mean"].includes(t.key.toLowerCase()) && t.key.toLowerCase() !== a.toLowerCase())?.key ?? candidates[1]?.key ?? "");
  const [op, setOp] = useState<ResidualOp>("diff");
  const options = candidates.map((t) => ({ value: t.key, label: t.label }));
  return (
    <div className="cv-respick">
      <div className="cv-respick__row">
        <Select value={a} onChange={setA} options={options} aria-label="Tier A" />
        <Segmented<ResidualOp> size="sm" value={op} onChange={setOp} aria-label="Residual"
          options={RESIDUAL_OPS.map((o) => ({ value: o, label: OP_LABEL[o] }))} />
        <Select value={b} onChange={setB} options={options} aria-label="Tier B" />
      </div>
      <p className="cv-respick__hint">Computed in the browser; a coarser tier is resampled onto the finer grid (flux per pixel conserved). σ is the std tier when there is one, else the robust σ of the difference.</p>
      <div className="cv-respick__row">
        <Button size="sm" variant="primary" disabled={!a || !b || a === b} onClick={() => ctrl.addResidual(op, a, b)}>Add residual tier</Button>
      </div>
      {residuals.length > 0 && (
        <ul className="cv-respick__list">
          {residuals.map((k) => {
            const r = parseResidualKey(k);
            return (
              <li key={k}>
                <span className="mono">{ctrl.tierLabel(k)}</span>
                {r && <Button size="sm" variant="ghost" onClick={() => ctrl.removeResidual(k)}>remove</Button>}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

export function Toolbar({ mode }: { mode: ToolbarMode }) {
  const ctrl = useController();
  const meta = useViewer((s) => s.meta);
  const tiers = useViewer((s) => s.tiers);
  const params = useViewer((s) => s.params);
  const layout = useViewer((s) => s.layout);
  const tool = useViewer((s) => s.tool);
  const view = useViewer((s) => s.view);
  const compare = useViewer((s) => s.compare);
  const histogram = useViewer((s) => s.histogram);
  const profileOpen = useViewer((s) => s.profileOpen);
  const unlinked = useViewer((s) => s.unlinked);
  const morphAmp = useViewer((s) => s.morphAmp);
  const morphSpeed = useViewer((s) => s.morphSpeed);
  const residuals = useViewer((s) => s.residuals);
  const blinkMs = useViewer((s) => s.blinkMs);
  const globalLinked = useDisplay((s) => s.linked);
  const settings = useSettings();
  if (!meta || mode === "none") return null;

  const shownTiers = (meta.tiers || []).filter((t) => !t.hidden);
  const logMode = meta.render_mode === "log";
  const frames = ctrl.frameKeys();
  const groups = ctrl.activeGroups();
  const bhr = meta.bhr_fwhm_control;
  const jwstOptions = meta.jwst_band_options || [];
  let colorIndex = 0;
  const colorChips = logMode ? [] : [
    ...meta.band_names.map((n) => ({ key: n, label: n, title: n })),
    ...COLOR_MODES_EXTRA.map((m) => ({ key: m.key, label: m.label, title: m.title })),
  ].map((c) => ({ ...c, shortcut: COLOR_KEYS[colorIndex++]?.toUpperCase() }));
  const colourKnown = colorChips.some((c) => c.key === settings.color);

  const tools = (
    <div className="cv-tools" role="toolbar" aria-label="Viewer tools">
      <Segmented<"pan" | "lens"> size="sm" value={tool} onChange={(t) => ctrl.setTool(t)} aria-label="Pointer tool"
        options={[
          { value: "pan", label: <span className="cv-tool"><VIcon name="pan" />pan</span>, title: "Drag to pan once zoomed, wheel to zoom (focused or ⌘/Ctrl) · hold Alt for the lens" },
          { value: "lens", label: <span className="cv-tool"><VIcon name="lens" />lens</span>, title: "Magnifier lens: wheel zooms it, click freezes the matched crop (L)" },
        ]} />
      <IconButton size="sm" label="Zoom out (−)" icon="zoomOut" onClick={() => frames[0] && ctrl.zoomView(frames[0], 1 / 1.5)} disabled={!view} />
      <IconButton size="sm" label="Zoom in (+)" icon="zoomIn" onClick={() => frames[0] && ctrl.zoomView(frames[0], 1.5)} />
      <IconButton size="sm" label="Fit: show the whole image (0, double-click)" icon={<VIcon name="fit" />} onClick={() => ctrl.resetView()} disabled={!view} />
      {mode === "full" && <>
        <span className="cv-sep" aria-hidden="true" />
        <Segmented<"off" | "blink" | "swipe"> size="sm" value={compare} onChange={(c) => ctrl.setCompare(c)} aria-label="Compare"
          disabled={frames.length < 2}
          options={[
            { value: "off", label: "side by side" },
            { value: "blink", label: <span className="cv-tool"><VIcon name="blink" />blink</span>, title: "Cycle the selected tiers in one frame (B)" },
            { value: "swipe", label: <span className="cv-tool"><VIcon name="swipe" />swipe</span>, title: "Reveal the second tier over the first with a divider" },
          ]} />
        {compare === "blink" && (
          <LinearRange label="blink" min={0.2} max={2} step={0.1} value={blinkMs / 1000}
            format={(v) => `${v.toFixed(1)} s`} onChange={(v) => ctrl.setBlinkMs(v * 1000)} />
        )}
        <Popover label="Residual tiers" width={420} trigger={
          <Button size="sm" variant={residuals.length ? "subtle" : "ghost"} icon={<VIcon name="residual" />}>
            residual{residuals.length ? ` (${residuals.length})` : ""}
          </Button>}>
          <ResidualPicker />
        </Popover>
        <IconButton size="sm" label="Histogram of the visible region" icon={<VIcon name="histogram" />} pressed={histogram}
          onClick={() => ctrl.setPanels({ histogram: !histogram })} />
        <IconButton size="sm" label="Profiles: shift-drag a line, click a point for a radial profile" icon={<VIcon name="profile" />}
          pressed={profileOpen} onClick={() => { ctrl.setPanels({ profileOpen: !profileOpen }); if (profileOpen) ctrl.setProfile(null); }} />
        <span className="cv-sep" aria-hidden="true" />
        <Select<Stretch> value={settings.stretch} aria-label="Stretch"
          onChange={(stretch) => (ctrl.linked() && !("stretch" in ctrl.s.override) ? useDisplay.getState().set({ stretch }) : ctrl.setOverride({ stretch }))}
          options={STRETCHES.map((s) => ({ value: s, label: STRETCH_LABEL[s] }))} />
        <Select<Colormap> value={settings.colormap} aria-label="Colormap"
          onChange={(colormap) => (ctrl.linked() && !("colormap" in ctrl.s.override) ? useDisplay.getState().set({ colormap }) : ctrl.setOverride({ colormap }))}
          options={COLORMAPS.map((c) => ({ value: c, label: c }))} />
        <IconButton size="sm" label={!globalLinked ? "Display panel: viewers are unlinked" : unlinked ? "Unlinked from the Display panel (click to follow it again)" : "Following the Display panel (click to give this viewer its own settings)"}
          icon={<VIcon name={unlinked || !globalLinked ? "unlink" : "link"} />} pressed={!unlinked && globalLinked}
          disabled={!globalLinked} onClick={() => ctrl.setUnlinked(!unlinked)} />
        <IconButton size="sm" label="Display panel (Shift+D)" icon={<VIcon name="palette" />} onClick={() => openDisplayPanel()} />
      </>}
    </div>
  );

  return (
    <div className={`cv-toolbar${mode === "compact" ? " cv-toolbar--compact" : ""}`}>
      <div className="cv-toolbar__row">
        {shownTiers.length > 1 && <>
          <span className="cv-grouplabel">Tier</span>
          <div className="cv-group">
            {shownTiers.map((t) => (
              <Chip key={t.key} active={tiers.includes(t.key)} disabled={ctrl.tierDisabled(t.key)}
                title="toggle — select more than one to compare side by side" onClick={() => ctrl.toggleTier(t.key)}>{t.label}</Chip>
            ))}
          </div>
        </>}
        {mode === "full" && frames.length >= 3 && <>
          <span className="cv-grouplabel">Layout</span>
          <div className="cv-group">
            <Chip active={layout === "one-row"} title="place all selected images side by side" onClick={() => ctrl.setLayout("one-row")}>one row</Chip>
            <Chip active={layout === "two-rows"} title="place selected images across two rows" onClick={() => ctrl.setLayout("two-rows")}>two rows</Chip>
          </div>
        </>}
        {bhr && mode === "full" && <>
          <span className="cv-grouplabel">Target PSF</span>
          <div className="cv-group cv-sliders">
            <LinearRange label="BHR FWHM" min={Number(bhr.min_arcsec)} max={Number(bhr.max_arcsec)} step={Number(bhr.step_arcsec)}
              value={Number.isFinite(Number(params[bhr.param || "bhr_fwhm_arcsec"])) ? Number(params[bhr.param || "bhr_fwhm_arcsec"]) : Number(bhr.default_arcsec)}
              format={(v) => `${v.toFixed(3)}″`} onChange={(v) => ctrl.setBhrFwhm(v)} />
          </div>
        </>}
        {jwstOptions.length > 1 && <>
          <span className="cv-grouplabel">JWST band</span>
          <div className="cv-group">
            {jwstOptions.map((o) => (
              <Chip key={o.value} active={params.jwst_band === o.value} disabled={!ctrl.jwstBandAvailable(o.value)}
                title={o.value === "colour" ? "all available filters as display colour" : `show native ${o.label}`}
                onClick={() => { if (ctrl.jwstBandAvailable(o.value)) ctrl.setParam("jwst_band", o.value); }}>{o.label}</Chip>
            ))}
          </div>
        </>}
        {!logMode && <>
          <span className="cv-grouplabel">{meta.color_label || "Colour"}</span>
          <div className="cv-group">
            {colorChips.map((c) => (
              <Chip key={c.key} active={settings.color === c.key} title={c.shortcut ? `${c.shortcut} — ${c.title}` : c.title}
                onClick={() => ctrl.setColor(c.key)}>{c.label}</Chip>
            ))}
            {!colourKnown && <span className="cv-chip active" title="set in the Display panel">{settings.color}</span>}
          </div>
        </>}
        <div className="cv-group cv-sliders">
          {groups.map((g) => {
            const t = ctrl.transfer(g, settings);
            const prefix = groups.length > 1 ? `${GROUP_LABEL[g] ?? g} ` : "";
            return (
              <span key={g} className="cv-sliderpair">
                {!logMode && <LogRange label={`${prefix}asinh knee`} min={KNEE_RANGE[0]} max={KNEE_RANGE[1]} value={t.knee}
                  format={(v) => `${Math.round(v)} e⁻`} onChange={(knee) => ctrl.setTransfer(g, { knee })} />}
                <LogRange label={`${prefix}brightness`} min={GAIN_RANGE[0]} max={GAIN_RANGE[1]} value={t.gain}
                  format={(v) => `${v.toFixed(2)}×`} onChange={(gain) => ctrl.setTransfer(g, { gain })} />
              </span>
            );
          })}
        </div>
        {tiers.includes("morph") && (
          <div className="cv-group cv-sliders cv-morph-ctl">
            <LogRange label="morph amplitude" min={0.05} max={3} value={morphAmp} format={(v) => `${v.toFixed(2)}σ`}
              onChange={(v) => ctrl.setPanels({ morphAmp: v })} />
            <LogRange label="morph speed" min={0.1} max={2} value={morphSpeed} format={(v) => `${v.toFixed(2)}×`}
              onChange={(v) => ctrl.setPanels({ morphSpeed: v })} />
          </div>
        )}
      </div>
      <div className="cv-toolbar__row cv-toolbar__row--tools">{tools}</div>
    </div>
  );
}
