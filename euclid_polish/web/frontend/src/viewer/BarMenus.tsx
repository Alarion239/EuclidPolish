/* The popover panels of the viewer's control bar (Bar.tsx):
 *
 *   TierMenu     every tier as a checkbox, with its extras on its entry
 *                (BHR target FWHM, JWST band, movie amplitude / speed)
 *   DisplayDock  (docked beside the frames, not a popover: the image stays
 *                in sight) stretch, knee / brightness / black point per
 *                transfer group (in the group's display unit), colormaps,
 *                invert, NaN colour, the histogram (behind a disclosure) and
 *                the Display-panel link; `basic` = knee + brightness only
 *   ToolsMenu    pan / lens, the profile panel, residual tiers, playback
 *
 * Every edit goes through the controller (linked → the Display panel store,
 * overridden / unlinked → this viewer), as the old toolbar did. */
import { useState, type ReactNode } from "react";
import { openDisplayPanel } from "../app/shellStore";
import { COLORMAPS, STRETCHES, useDisplay, type Colormap, type Stretch } from "../state/display";
import { Button, Checkbox, IconButton, Kbd, Segmented, Select, Slider, Switch } from "../ui";
import { GAIN_SLIDER_RANGE, KNEE_SLIDER_RANGE, formatSig, groupUnit, parseNumber, sentenceLabel } from "./barModel";
import { HistogramPanel } from "./HistogramPanel";
import { useController, useSettings, useViewer } from "./hooks";
import { VIcon } from "./icons";
import { RESIDUAL_OPS, parseResidualKey, type ResidualOp } from "./residual";
import type { TierMeta } from "./types";

const GROUP_LABEL: Record<string, string> = { jwst: "JWST", euclid: "Euclid", default: "" };
export const STRETCH_LABEL: Record<Stretch, string> = {
  "asinh-abs": "Asinh (absolute)", linear: "Linear", log: "Log", sqrt: "Square root", "asinh-auto": "Asinh (auto)", zscale: "Zscale",
};
/** The Display panel's wording (app/DisplayPanel.tsx), kept identical. */
const CMAP_LABEL: Record<Colormap, string> = {
  gray: "Gray", viridis: "Viridis", magma: "Magma", inferno: "Inferno", cividis: "Cividis", rdbu: "Red–blue (diverging)",
};
const OP_LABEL: Record<ResidualOp, string> = { diff: "A − B", ratio: "log₂ A/B", chi: "(A − B)/σ" };
/** Stretches anchored at a black point (the knee only shapes asinh-abs). */
const ANCHORED = new Set<Stretch>(["asinh-abs", "linear", "sqrt", "log"]);

function Row({ label, children, hint, inline }: { label: string; children: ReactNode; hint?: ReactNode; inline?: boolean }) {
  return (
    <div className="cv-menu__row" data-inline={inline || undefined}>
      <span className="cv-menu__label">{label}</span>
      <span className="cv-menu__ctl">{children}</span>
      {hint && <span className="cv-menu__hint">{hint}</span>}
    </div>
  );
}

/** A log slider plus a typed value, shown in `unit` (native = transfer ÷ scale). */
function ValueSlider({ label, value, min, max, scale = 1, unit, onChange, suffix }: {
  label: string; value: number; min: number; max: number; scale?: number; unit: string;
  onChange: (v: number) => void; suffix?: string;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const shown = formatSig(value / scale);
  const commit = () => {
    const v = parseNumber(draft ?? "");
    setDraft(null);
    if (v != null && v > 0) onChange(v * scale);
  };
  return (
    <span className="cv-valslider">
      <Slider value={value} min={min} max={max} scale="log" aria-label={label} onChange={onChange}
        format={(v) => `${formatSig(v / scale)}${unit ? ` ${unit}` : ""}${suffix ?? ""}`} />
      <input className="cv-num" type="text" inputMode="decimal" aria-label={`${label}${unit ? ` (${unit})` : ""}`}
        value={draft ?? shown} onChange={(e) => setDraft(e.target.value)} onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter") { commit(); (e.target as HTMLInputElement).blur(); } if (e.key === "Escape") setDraft(null); }} />
      <span className="cv-menu__unit">{unit}{suffix}</span>
    </span>
  );
}

// ---- tiers --------------------------------------------------------------------------

function TierExtras({ tier }: { tier: TierMeta | null }) {
  const ctrl = useController();
  const meta = useViewer((s) => s.meta);
  const params = useViewer((s) => s.params);
  const morphAmp = useViewer((s) => s.morphAmp);
  const morphSpeed = useViewer((s) => s.morphSpeed);
  if (!meta) return null;
  const key = tier?.key.toLowerCase() ?? null;
  const keys = (meta.tiers ?? []).map((t) => t.key.toLowerCase());
  const bhr = meta.bhr_fwhm_control;
  const jwst = meta.jwst_band_options ?? [];
  // An extra sits on its tier's entry; without that tier it heads the list (tier = null).
  const showBhr = !!bhr && (key === "bhr" || (tier === null && !keys.includes("bhr")));
  const showJwst = jwst.length > 1 && (key === "jwst" || (tier === null && !keys.includes("jwst")));
  const showMorph = key === "morph";
  if (!showBhr && !showJwst && !showMorph) return null;
  const bhrParam = bhr?.param || "bhr_fwhm_arcsec";
  const bhrValue = Number.isFinite(Number(params[bhrParam])) ? Number(params[bhrParam]) : Number(bhr?.default_arcsec);
  return (
    <div className="cv-menu__extras">
      {showBhr && bhr && (
        <Row label="Target PSF FWHM">
          <Slider value={bhrValue} min={Number(bhr.min_arcsec)} max={Number(bhr.max_arcsec)} step={Number(bhr.step_arcsec)}
            aria-label="BHR target FWHM" format={(v) => `${v.toFixed(3)}″`} showValue onChange={(v) => ctrl.setBhrFwhm(v)} />
        </Row>
      )}
      {showJwst && (
        <Row label="JWST band">
          <Segmented size="sm" value={params.jwst_band ?? jwst[0].value} aria-label="JWST band" className="cv-seg-wrap"
            onChange={(v) => { if (ctrl.jwstBandAvailable(v)) ctrl.setParam("jwst_band", v); }}
            options={jwst.map((o) => ({
              value: o.value, label: o.label, disabled: !ctrl.jwstBandAvailable(o.value),
              title: o.value === "colour" ? "All available filters as a colour image" : `Native ${o.label}`,
            }))} />
        </Row>
      )}
      {showMorph && <>
        <Row label="Movie amplitude">
          <Slider value={morphAmp} min={0.05} max={3} scale="log" aria-label="Movie amplitude" showValue
            format={(v) => `${v.toFixed(2)}σ`} onChange={(v) => ctrl.setPanels({ morphAmp: v })} />
        </Row>
        <Row label="Movie speed">
          <Slider value={morphSpeed} min={0.1} max={2} scale="log" aria-label="Movie speed" showValue
            format={(v) => `${v.toFixed(2)}×`} onChange={(v) => ctrl.setPanels({ morphSpeed: v })} />
        </Row>
      </>}
    </div>
  );
}

function TierItem({ t }: { t: TierMeta }) {
  const ctrl = useController();
  const tiers = useViewer((s) => s.tiers);
  useViewer((s) => s.index);   // availability follows the object
  const on = tiers.includes(t.key);
  const disabled = !on && ctrl.tierDisabled(t.key);
  return (
    <li className="cv-menu__tier">
      <Checkbox checked={on} disabled={disabled} onChange={() => ctrl.toggleTier(t.key)}>{sentenceLabel(t.label)}</Checkbox>
      {disabled && <span className="cv-menu__hint">{ctrl.missingTierLabel(t.key)}</span>}
      <TierExtras tier={t} />
    </li>
  );
}

export function TierMenu() {
  const meta = useViewer((s) => s.meta);
  if (!meta) return null;
  const all = meta.tiers ?? [];
  const shown = all.filter((t) => !t.hidden);
  const hidden = all.filter((t) => t.hidden);
  return (
    <div className="cv-menu">
      <p className="cv-menu__title">Tiers</p>
      <TierExtras tier={null} />
      <ul className="cv-menu__list">{shown.map((t) => <TierItem key={t.key} t={t} />)}</ul>
      {hidden.length > 0 && <>
        <p className="cv-menu__title">More tiers</p>
        <ul className="cv-menu__list cv-menu__list--scroll">{hidden.map((t) => <TierItem key={t.key} t={t} />)}</ul>
      </>}
      <p className="cv-menu__note">Select several tiers to compare them side by side.</p>
    </div>
  );
}

// ---- display ------------------------------------------------------------------------

/** The Display dock: docked BESIDE the frames on the light table (below
 *  them in a narrow viewer), so the image stays in sight while the knee,
 *  brightness or stretch change. In the order they are used: per transfer
 *  group the knee and brightness (+ reset), the stretch, "Match surface
 *  brightness" (when the shown tiers have different pixel scales, area.ts);
 *  the rest (black point, colormaps, invert, NaN colour, the histogram)
 *  behind "More settings"; last, sharing the settings with every viewer and
 *  the Display panel. `basic` (compact viewers): the knee, brightness and
 *  the sharing switch only. */
export function DisplayDock({ basic = false }: { basic?: boolean }) {
  const ctrl = useController();
  const meta = useViewer((s) => s.meta);
  const shown = useViewer((s) => s.shown);
  const unlinked = useViewer((s) => s.unlinked);
  const perArea = useViewer((s) => s.perArea);
  const areaRef = useViewer((s) => ctrl.areaRef(s));
  const globalLinked = useDisplay((s) => s.linked);
  const settings = useSettings();
  const [moreOpen, setMoreOpen] = useState(false);
  const [histOpen, setHistOpen] = useState(false);
  if (!meta) return null;
  const logMode = meta.render_mode === "log";
  const groups = ctrl.activeGroups();
  const recs = Object.values(shown).map((s) => s.rec);
  const fallbackUnit = meta.tiers?.find((t) => t.unit)?.unit ?? "";
  const K0 = ctrl.K0();
  const following = globalLinked && !unlinked;
  const perAreaOn = perArea && areaRef > 0;
  const unitOf = (g: string) => {
    const u = groupUnit(recs, (r) => ctrl.groupOf(r as never) === g, g === "jwst" ? "" : fallbackUnit);
    return u;
  };
  const titleOf = (g: string) => (groups.length > 1 ? `${GROUP_LABEL[g] ?? g}` : "");
  return (
    <aside className="cv-dock" aria-label="Display settings for this viewer" data-basic={basic || undefined}>
      <div className="cv-dock__scroll">
        <div className="cv-menu cv-menu--display">
          <div className="cv-menu__head">
            <h3 className="cv-menu__title">Display</h3>
            <IconButton size="sm" className="cv-ib" icon={<VIcon name="close" />} label="Close the Display settings" tooltip={<span className="cv-tip">Close <Kbd keys="Escape" /></span>}
              onClick={() => ctrl.setDock(false)} />
          </div>
          {groups.map((g) => {
            const t = ctrl.transfer(g, settings);
            const u = unitOf(g);
            const title = titleOf(g);
            return (
              <fieldset key={g} className="cv-menu__group">
                {title && <legend className="cv-menu__legend">{title}</legend>}
                {!logMode && settings.stretch === "asinh-abs" && (
                  <Row label="Knee">
                    <ValueSlider label={`${title ? `${title} ` : ""}knee`} value={t.knee} min={KNEE_SLIDER_RANGE[0]} max={KNEE_SLIDER_RANGE[1]}
                      scale={u.scale} unit={u.unit} onChange={(knee) => ctrl.setTransfer(g, { knee: Math.min(1e12, knee) })} />
                  </Row>
                )}
                <Row label="Brightness">
                  <ValueSlider label={`${title ? `${title} ` : ""}brightness`} value={t.gain} min={GAIN_SLIDER_RANGE[0]} max={GAIN_SLIDER_RANGE[1]}
                    unit="" suffix="×" onChange={(gain) => ctrl.setTransfer(g, { gain })} />
                </Row>
                <div className="cv-menu__actions">
                  <Button size="sm" variant="ghost" onClick={() => ctrl.setTransfer(g, { knee: K0, gain: 1, black: 0 })}>
                    Reset{title ? ` ${title}` : ""} to the default
                  </Button>
                </div>
              </fieldset>
            );
          })}
          {!basic && (
            <Row label="Stretch">
              <Select<Stretch> size="sm" value={settings.stretch} aria-label="Stretch"
                onChange={(stretch) => ctrl.setDisplay({ stretch })}
                options={STRETCHES.map((s) => ({ value: s, label: STRETCH_LABEL[s] }))} />
            </Row>
          )}
          {!basic && !logMode && settings.stretch !== "asinh-abs" && (
            <p className="cv-menu__note">The knee shapes the absolute asinh stretch only.</p>
          )}
          {!logMode && areaRef > 0 && (
            <div className="cv-menu__tier">
              <Switch checked={perArea} onChange={(on) => ctrl.setPerArea(on)}>Match surface brightness</Switch>
              <span className="cv-menu__hint cv-menu__hint--flush">
                {perAreaOn
                  ? `Every frame is shown per ${areaRef}″ pixel, so the same sky looks equally bright on every grid; the knee reads e⁻ per ${areaRef}″ pixel. Pixel values stay native.`
                  : "Off: each frame in e⁻ per its own pixel (finer grids look dimmer)."}
              </span>
            </div>
          )}
          {!basic && <>
            <button type="button" className="cv-disclosure" aria-expanded={moreOpen} onClick={() => setMoreOpen((o) => !o)}>
              <VIcon name="chevron" /> More settings
            </button>
            {moreOpen && <>
              {!logMode && ANCHORED.has(settings.stretch) && groups.map((g) => {
                const t = ctrl.transfer(g, settings);
                const u = unitOf(g);
                const title = titleOf(g);
                return (
                  <Row key={g} label={`${title ? `${title} b` : "B"}lack point`}>
                    <BlackPoint value={t.black} scale={u.scale} unit={u.unit} label={`${title ? `${title} ` : ""}black point`}
                      onChange={(black) => ctrl.setTransfer(g, { black })} />
                  </Row>
                );
              })}
              <Row label="Colormap">
                <Select<Colormap> size="sm" value={settings.colormap} aria-label="Colormap"
                  onChange={(colormap) => ctrl.setDisplay({ colormap })}
                  options={COLORMAPS.map((c) => ({ value: c, label: CMAP_LABEL[c] }))} />
              </Row>
              <Row label="Residual colormap">
                <Select<Colormap> size="sm" value={settings.residualColormap} aria-label="Residual colormap"
                  onChange={(residualColormap) => ctrl.setDisplay({ residualColormap })}
                  options={COLORMAPS.map((c) => ({ value: c, label: CMAP_LABEL[c] }))} />
              </Row>
              <Row label="Invert" inline>
                <Switch checked={settings.invert} onChange={(invert) => ctrl.setDisplay({ invert })} aria-label="Invert" />
              </Row>
              <Row label="NaN colour" inline>
                <input type="color" className="cv-color" aria-label="NaN colour" value={toHex(settings.nanColor)}
                  onChange={(e) => ctrl.setDisplay({ nanColor: e.target.value })} />
              </Row>
              <button type="button" className="cv-disclosure" aria-expanded={histOpen} onClick={() => setHistOpen((o) => !o)}>
                <VIcon name="chevron" /> Histogram and cuts
              </button>
              {histOpen && <div className="cv-menu__hist"><HistogramPanel width={300} /></div>}
            </>}
          </>}
          <div className="cv-menu__sep" role="separator" />
          <div className="cv-menu__tier">
            <Switch checked={following} disabled={!globalLinked} onChange={(on) => ctrl.setUnlinked(!on)}>
              Share with every viewer
            </Switch>
            <span className="cv-menu__hint cv-menu__hint--flush">
              {!globalLinked ? "Linking is off in the Display panel: this viewer keeps its own settings."
                : following ? "Changes here apply to every linked viewer (the Display panel)."
                  : "This viewer keeps its own settings."}
            </span>
          </div>
          <div className="cv-menu__actions">
            <Button size="sm" variant="ghost" icon={<VIcon name="palette" />} onClick={() => openDisplayPanel()}>
              Open the Display panel <Kbd keys="Shift+D" />
            </Button>
          </div>
        </div>
      </div>
    </aside>
  );
}

function BlackPoint({ value, scale, unit, label, onChange }: { value: number; scale: number; unit: string; label: string; onChange: (v: number) => void }) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    const v = parseNumber(draft ?? "");
    setDraft(null);
    if (v != null) onChange(v * scale);
  };
  return (
    <span className="cv-valslider">
      <input className="cv-num cv-num--wide" type="text" inputMode="decimal" aria-label={`${label}${unit ? ` (${unit})` : ""}`}
        value={draft ?? formatSig(value / scale)} onChange={(e) => setDraft(e.target.value)} onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter") { commit(); (e.target as HTMLInputElement).blur(); } if (e.key === "Escape") setDraft(null); }} />
      <span className="cv-menu__unit">{unit}</span>
    </span>
  );
}

/** <input type=color> needs #rrggbb. */
function toHex(css: string): string {
  const s = String(css || "").trim();
  if (/^#[0-9a-f]{6}$/i.test(s)) return s;
  if (/^#[0-9a-f]{3}$/i.test(s)) return `#${s.slice(1).split("").map((c) => c + c).join("")}`;
  return "#5b6475";
}

// ---- tools --------------------------------------------------------------------------

function ResidualPicker() {
  const ctrl = useController();
  const meta = useViewer((s) => s.meta);
  const residuals = useViewer((s) => s.residuals);
  const candidates = (meta?.tiers ?? []).filter((t) => !t.disabled && t.key !== "morph" && !/^pca\d+$/.test(t.key));
  const [a, setA] = useState(candidates.find((t) => t.key.toLowerCase() === "sr")?.key ?? candidates[0]?.key ?? "");
  const [b, setB] = useState(candidates.find((t) => ["hr", "lr", "mean"].includes(t.key.toLowerCase()) && t.key.toLowerCase() !== a.toLowerCase())?.key ?? candidates[1]?.key ?? "");
  const [op, setOp] = useState<ResidualOp>("diff");
  const options = candidates.map((t) => ({ value: t.key, label: t.label }));
  if (candidates.length < 2) return <p className="cv-menu__note">A residual needs two tiers.</p>;
  return (
    <div className="cv-respick">
      <div className="cv-respick__row">
        <Select size="sm" value={a} onChange={setA} options={options} aria-label="Tier A" />
        <Segmented<ResidualOp> size="sm" value={op} onChange={setOp} aria-label="Residual" className="cv-seg-plain"
          options={RESIDUAL_OPS.map((o) => ({ value: o, label: OP_LABEL[o] }))} />
        <Select size="sm" value={b} onChange={setB} options={options} aria-label="Tier B" />
      </div>
      <div className="cv-respick__row">
        <Button size="sm" variant="primary" disabled={!a || !b || a === b} onClick={() => ctrl.addResidual(op, a, b)}>Add residual tier</Button>
      </div>
      <p className="cv-menu__note">Computed in the browser; a coarser tier is resampled onto the finer grid (flux per pixel conserved). σ is the std tier when there is one, else the robust σ of the difference.</p>
      {residuals.length > 0 && (
        <ul className="cv-respick__list">
          {residuals.map((k) => (
            <li key={k}>
              <span className="mono">{ctrl.tierLabel(k)}</span>
              {parseResidualKey(k) && <Button size="sm" variant="ghost" onClick={() => ctrl.removeResidual(k)}>Remove</Button>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ToolsMenu() {
  const ctrl = useController();
  const tool = useViewer((s) => s.tool);
  const profileOpen = useViewer((s) => s.profileOpen);
  const playMs = useViewer((s) => s.playMs);
  const blinkMs = useViewer((s) => s.blinkMs);
  const count = useViewer((s) => s.meta?.count ?? 0);
  return (
    <div className="cv-menu cv-menu--tools">
      <p className="cv-menu__title">Pointer</p>
      <Segmented<"pan" | "lens"> size="sm" value={tool} onChange={(t) => ctrl.setTool(t)} aria-label="Pointer tool" className="cv-seg-plain"
        options={[
          { value: "pan", label: <span className="cv-tool"><VIcon name="pan" />Pan and zoom</span> },
          { value: "lens", label: <span className="cv-tool"><VIcon name="lens" />Magnifier lens</span> },
        ]} />
      <p className="cv-menu__note">
        <Kbd keys="L" /> toggles the lens, hold <Kbd keys="Alt" /> for a moment&apos;s lens. A click with the lens freezes the matched crop; <Kbd keys="S" /> saves it.
      </p>
      <p className="cv-menu__title">Profiles</p>
      <Switch checked={profileOpen} onChange={(on) => { ctrl.setPanels({ profileOpen: on }); if (!on) ctrl.setProfile(null); }}>
        Show the profile panel
      </Switch>
      <p className="cv-menu__note">Shift-drag across a frame for a line profile; click a point for a radial profile.</p>
      <p className="cv-menu__title">Residual tier</p>
      <ResidualPicker />
      <p className="cv-menu__title">Playback</p>
      {count > 1 && (
        <Row label="Run-through">
          <Slider value={playMs / 1000} min={0.3} max={3} scale="log" aria-label="Seconds per object" showValue
            format={(v) => `${v.toFixed(1)} s`} onChange={(v) => ctrl.setPlaySpeed(v * 1000)} />
        </Row>
      )}
      <Row label="Blink interval">
        <Slider value={blinkMs / 1000} min={0.2} max={2} scale="log" aria-label="Blink interval" showValue
          format={(v) => `${v.toFixed(1)} s`} onChange={(v) => ctrl.setBlinkMs(v * 1000)} />
      </Row>
    </div>
  );
}
