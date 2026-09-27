/* The readout: ONE line of reserved height under the frames (no layout jump).
 *
 *   hovering  x 204  y 346   RA Dec (copy)   VIS  LR 76.9 e⁻  HR 0.23 e⁻
 *   idle      RA Dec of the object (copy)   LR VIS 21.48 AB  HR …   2.0× 12.8″
 *
 * with the save status or the object's label at the right. Values are the
 * shown band's (a composite shows the first band); every band of every tier
 * is in the line's tooltip and in `getReadout()`. Below ~560 px of viewer
 * width, or when the widest line for the shown tiers would not fit
 * (`readoutLines`: the inspector, the bottom sheet, three tiers at 720 px),
 * the readout reserves TWO or more lines — the position first, then the
 * per-tier values, each tier's name, value and unit kept together and the
 * line broken only between tiers — decided before any hover, so it never
 * changes height under the pointer; the object's label gives way first. */

let measureCtx: CanvasRenderingContext2D | null | undefined;
/** A string's width in the readout's font (≈ 6.6 px a character without canvas). */
function measurer(el: HTMLElement): (text: string) => number {
  if (measureCtx === undefined) {
    try { measureCtx = document.createElement("canvas").getContext("2d"); } catch { measureCtx = null; }
  }
  const ctx = measureCtx;
  const cs = getComputedStyle(el);
  const size = parseFloat(cs.fontSize) || 12;
  if (!ctx) return (t) => t.length * size * 0.55;
  const mono = getComputedStyle(document.documentElement).getPropertyValue("--font-mono").trim() || "monospace";
  // numbers are tabular mono, names the UI face: measure in mono (the wider)
  ctx.font = `${size}px ${mono}`;
  return (t) => ctx.measureText(t).width;
}
import { useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import { formatDec, formatDeg, formatRA } from "../format";
import { CopyButton } from "../ui";
import { bandLabel, readoutTierName } from "./barModel";
import { useController, useSettings, useViewer } from "./hooks";
import { cubeIsEmpty, formatValue, readoutLines, unitLabel } from "./readout";
import type { ReadoutTier } from "./types";

const fovText = (fov: number) => `${fov < 10 ? fov.toFixed(2) : fov.toFixed(1)}″`;

/** The index of the band a tier's value is read in (the shown band, else the first). */
function bandIndex(t: ReadoutTier, color: string): number {
  const k = t.bands.indexOf(color);
  return k >= 0 ? k : 0;
}

export function ReadoutBar() {
  const ctrl = useController();
  const readout = useViewer((s) => s.readout);
  const meta = useViewer((s) => s.meta);
  const index = useViewer((s) => s.index);
  const view = useViewer((s) => s.view);
  const mags = useViewer((s) => s.mags);
  const save = useViewer((s) => s.save);
  const shown = useViewer((s) => s.shown);
  const settings = useSettings();
  const ref = useRef<HTMLDivElement>(null);
  const [lines, setLines] = useState(1);
  const tierKey = meta ? ctrl.frameKeys().map((k) => `${k}:${ctrl.tierLabel(k)}`).join("|") : "";
  const anyWcs = Object.values(shown).some((sh) => !!sh.rec.wcs);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !meta) return;
    const keys = ctrl.frameKeys();
    const hasStd = (meta.tiers ?? []).some((t) => t.key === "std");
    const tiers = keys.map((k) => ({
      name: k.startsWith("res:") ? ctrl.tierLabel(k) : readoutTierName(ctrl.tierLabel(k)), unit: ctrl.tierMeta(k)?.unit ?? "",
      sigma: hasStd && k.toLowerCase() === "sr",
    }));
    const hasSky = anyWcs || (meta.objects ?? []).some((o) => Number.isFinite(o.ra));
    const decide = () => {
      const cs = getComputedStyle(el);
      const avail = el.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0);
      if (!(avail > 0)) return;
      setLines(readoutLines(tiers, measurer(el), { width: avail, hasSky }));
    };
    decide();
    if (typeof ResizeObserver === "undefined") return;
    let w = el.clientWidth;
    const ro = new ResizeObserver(() => { if (el.clientWidth !== w) { w = el.clientWidth; decide(); } });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ctrl, meta, tierKey, anyWcs]);
  if (!meta) return <div className="cv-readout" aria-hidden="true" />;
  const obj = meta.objects?.[index];
  const keys = ctrl.frameKeys();
  const first = keys.map((k) => ctrl.geomOf(k)).find(Boolean);
  const crop = view && first ? ctrl.cropOf(first.tier, view) : null;
  const zoom = crop && first ? Math.min(first.width, first.height) / crop.side : 1;
  const fov = crop?.angularSideArcsec ?? (first?.pixscale ? Math.min(first.width, first.height) * first.pixscale : null);
  const name = (k: string) => (k.startsWith("res:") ? ctrl.tierLabel(k) : readoutTierName(ctrl.tierLabel(k)));

  const src = readout?.tiers.find((t) => t.tier === readout.tier);
  const sky = readout?.sky ?? null;
  const pos = sky ?? (obj && Number.isFinite(obj.ra) && Number.isFinite(obj.dec) ? { ra: obj.ra as number, dec: obj.dec as number } : null);
  const band = src && src.bands.length > 1 ? src.bands[bandIndex(src, settings.color)] : "";
  const full = readout?.tiers.map((t) => `${t.label}: ${t.values ? t.values.map((v, k) => `${t.bands[k] ?? `ch${k}`} ${formatValue(v)}`).join(", ") : "outside"} ${unitLabel(t.unit)}`).join("\n");
  // The status region is always mounted (announced when its text changes).
  const right = <>
    <span className="cv-readout__status" role="status" aria-live="polite" data-tone={save.tone || undefined}>{save.text}</span>
    {!save.text && obj?.label && <span className="cv-readout__obj" title={obj.id}>{obj.label}</span>}
  </>;

  return (
    <div ref={ref} className="cv-readout" data-lines={lines > 1 ? String(lines) : undefined} aria-label="Pixel readout" title={full || undefined}
      style={lines > 1 ? { "--cv-readout-lines": lines } as CSSProperties : undefined}>
      <span className="cv-readout__main">
        {readout ? (
          src && src.x != null
            ? <span className="cv-readout__item mono" title={`Pixel on the ${src.label} grid (0-based)`}>x {src.x}&ensp;y {src.y}</span>
            : <span className="cv-readout__hint">Outside the image</span>
        ) : null}
        {pos && (
          <span className="cv-readout__item cv-readout__sky mono" title={`${sky ? "Under the pointer" : "Object position"}: ${formatDeg(pos.ra, 6)} ${formatDeg(pos.dec, 6, { signed: true })}`}>
            {formatRA(pos.ra)} {formatDec(pos.dec)}
            <CopyButton value={() => `${pos.ra.toFixed(7)} ${pos.dec.toFixed(7)}`} label="Copy RA/Dec (degrees)" />
          </span>
        )}
        {/* a narrow viewer wraps the line here: the position on line 1, the values on line 2 */}
        {pos || readout ? <span className="cv-readout__break" aria-hidden="true" /> : null}
        {readout ? <>
          {band && <span className="cv-readout__band">{bandLabel(band)}</span>}
          {readout.tiers.map((t) => {
            const sh = shown[t.tier];
            // a tier with no data at all reads "no data", not "NaN MJy/sr" at every pixel
            const blank = sh?.kind === "cube" && cubeIsEmpty(sh.rec);
            return (
              <span key={t.tier} className="cv-readout__item" data-src={t.tier === readout.tier || undefined}>
                <b>{name(t.tier)}</b>{" "}
                {blank ? <span className="cv-readout__hint">no data</span> : <>
                  <span className="mono">{t.values ? formatValue(t.values[bandIndex(t, settings.color)] ?? NaN) : "—"}</span>
                  {t.values && t.unit && <span className="cv-readout__unit"> {unitLabel(t.unit)}</span>}
                </>}
              </span>
            );
          })}
        </> : <>
          {keys.filter((k) => mags[k]).map((k) => (
            <span key={k} className="cv-readout__item"><b>{name(k)}</b> <span className="mono">{mags[k]}</span></span>
          ))}
          {!keys.some((k) => mags[k]) && <span className="cv-readout__hint">Hover a frame for pixel values</span>}
          {(zoom > 1.001 || !!fov) && (
            <span className="cv-readout__item cv-readout__zoom mono"
              title={zoom > 1.001 ? `Zoomed ${zoom.toFixed(1)}×${fov ? `: ${fovText(fov)} on a side` : ""}` : `The whole image${fov ? `, ${fovText(fov)} on a side` : ""}`}>
              {zoom > 1.001 ? `${zoom.toFixed(1)}×${fov ? ` ${fovText(fov)}` : ""}` : fovText(fov as number)}
            </span>
          )}
        </>}
      </span>
      <span className="cv-readout__side">{right}</span>
    </div>
  );
}
