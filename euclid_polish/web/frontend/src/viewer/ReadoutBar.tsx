/* Pixel readout: x/y on the hovered tier's grid, RA/Dec (sexagesimal and
 * degrees) when the tier has a WCS, and the value of every band with its unit
 * for every visible tier at the same sky position (the crosshair). Without a
 * pointer it shows the object's catalogue position and the zoom. */
import { formatDec, formatDeg, formatRA } from "../format";
import { CopyButton } from "../ui";
import { useController, useViewer } from "./hooks";
import { formatValue, unitLabel } from "./readout";

const SHORT: Record<string, string> = { VIS: "VIS", Y_E: "Y", J_E: "J", H_E: "H" };

export function ReadoutBar() {
  const ctrl = useController();
  const readout = useViewer((s) => s.readout);
  const meta = useViewer((s) => s.meta);
  const index = useViewer((s) => s.index);
  const view = useViewer((s) => s.view);
  useViewer((s) => s.shown);
  if (!meta) return null;
  const obj = meta.objects?.[index];
  const first = ctrl.frameKeys().map((k) => ctrl.geomOf(k)).find(Boolean);
  const crop = view && first ? ctrl.cropOf(first.tier, view) : null;
  const zoom = crop && first ? Math.min(first.width, first.height) / crop.side : 1;
  const fov = crop?.angularSideArcsec ?? (first?.pixscale ? Math.min(first.width, first.height) * first.pixscale : null);

  const src = readout?.tiers.find((t) => t.tier === readout.tier);
  const sky = readout?.sky ?? null;
  const pos = sky ?? (obj && Number.isFinite(obj.ra) && Number.isFinite(obj.dec) ? { ra: obj.ra as number, dec: obj.dec as number } : null);
  const posText = pos ? `${formatRA(pos.ra)} ${formatDec(pos.dec)}` : "";
  const posDeg = pos ? `${formatDeg(pos.ra, 6)} ${formatDeg(pos.dec, 6, { signed: true })}` : "";
  return (
    <div className="cv-readout" aria-label="Pixel readout">
      <div className="cv-readout__pos">
        {src && src.x != null ? <span className="mono" title={`pixel on the ${src.label} grid (0-based)`}>x {src.x} · y {src.y}</span>
          : <span className="cv-readout__hint">{readout ? "outside the image" : "hover a frame for pixel values"}</span>}
        {pos && <span className="mono cv-readout__sky" title={sky ? "RA/Dec under the pointer (tier WCS)" : "object position (catalogue)"}>
          {sky ? "" : "object "}{posText} <span className="cv-readout__deg">({posDeg})</span>
          <CopyButton value={() => `${pos.ra.toFixed(7)} ${pos.dec.toFixed(7)}`} label="Copy RA/Dec (degrees)" />
        </span>}
        {!readout && <span className="mono cv-readout__zoom">{zoom > 1.001 ? `${zoom.toFixed(1)}×` : "fit"}{fov ? ` · ${fov < 10 ? fov.toFixed(2) : fov.toFixed(1)}″ field` : ""}</span>}
      </div>
      {readout && (
        <div className="cv-readout__tiers">
          {readout.tiers.map((t) => (
            <span key={t.tier} className="cv-readout__tier" data-src={t.tier === readout.tier || undefined}>
              <b>{t.label}</b>
              {t.values
                ? <span className="mono">
                  {t.values.slice(0, 6).map((v, k) => (
                    <span key={k} className="cv-readout__val">{t.bands.length > 1 ? `${SHORT[t.bands[k]] ?? t.bands[k] ?? `ch${k}`} ` : ""}{formatValue(v)}</span>
                  ))}
                  {t.values.length > 6 && <span className="cv-readout__val">+{t.values.length - 6}</span>}
                  {t.unit && <span className="cv-readout__unit">{unitLabel(t.unit)}</span>}
                  {t.tier !== readout.tier && t.x != null && <span className="cv-readout__xy">@{t.x},{t.y}</span>}
                </span>
                : <span className="cv-readout__hint">—</span>}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
