/* Status bar: cursor (or centre) RA/Dec in sexagesimal + degrees, a galactic
 * toggle, the field of view, the projection and the background HiPS value
 * under the cursor (`readPixel`). */
import { formatDec, formatDeg, formatRA } from "../../../format";
import { formatFov, icrsToGalactic } from "../../../sky/geometry";
import { PROJECTIONS, baseSurvey, type Projection } from "../../../sky/surveys";
import { CopyButton, Segmented, Switch, Tooltip } from "../../../ui";
import { coordText } from "./actions";
import { useAtlas } from "./store";
import type { AtlasUrl } from "./useAtlasUrl";

export function formatPixel(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") return Number.isFinite(v) ? String(Number(v.toPrecision(4))) : "—";
  if (Array.isArray(v) || ArrayBuffer.isView(v)) {
    const arr = Array.from(v as ArrayLike<number>).slice(0, 3);
    return arr.every((x) => typeof x === "number" && Number.isFinite(x)) ? `rgb ${arr.join(" ")}` : "—";
  }
  return "—";
}

export function StatusBar({ url }: { url: AtlasUrl }) {
  const cursor = useAtlas((s) => s.cursor);
  const view = useAtlas((s) => s.view);
  const pixel = useAtlas((s) => s.pixel);
  const pos = cursor ?? (view ? { ra: view.ra, dec: view.dec } : null);
  const gal = pos && url.gal ? icrsToGalactic(pos.ra, pos.dec) : null;
  const survey = baseSurvey(url.base);
  return (
    <div className="sky-status" role="group" aria-label="Sky status">
      <span className="sky-status__pos mono" aria-live="off">
        <span className="sky-status__k">{cursor ? "Cursor" : "Centre"}</span>
        {pos ? (
          gal ? (
            <span>l {formatDeg(gal[0], 4)} b {formatDeg(gal[1], 4, { signed: true })}</span>
          ) : (
            <>
              <span>{formatRA(pos.ra)} {formatDec(pos.dec)}</span>
              <span className="sky-status__deg">{formatDeg(pos.ra, 5)} {formatDeg(pos.dec, 5, { signed: true })}</span>
            </>
          )
        ) : <span>—</span>}
        {pos && <CopyButton value={() => coordText(pos.ra, pos.dec)} label="Copy coordinates (degrees)" />}
      </span>
      <Switch size="sm" checked={url.gal} onChange={url.setGal}>Galactic</Switch>
      <span className="sky-status__item"><span className="sky-status__k">FoV</span> <span className="mono">{view ? formatFov(view.fov) : "—"}</span></span>
      <Segmented<Projection> size="sm" aria-label="Projection" value={url.proj} onChange={url.setProj}
        options={PROJECTIONS.map((p) => ({ value: p, label: p, title: PROJ_TITLE[p] }))} />
      <Tooltip content={`${survey.label} value under the cursor${survey.format === "fits" ? " (FITS tiles: HiPS units)" : " (RGB tiles)"}`}>
        <span className="sky-status__item" tabIndex={0}>
          <span className="sky-status__k">Pixel</span> <span className="mono">{cursor ? formatPixel(pixel) : "—"}</span>
        </span>
      </Tooltip>
    </div>
  );
}

const PROJ_TITLE: Record<Projection, string> = {
  MOL: "Mollweide (all-sky)", AIT: "Hammer–Aitoff (all-sky)", SIN: "Orthographic (globe)", TAN: "Gnomonic (small fields)",
};
