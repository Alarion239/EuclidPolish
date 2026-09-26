/* Image figures: Gallery (thumbnail grid) and PngFigure (a server-rendered
   matplotlib PNG on a paper-white inset, readable in both themes). */
import type { ReactNode } from "react";
import { buttonClass } from "./Button";
import { Empty } from "./display";

export type GalleryItem = { src: string; href?: string; label?: string; onClick?: () => void };

/** Responsive thumbnail grid on paper-white cells. Each cell links out (new
 *  tab) or fires onClick. */
export function Gallery({ items, thumb = 150, empty }: { items: GalleryItem[]; thumb?: number; empty?: ReactNode }) {
  if (!items.length) return <Empty>{empty ?? "nothing rendered yet"}</Empty>;
  return (
    <div className="ui-gallery" style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${thumb}px, 1fr))` }}>
      {items.map((it, i) => {
        const inner = <>
          <img src={it.src} loading="lazy" alt={it.label ?? ""} />
          {it.label && <span className="ui-gallery__cap mono">{it.label}</span>}
        </>;
        return it.href
          ? <a key={i} className="ui-gallery__cell" href={it.href} target="_blank" rel="noreferrer">{inner}</a>
          : <button key={i} className="ui-gallery__cell" onClick={it.onClick} type="button">{inner}</button>;
      })}
    </div>
  );
}

/** A matplotlib PNG with an optional chip toolbar. `srcFor(active)` builds the
 *  URL for the selected toolbar key. */
export function PngFigure(
  { srcFor, toolbar, active, onActive, downloadSrc, alt, minHeight = 220 }: {
    srcFor: (active?: string) => string;
    toolbar?: { key: string; label: string }[];
    active?: string; onActive?: (key: string) => void;
    downloadSrc?: (active?: string) => string;
    alt?: string; minHeight?: number;
  },
) {
  const src = srcFor(active);
  return (
    <div className="ui-figure">
      {((toolbar && toolbar.length > 0) || downloadSrc) && (
        <div className="ui-figure__bar">
          {(toolbar ?? []).map((t) => (
            <button key={t.key} type="button" className="ui-chip" data-on={t.key === active}
              aria-pressed={t.key === active} onClick={() => onActive?.(t.key)}>{t.label}</button>
          ))}
          {downloadSrc && (
            <a className={buttonClass("default", "sm", "ui-figure__download")} href={downloadSrc(active)} download>
              Download 300 dpi
            </a>
          )}
        </div>
      )}
      <div className="ui-figure__paper" style={{ minHeight }}>
        <img src={src} alt={alt ?? ""} loading="lazy" />
      </div>
    </div>
  );
}
