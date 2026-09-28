/* The full-size view of a figure image (image-first pass 2026-09-27): the
 * whole window (a thin inset) on the viewer's neutral dark light table in
 * both themes, ONE header row (title, what it is, scale, actions, close) and
 * the image as large as the rest allows — always larger than the in-page
 * preview it opens from (model.ts lightboxStageHeight / previewPaperCap).
 * "Fit" shows the whole image (crops upscaled with no smoothing, so every
 * pixel stays a pixel); "Fit width" fills the width and scrolls down (the
 * tall A4 sheet); "Actual size" shows the image's own pixels and scrolls.
 * Used by the sheet preview (the A4 sheet) and the saved-crop thumbnails
 * (the Sheet's crop pool), with the result's panels one click
 * apart. Esc or Close returns, focus back on what opened it. */
import * as RDialog from "@radix-ui/react-dialog";
import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { openInspector } from "../../app/inspector";
import { formatNumber } from "../../format";
import { Button, Icon, Segmented } from "../../ui";
import { panelUrl, type SavedResult } from "./api";
import { ServerImage } from "./common";
import { LIGHTBOX, cropSideArcsec, gridHref, recipeLabel, resultRegime, sourceLabel, viewerLink } from "./model";

type Scale = "fit" | "width" | "actual";

const GEOMETRY = {
  "--fig-lb-inset": `${LIGHTBOX.inset}px`,
  "--fig-lb-head": `${LIGHTBOX.head}px`,
  "--fig-lb-panels": `${LIGHTBOX.panels}px`,
} as CSSProperties;

export function Lightbox({ open, onOpenChange, title, description, src, alt, pixelated = false, actual = true, before, footer }: {
  open: boolean; onOpenChange: (open: boolean) => void; title: ReactNode; description?: ReactNode;
  src: string | null; alt: string;
  /** Upscale with no smoothing (saved crops are tens of pixels). */
  pixelated?: boolean;
  /** Offer "Fit width" and "Actual size" (a large sheet); off for tiny crops. */
  actual?: boolean;
  /** A row under the header (e.g. the panel choices). */
  before?: ReactNode;
  /** Actions in the header row, before Close. */
  footer?: ReactNode;
}) {
  const [scale, setScale] = useState<Scale>("fit");
  const stage = useRef<HTMLDivElement>(null);
  /** What had focus when the view opened: focus goes back there on close
   *  (Radix returns it only to a Dialog.Trigger, and these open from plain
   *  buttons). Read in a layout effect, before the dialog takes focus. */
  const opener = useRef<HTMLElement | null>(null);
  useLayoutEffect(() => {
    if (open) opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  }, [open]);
  // "Actual size" / "Fit width" open on the top middle of the sheet, not its left margin.
  useEffect(() => {
    const el = stage.current;
    if (scale === "fit" || !el) return;
    const id = requestAnimationFrame(() => { el.scrollLeft = Math.max(0, (el.scrollWidth - el.clientWidth) / 2); el.scrollTop = 0; });
    return () => cancelAnimationFrame(id);
  }, [scale]);
  const hasDesc = description != null && description !== false && description !== "";
  return (
    <RDialog.Root open={open} onOpenChange={(o) => { if (!o) setScale("fit"); onOpenChange(o); }}>
      <RDialog.Portal>
        <RDialog.Overlay className="fig-lightbox__scrim" />
        <RDialog.Content className="fig-lightbox" style={GEOMETRY}
          {...(hasDesc ? {} : { "aria-describedby": undefined })}
          onCloseAutoFocus={(e) => {
            const el = opener.current;
            if (el && el.isConnected) { e.preventDefault(); el.focus(); }
          }}>
          <header className="fig-lightbox__head">
            <RDialog.Title className="fig-lightbox__title">{title}</RDialog.Title>
            {hasDesc
              ? <RDialog.Description asChild><span className="fig-lightbox__desc">{description}</span></RDialog.Description>
              : <span className="fig-lightbox__desc" />}
            {actual && (
              <Segmented<Scale> size="sm" aria-label="Image scale" value={scale} onChange={setScale} className="fig-lightbox__scale"
                options={[
                  { value: "fit", label: "Fit", title: "The whole image" },
                  { value: "width", label: "Fit width", title: "Fill the width; scroll down" },
                  { value: "actual", label: "Actual size", title: "The image's own pixels (scrolls)" },
                ]} />
            )}
            {footer && <span className="fig-lightbox__actions">{footer}</span>}
            <RDialog.Close asChild>
              <button type="button" className="ui-iconbtn ui-iconbtn--ghost ui-iconbtn--sm fig-lightbox__close" aria-label="Close" title="Close (Esc)">
                <Icon name="close" />
              </button>
            </RDialog.Close>
          </header>
          {before && <div className="fig-lightbox__before">{before}</div>}
          <div ref={stage} className="fig-lightbox__stage" data-scale={scale} data-pixelated={pixelated || undefined}>
            <ServerImage src={src} alt={alt} paper={false} minHeight={220} className="fig-lightbox__image" />
          </div>
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  );
}

/** A saved result at full size: its panels (recipes) one click apart, and
 *  the ways on (its card, the sheet, its source viewer). Stays mounted while
 *  closing (the last result), so focus returns to the thumbnail. */
export function ResultLightbox({ result, onClose }: { result: SavedResult | null; onClose: () => void }) {
  const [recipe, setRecipe] = useState<string | null>(null);
  const [last, setLast] = useState<SavedResult | null>(result);
  if (result && result !== last) setLast(result); // state derived from the previous render
  const r = result ?? last;
  if (!r) return null;
  const shown = recipe && r.recipes.includes(recipe as never) ? recipe : r.thumbnail ?? r.recipes[0] ?? null;
  const side = cropSideArcsec(r);
  const link = viewerLink(r);
  const close = () => { setRecipe(null); onClose(); };
  return (
    <Lightbox open={!!result} onOpenChange={(o) => { if (!o) close(); }} title={r.label} pixelated actual={false}
      description={[sourceLabel(r), r.logical_tiers.join(", "), side != null ? `${formatNumber(side, { digits: 2 })}″ crop` : null]
        .filter(Boolean).join(" · ")}
      src={shown ? panelUrl(r.id, shown) : null} alt={`${r.label}, ${shown ? recipeLabel(shown) : ""}`}
      before={r.recipes.length > 1 ? (
        <div className="fig-lightbox__panels" role="group" aria-label="Panel">
          {r.recipes.map((k) => (
            <button key={k} type="button" className="fig-lightbox__panel" data-on={k === shown} aria-pressed={k === shown}
              onClick={() => setRecipe(k)}>{recipeLabel(k)}</button>
          ))}
        </div>
      ) : undefined}
      footer={<>
        <Button size="sm" variant="ghost" onClick={() => { close(); openInspector({ kind: "figure", id: r.id }); }}>Open its card</Button>
        <Button asChild size="sm" variant="ghost" icon="columns"><Link to={gridHref([r.id], resultRegime(r) ?? "real")} onClick={close}>Sheet</Link></Button>
        {link && <Button asChild size="sm"><Link to={link.to} onClick={close}>{link.label}</Link></Button>}
      </>} />
  );
}
