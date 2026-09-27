/* Shared pieces of the Figures tabs: server-rendered images that show the
 * server's error text, saved-result thumbnails and badges. */
import { useEffect, useState, type ReactNode } from "react";
import { ApiError, apiGet } from "../../api/client";
import { Badge, Button, Icon, Skeleton, Tooltip, cx } from "../../ui";
import { panelUrl, type SavedResult } from "./api";
import { wcsState } from "./model";

/** Why an image URL failed: the server's `{error}` text when it sent one. */
export async function imageError(src: string): Promise<string> {
  try {
    await apiGet(src);
    return "The server answered but the image could not be decoded.";
  } catch (e) {
    if (e instanceof ApiError) {
      if (e.status === 200) return "The server answered but the image could not be decoded.";
      return e.message;
    }
    return e instanceof Error ? e.message : String(e);
  }
}

type ImgState = { src: string; state: "loading" | "ready" | "error"; error?: string };

/** A server-rendered image (matplotlib plate, grid preview): a skeleton while
 *  it renders, the server's error text (with Retry) when it fails.
 *  `keepPrevious` holds the last rendered image (dimmed, with an "Updating"
 *  badge) while the next `src` loads instead of blanking the box; `pending`
 *  marks the shown image as about to be replaced (e.g. an edit settling). */
export function ServerImage({ src, alt, className, minHeight = 240, paper = true, onError, keepPrevious = false, pending = false, children }: {
  src: string | null; alt: string; className?: string; minHeight?: number; paper?: boolean;
  onError?: (message: string) => void; keepPrevious?: boolean; pending?: boolean; children?: ReactNode;
}) {
  const [s, setS] = useState<ImgState | null>(null);
  const [nonce, setNonce] = useState(0);
  const [shown, setShown] = useState<string | null>(null);
  const cur = s && s.src === src ? s : src ? { src, state: "loading" as const } : null;
  useEffect(() => { if (cur?.state === "error" && cur.error) onError?.(cur.error); }, [cur?.state, cur?.error, onError]);
  if (!src || !cur) return <div className={cx("fig-image", paper && "fig-image--paper", className)} style={{ minHeight }}>{children}</div>;
  const url = nonce ? `${src}${src.includes("?") ? "&" : "?"}_r=${nonce}` : src;
  const stale = keepPrevious && cur.state === "loading" && shown && shown !== url ? shown : null;
  const updating = !!stale || (pending && cur.state === "ready");
  return (
    <div className={cx("fig-image", paper && "fig-image--paper", className)}
      style={{ minHeight: cur.state === "error" ? Math.min(minHeight, 120) : minHeight }}
      aria-busy={cur.state === "loading" || updating} data-state={cur.state}>
      {stale && <img key={`stale:${stale}`} src={stale} alt="" aria-hidden className="is-stale" />}
      {cur.state !== "error" && (
        <img key={url} src={url} alt={alt}
          className={cur.state === "loading" ? "is-loading" : pending ? "is-stale" : undefined}
          onLoad={() => { setS({ src, state: "ready" }); if (keepPrevious) setShown(url); }}
          onError={() => { void imageError(url).then((error) => setS({ src, state: "error", error })); }} />
      )}
      {cur.state === "loading" && !stale && <div className="fig-image__overlay" role="status"><Skeleton height={Math.max(60, minHeight - 40)} /><span className="sr-only">Rendering {alt}</span></div>}
      {updating && <span className="fig-image__updating" role="status"><Badge size="sm" tone="info" dot>Updating</Badge></span>}
      {cur.state === "error" && (
        <div className="fig-image__error" role="alert">
          <Icon name="warn" />
          <span>{cur.error}</span>
          <Button size="sm" onClick={() => { setNonce(Date.now()); setS({ src, state: "loading" }); }}>Retry</Button>
        </div>
      )}
    </div>
  );
}

/** A saved result's thumbnail (its own thumbnail recipe, or `recipe`). */
export function ResultThumb({ result, recipe, size = 64, className, title }: {
  result: Pick<SavedResult, "id" | "label">; recipe?: string | null; size?: number; className?: string; title?: string;
}) {
  const [failed, setFailed] = useState<string | null>(null);
  const src = panelUrl(result.id, recipe, Math.round(size * 2));
  if (failed === src) {
    return <span className={cx("fig-thumb fig-thumb--empty", className)} style={{ width: size, height: size }} title={title ?? "No preview"}><Icon name="image" /></span>;
  }
  return (
    <img className={cx("fig-thumb", className)} src={src} width={size} height={size} loading="lazy" decoding="async"
      alt={title ?? `${result.label} preview`} title={title} onError={() => setFailed(src)} />
  );
}

export function WcsBadge({ result, size = "sm" }: { result: SavedResult; size?: "sm" | "md" }) {
  const state = wcsState(result);
  const tip = state === "all" ? "Every saved crop keeps its celestial WCS (sky-matched crops)."
    : state === "partial" ? `WCS kept for ${(result.wcs_tiers ?? []).join(", ")} only.`
      : result.regime === "synthetic" ? "Synthetic scene: no sky coordinates." : "Saved before crops kept their WCS.";
  return (
    <Tooltip content={tip}>
      <span tabIndex={0} className="fig-badge-wrap">
        <Badge size={size} tone={state === "all" ? "good" : state === "partial" ? "warn" : "neutral"} dot={state !== "none"}>
          {state === "all" ? "WCS" : state === "partial" ? "WCS partial" : "no WCS"}
        </Badge>
      </span>
    </Tooltip>
  );
}

export function RegimeBadge({ regime }: { regime: string | null | undefined }) {
  if (!regime) return <Badge size="sm">—</Badge>;
  return <Badge size="sm" tone={regime === "real" ? "info" : "accent"}>{regime}</Badge>;
}
