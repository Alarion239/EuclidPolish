/* Synthetic › PSF › cutouts: the real Euclid star cutouts the ePSFs are
   stacked from, image first.

   One row above the viewer: the star in it (its field and its catalogue VIS
   magnitude, kept apart from the viewer readout's magnitude, which is the
   whole cutout's: neighbours and sky included), its catalogue card and the
   atlas; at the right, a badge only when it matters (offline: cached stars
   only; a stale catalogue) and the catalogue link. Then the navigator (viewer
   collection `cutouts`, served in electrons via each band's MAGZERO, the
   target marked at the centre) with the cutouts cached on this machine
   beside it when there is room (else below): one band at a time (`?band=`),
   ONE thumbnail per star on its own per-star stretch, the navigator's star
   outlined, a click shows that star. The navigator's count is its pager's
   (no badge repeats it). Below: the per-band cutout validity as one 100%
   stacked bar per band. The download step is in the How-this-is-produced
   drawer. */
import { useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { formatCount, formatNumber } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Caption, EmptyState, IconButton, Segmented, Skeleton, Toolbar, ToolbarSpacer, Tooltip,
} from "../../../ui";
import { ImageViewer, type ViewerApi, type ViewerState } from "../../../viewer";
import { URLS, type GalleryPage, type StarsPayload, type Totals } from "../dataApi";
import { Freshness, LinkButton, SkyButton, useFasrcOnline } from "../dataCommon";
import { BAND_STATE_HELP, BANDS, bandShort, cutoutTiles, decodeStars, type CutoutTile } from "../dataModel";
import { starMarker, validityBars } from "./psfModel";

/** Files per gallery page: the cache holds one file per cutout size (two
 *  sizes per star as a rule), so ~48 stars — one tile each. */
const PER_PAGE = 96;
/** Thumbnail pixels asked of the server: sharp at 2× for a ~128 px cell. */
const THUMB_PX = 256;
const STALE_S = 3 * 24 * 3600;
const isBand = (b: string) => (BANDS as readonly string[]).includes(b);

const thumbLabel = (it: CutoutTile, navSize: number | null) => [
  `Star ${it.id ?? it.file}`,
  it.mag != null ? `VIS ${it.mag.toFixed(2)}` : null,
  it.size && it.size !== navSize ? `${it.size} px` : null,
].filter(Boolean).join(", ");

const thumbTitle = (it: CutoutTile) =>
  it.sizes.length > 1 ? `${it.file} (cached at ${it.sizes.join(" and ")} px)` : it.file;

/** The gallery band: `?band=` (the old /cutouts/<band> links land there),
 *  else the interim `?gband=`, else VIS. */
function useGalleryBand(): [string, (b: string) => void] {
  const [band, setBand] = useUrlState("band", "");
  const [legacy, setLegacy] = useUrlState("gband", "");
  const value = isBand(band) ? band : isBand(legacy) ? legacy : "VIS";
  return [value, (b: string) => { setBand(b === "VIS" ? "" : b); setLegacy(""); }];
}

function CutoutGallery({ current, navSize, onPick }: { current: string | null; navSize: number | null; onPick: (id: number) => void }) {
  const [band, setBand] = useGalleryBand();
  const [page, setPage] = useUrlState("gpage", 1);
  const res = useResource<GalleryPage>(URLS.gallery(band, page, PER_PAGE), [band, page], { ttl: 60_000 });
  const data = res.data;
  const pages = data?.n_pages ?? 1;
  const tiles = useMemo(() => cutoutTiles(data?.items ?? [], navSize), [data, navSize]);
  return (
    <div className="dt-gallery">
      <div className="dt-gallery__head">
        <span className="dt-gallery__title">On this machine</span>
        <Segmented size="sm" value={band} onChange={(b) => { setBand(b); setPage(1); }} aria-label="Band of the cached cutouts"
          options={BANDS.map((b) => ({ value: b, label: bandShort(b) }))} />
        {data && tiles.length > 0 && <span className="dt-gallery__count">{`${formatCount(tiles.length)} star${tiles.length === 1 ? "" : "s"} on this page`}</span>}
      </div>
      {res.loading && !data ? <Skeleton lines={4} />
        : res.error ? <Callout tone="bad" title="Gallery did not load"><span className="dt-pre">{res.error.message}</span></Callout>
          : !data?.items.length ? (
            <EmptyState compact icon="image" title={`No ${bandShort(band)} cutouts cached yet`}>
              Showing a star in the viewer pulls its four cutouts.
            </EmptyState>
          ) : (
            <ul className="dt-thumbs" aria-label={`Cached ${bandShort(band)} cutouts, page ${data.page} of ${pages}`}>
              {tiles.map((it) => {
                const on = it.id != null && String(it.id) === current;
                const label = thumbLabel(it, navSize);
                return (
                  <li key={it.key}>
                    <button type="button" className="dt-thumb" aria-current={on || undefined} disabled={it.id == null}
                      aria-label={`${label}: show in the viewer`} title={thumbTitle(it)} onClick={() => { if (it.id != null) onPick(it.id); }}>
                      <img src={`${URLS.cutoutImage(band, it.file, THUMB_PX, data.output_dir ?? "")}&stretch=star`} alt="" loading="lazy" />
                      <span className="dt-thumb__cap">
                        <span>{it.id ?? it.file}</span>
                        <span className="dt-thumb__meta">
                          {it.mag != null ? it.mag.toFixed(2) : ""}{it.size && it.size !== navSize ? ` ${it.size} px` : ""}
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
      {pages > 1 && (
        <div className="dt-pager">
          <IconButton icon="chevronLeft" size="sm" label="Previous page" disabled={page <= 1} onClick={() => setPage(page - 1)} />
          <span className="dt-pager__pos">Page {data?.page} of {pages}</span>
          <IconButton icon="chevronRight" size="sm" label="Next page" disabled={page >= pages} onClick={() => setPage(page + 1)} />
        </div>
      )}
      <Caption>Each thumbnail on its own star's range, so bright and faint stars read alike.</Caption>
    </div>
  );
}

function ValidityBars({ data }: { data: StarsPayload }) {
  const bars = validityBars(data.band_stats);
  if (!bars.length) return null;
  const states = ["valid", "corrupted", "failed", "pending"] as const;
  return (
    <section className="rl-stack" aria-label="Cutout validity per band">
      <h3 className="syn-subtitle">Cutout validity per band</h3>
      <div className="syn-bars">
        {bars.map((b) => (
          <div key={b.band} className="syn-bar-row">
            <strong>{b.band}</strong>
            <Tooltip content={b.label}>
              <span className="syn-stack" role="img" aria-label={b.label} tabIndex={0}>
                {b.parts.map((p) => <span key={p.state} data-state={p.state} style={{ width: `${100 * p.fraction}%` }} />)}
              </span>
            </Tooltip>
            <span className="syn-num">{`${formatNumber(100 * (b.parts.find((p) => p.state === "valid")?.fraction ?? 0), { digits: 0 })}% valid`}</span>
          </div>
        ))}
      </div>
      <div className="syn-stack-legend" aria-label="States">
        {states.map((st) => (
          <Tooltip key={st} content={BAND_STATE_HELP[st]}><span tabIndex={0}><i data-state={st} aria-hidden="true" />{st}</span></Tooltip>
        ))}
      </div>
      <Caption>Each star counts once per band, under its best outcome (valid › corrupted › failed › pending).</Caption>
    </section>
  );
}

export function Cutouts({ data }: { data: StarsPayload }) {
  const totals = useResource<Totals>(URLS.totals, [], { ttl: 60_000 });
  const t = totals.data;
  const [current, setCurrent] = useState<string | null>(null);
  const api = useRef<ViewerApi | null>(null);
  const navigate = useNavigate();
  const { online } = useFasrcOnline();
  const stars = useMemo(() => decodeStars(data), [data]);
  const star = useMemo(() => (current ? stars.find((s) => String(s.id) === current) ?? null : null), [stars, current]);
  const pick = (id: number) => {
    void api.current?.goToId(String(id)).then((ok) => { if (!ok) openInspector({ kind: "star", id: String(id) }); });
  };
  usePageActions([
    { id: "cutouts-details", label: "Show the current star's catalogue card", group: "PSF", disabled: !current,
      run: () => { if (current) openInspector({ kind: "star", id: current }); } },
    { id: "cutouts-catalog", label: "Open the usable stars in the catalogue", group: "PSF", run: () => navigate("/synthetic/psf?cut=nav") },
  ]);
  const syncedAt = t?.catalog.present ? t.catalog.mtime : null;
  const stale = syncedAt != null && Date.now() / 1000 - syncedAt > STALE_S;
  const markers = useMemo(() => starMarker(t?.size, current), [t?.size, current]);
  const onViewerState = (s: ViewerState) => { setCurrent(s.id); };
  if (t && !t.catalog.present) {
    return (
      <EmptyState icon="database" title="No synchronised star catalogue"
        action={<Button asChild variant="primary"><Link to="/synthetic/psf?how=1">How this is produced</Link></Button>}>
        The navigator reads the FASRC catalogue mirror: pull stars.csv first.
      </EmptyState>
    );
  }
  return (
    <div className="rl-stack">
      <Toolbar label="The star in the viewer">
        {current && (
          <span className="dt-star" role="group" aria-label={`Star ${current}`}>
            <strong>Star {current}</strong>
            {star?.field && <Badge size="sm">{star.field}</Badge>}
            {star && (
              <Tooltip content="The catalogue's VIS PSF magnitude of this star. The viewer's own magnitude is the whole cutout's, neighbours and sky included.">
                <span tabIndex={0} className="dt-star__mag mono">{`star VIS ${formatNumber(star.mag, { digits: 2 })} AB`}</span>
              </Tooltip>
            )}
            <Button size="sm" variant="ghost" icon="info" onClick={() => openInspector({ kind: "star", id: current })}>Details</Button>
            {star && <SkyButton ra={star.ra} dec={star.dec} fov={0.02} layers={["stars", "q1-tiles:0.3"]} inspect={`star:${star.id}`} />}
          </span>
        )}
        <ToolbarSpacer />
        {!online && (
          <Tooltip content="FASRC is offline: only stars whose cutouts are cached on this machine can be shown.">
            <span tabIndex={0}><Badge size="sm" tone="warn" dot>Cached only</Badge></span>
          </Tooltip>
        )}
        {stale && <Freshness at={syncedAt} label="catalogue synced" stale={STALE_S} />}
        <LinkButton to="/synthetic/psf?cut=nav" icon="table" label="Catalogue" hint="The usable stars in the catalogue" />
      </Toolbar>
      {totals.error && !t && <Callout tone="bad" title="Totals did not load"><span className="dt-pre">{totals.error.message}</span></Callout>}
      <div className="dt-vsplit">
        <div className="dt-vsplit__viewer">
          <ImageViewer collection="cutouts" urlKey="cut" markers={markers} onReady={(a) => { api.current = a; }}
            onState={onViewerState} />
        </div>
        <div className="dt-vsplit__aside" role="region" aria-label="Cached cutouts" data-fill="">
          <CutoutGallery current={current} navSize={t?.size ?? null} onPick={pick} />
        </div>
      </div>
      <ValidityBars data={data} />
    </div>
  );
}
