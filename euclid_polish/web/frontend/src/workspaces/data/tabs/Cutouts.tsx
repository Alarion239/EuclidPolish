/* Data › Cutouts (spec §8.4): real Euclid star cutouts, image first
 * (docs/superpowers/specs/2026-09-27-image-first-viewer-design.md).
 *
 * One toolbar row: the star in the viewer (its field, VIS magnitude, the
 * catalogue inspector and the atlas; its position and the copy button are in
 * the viewer's readout) and, at the right, the navigator's size (the stars
 * valid in all four bands at one common size, from the synchronised
 * FASRC-mirror catalogue — works offline; its freshness in the tip, a badge
 * only when stale) and the catalogue. Then the navigator (viewer collection
 * `cutouts`, served in electrons via each band's MAGZERO, so the console's
 * absolute e⁻ transfer shows it like every other image; it fits its own
 * frames under its top), with the cutouts cached on this machine beside it when there is room (else
 * below): one band at a time, a thumbnail per star, the navigator's star
 * outlined, a click shows that star. The download_euclid_cutouts FASRC step
 * and the archive login link sit in the collapsed section at the bottom. The
 * viewer object, gallery band and page are in the URL. */
import { useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { StepById } from "../../../fasrc";
import { formatCount, formatDateTime, formatNumber, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, EmptyState, IconButton, Page, Section, Segmented, Skeleton, Tooltip,
} from "../../../ui";
import { ImageViewer, type ViewerApi, type ViewerState } from "../../../viewer";
import { URLS, useStars, type GalleryPage, type Totals } from "../api";
import { BarActions, DataBar, Freshness, LinkButton, OFFLINE_HINT, SkyButton, Spacer, useFasrcOnline } from "../common";
import { BANDS, bandShort, cutoutTiles, decodeStars, type CutoutTile } from "../model";
import "../register";
import "../data.css";

/** Files per gallery page: the cache holds one file per cutout size (two
 *  sizes per star as a rule), so ~48 stars — one tile each. */
const PER_PAGE = 96;
/** Thumbnail pixels asked of the server: sharp at 2× for a ~128 px cell. */
const THUMB_PX = 256;
const STALE_S = 3 * 24 * 3600;

/** The star in the viewer: its facts (left of the toolbar) and its links (the actions). */
function useCurrentStar(id: string | null) {
  const stars = useStars();
  return useMemo(() => (id ? decodeStars(stars.data).find((s) => String(s.id) === id) ?? null : null), [stars.data, id]);
}

function CurrentStar({ id, star }: { id: string | null; star: ReturnType<typeof useCurrentStar> }) {
  if (!id) return null;
  if (!star) return <span className="dt-star"><strong>Star {id}</strong></span>;
  return (
    <span className="dt-star" role="group" aria-label={`Star ${star.id}`}>
      <strong>Star {star.id}</strong>
      {star.field && <Badge size="sm">{star.field}</Badge>}
      <span className="dt-star__mag mono">VIS {formatNumber(star.mag, { digits: 2 })}</span>
    </span>
  );
}

const thumbLabel = (it: CutoutTile, navSize: number | null) => [
  `Star ${it.id ?? it.file}`,
  it.mag != null ? `VIS ${it.mag.toFixed(2)}` : null,
  it.size && it.size !== navSize ? `${it.size} px` : null,
].filter(Boolean).join(", ");

/** The tile's tooltip: the file and the sizes cached for the star. */
const thumbTitle = (it: CutoutTile) =>
  it.sizes.length > 1 ? `${it.file} (cached at ${it.sizes.join(" and ")} px)` : it.file;

/** The cached cutouts of one band as a grid of thumbnails on the light-table
 *  surround (white stars on black, like the viewer), one per star (its
 *  navigator-size file when cached); the navigator's star is outlined and a
 *  click shows that star in the viewer. */
function CutoutGallery({ current, navSize, onPick }: { current: string | null; navSize: number | null; onPick: (id: number) => void }) {
  const [band, setBand] = useUrlState("gband", "VIS");
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
        {data && <span className="dt-gallery__count">{formatCount(data.total)} files</span>}
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
                      <img src={URLS.cutoutImage(band, it.file, THUMB_PX, data.output_dir ?? "")} alt="" loading="lazy" />
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
    </div>
  );
}

export default function Cutouts() {
  const totals = useResource<Totals>(URLS.totals, [], { ttl: 60_000 });
  const t = totals.data;
  const [current, setCurrent] = useState<string | null>(null);
  const [steps, setSteps] = useUrlState("steps", false);
  const api = useRef<ViewerApi | null>(null);
  const navigate = useNavigate();
  const { online } = useFasrcOnline();
  const pick = (id: number) => {
    void api.current?.goToId(String(id)).then((ok) => { if (!ok) openInspector({ kind: "star", id: String(id) }); });
  };
  usePageActions([
    { id: "cutouts-details", label: "Show the current star's details", group: "Cutouts", disabled: !current,
      run: () => { if (current) openInspector({ kind: "star", id: current }); } },
    { id: "cutouts-download", label: "Download Euclid star cutouts (FASRC)…", group: "Cutouts", run: () => setSteps(true) },
    { id: "cutouts-catalog", label: "Open the star catalogue", group: "Cutouts", run: () => navigate("/data/catalog?cut=nav") },
  ]);
  const noCatalog = t && !t.catalog.present;
  const syncedAt = t?.catalog.present ? t.catalog.mtime : null;
  const stale = syncedAt != null && Date.now() / 1000 - syncedAt > STALE_S;
  const onViewerState = (s: ViewerState) => { setCurrent(s.id); };
  const star = useCurrentStar(current);
  return (
    <Page className="dt-page dt-page--image">
      <DataBar label="Cutouts" compactable>
        <CurrentStar id={current} star={star} />
        <Spacer />
        <div className="dt-bar__status">
          {!online && (
            <Tooltip content="FASRC is offline: only stars whose cutouts are cached on this machine can be shown.">
              <span tabIndex={0} className="dt-tipbadge"><Badge size="sm" tone="warn" dot>Cached only</Badge></span>
            </Tooltip>
          )}
          {t ? (
            <Tooltip content={`Stars valid in all four bands at one common cutout size${t.size ? ` (${t.size} px)` : ""}, from the synchronised FASRC catalogue${syncedAt != null ? `; synced ${formatRelative(syncedAt * 1000)}, ${formatDateTime(syncedAt * 1000)}` : ""}`}>
              <span tabIndex={0} className="dt-tipbadge">
                <Badge size="sm" tone={t.count ? "neutral" : "warn"}>{formatCount(t.count)} stars</Badge>
              </span>
            </Tooltip>
          ) : <Badge size="sm">…</Badge>}
          {stale && <Freshness at={syncedAt} label="catalogue synced" stale={STALE_S} />}
        </div>
        <BarActions>
          {star && (
            <>
              <Button size="sm" variant="ghost" icon="info" aria-label="Details" title="The star's catalogue entry"
                onClick={() => openInspector({ kind: "star", id: String(star.id) })}>Details</Button>
              <SkyButton ra={star.ra} dec={star.dec} fov={0.02} layers={["stars", "q1-tiles:0.3"]} inspect={`star:${star.id}`} />
            </>
          )}
          <LinkButton to="/data/catalog?cut=nav" icon="table" label="Catalogue" hint="The navigator's stars in the catalogue" />
        </BarActions>
      </DataBar>
      {totals.error && !t && <Callout tone="bad" title="Totals did not load"><span className="dt-pre">{totals.error.message}</span></Callout>}
      {noCatalog ? (
        <EmptyState icon="database" title="No synchronised star catalogue"
          action={<Button asChild variant="primary"><Link to="/data/catalog">Open the catalogue</Link></Button>}>
          The navigator reads the FASRC catalogue mirror — pull it on the Catalog tab{online ? "" : ` (${OFFLINE_HINT})`}.
        </EmptyState>
      ) : (
        <div className="dt-vsplit">
          <div className="dt-vsplit__viewer">
            <ImageViewer collection="cutouts" urlKey="cut" onReady={(a) => { api.current = a; }}
              onState={onViewerState} />
          </div>
          <div className="dt-vsplit__aside" role="region" aria-label="Cached cutouts" data-fill="">
            <CutoutGallery current={current} navSize={t?.size ?? null} onPick={pick} />
          </div>
        </div>
      )}
      <Section title="Download Euclid star cutouts (FASRC)" collapsible open={steps} onOpenChange={setSteps}
        right={<LinkButton to="/settings/connections" icon="settings" label="Archive login" />}>
        <StepById stepId="download_euclid_cutouts" />
      </Section>
    </Page>
  );
}
