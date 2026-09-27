/* Data › Cutouts (spec §8.4): real Euclid star cutouts.
 *
 * The navigator (viewer collection `cutouts`: the stars valid in all four
 * bands at one common size, from the synchronised FASRC-mirror catalogue —
 * works offline) with the current star's id / RA / Dec / magnitude and its
 * links (catalogue inspector, atlas); beside it the gallery of the cutouts
 * cached on this machine, per band, each labelled with its star (a click
 * opens that star in the navigator). The download_euclid_cutouts FASRC step
 * sits at the bottom; the archive credentials live in Settings › Connections.
 * The viewer object, gallery band and page are in the URL. */
import { useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useResource } from "../../../api/query";
import { openInspector } from "../../../app/inspector";
import { usePageActions } from "../../../app/palette";
import { StepById } from "../../../fasrc";
import { formatCount, formatNumber, formatRaDec } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, CopyButton, EmptyState, Gallery, IconButton, Page, Section,
  Segmented, Skeleton, Tooltip, type GalleryItem,
} from "../../../ui";
import { ImageViewer, type ViewerApi, type ViewerState } from "../../../viewer";
import { URLS, useStars, type GalleryPage, type Totals } from "../api";
import { DataBar, Freshness, OFFLINE_HINT, SkyButton, Spacer, useFasrcOnline } from "../common";
import { BANDS, bandShort, decodeStars } from "../model";
import "../register";
import "../data.css";

const PER_PAGE = 48;

function CurrentStar({ id }: { id: string | null }) {
  const stars = useStars();
  const star = useMemo(() => (id ? decodeStars(stars.data).find((s) => String(s.id) === id) ?? null : null),
    [stars.data, id]);
  if (!id) return null;
  if (!star) return <div className="dt-facts"><span className="mono">star {id}</span></div>;
  return (
    <div className="dt-facts" aria-label={`Star ${star.id}`}>
      <strong>star {star.id}</strong>
      {star.field && <Badge size="sm">{star.field}</Badge>}
      <span className="mono">{formatRaDec(star.ra, star.dec, { mode: "both" })}</span>
      <CopyButton value={() => `${star.ra} ${star.dec}`} label="Copy the coordinates" />
      <span>VIS {formatNumber(star.mag, { digits: 3 })}</span>
      <Button size="sm" variant="ghost" icon="info" onClick={() => openInspector({ kind: "star", id: String(star.id) })}>Details</Button>
      <SkyButton ra={star.ra} dec={star.dec} fov={0.02} layers={["stars", "q1-tiles:0.3"]} inspect={`star:${star.id}`} />
    </div>
  );
}

function CutoutGallery({ onPick }: { onPick: (id: number) => void }) {
  const [band, setBand] = useUrlState("gband", "VIS");
  const [page, setPage] = useUrlState("gpage", 1);
  const res = useResource<GalleryPage>(URLS.gallery(band, page, PER_PAGE), [band, page], { ttl: 60_000 });
  const data = res.data;
  const items: GalleryItem[] = (data?.items ?? []).map((it) => ({
    src: URLS.cutoutImage(band, it.file, 170, data?.output_dir ?? ""),
    label: `${it.id ?? it.file}${it.size ? ` · ${it.size}px` : ""}${it.mag != null ? ` · ${it.mag.toFixed(2)}` : ""}`,
    onClick: () => { if (it.id != null) onPick(it.id); },
  }));
  return (
    <Card>
      <CardHead title="Cached cutouts" sub={data ? `${formatCount(data.total)} ${bandShort(band)} files on this machine` : undefined}
        right={<Segmented size="sm" value={band} onChange={(b) => { setBand(b); setPage(1); }} aria-label="Band"
          options={BANDS.map((b) => ({ value: b, label: bandShort(b) }))} />} />
      <CardBody>
        {res.loading && !data ? <Skeleton lines={4} />
          : res.error ? <Callout tone="bad" title="Gallery did not load"><span className="dt-pre">{res.error.message}</span></Callout>
            : <Gallery items={items} thumb={112}
                empty={`No ${bandShort(band)} cutouts cached yet — browsing a star in the navigator pulls its four cutouts.`} />}
        {(data?.n_pages ?? 1) > 1 && (
          <div className="dt-pager">
            <IconButton icon="chevronLeft" size="sm" label="Previous page" disabled={page <= 1} onClick={() => setPage(page - 1)} />
            <span className="mono muted">page {data?.page} / {data?.n_pages}</span>
            <IconButton icon="chevronRight" size="sm" label="Next page" disabled={page >= (data?.n_pages ?? 1)} onClick={() => setPage(page + 1)} />
          </div>
        )}
      </CardBody>
    </Card>
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
  return (
    <Page className="dt-page">
      <DataBar label="Cutouts">
        <span className="dt-bar__label">Navigator</span>
        {t ? (
          <Tooltip content="Stars valid in all four bands at one common cutout size, from the synchronised FASRC catalogue">
            <span tabIndex={0}><Badge size="sm" tone={t.count ? "good" : "warn"}>{formatCount(t.count)} stars{t.size ? ` @ ${t.size} px` : ""}</Badge></span>
          </Tooltip>
        ) : <Badge size="sm">…</Badge>}
        {t?.catalog.present && <Freshness at={t.catalog.mtime} label="catalogue synced" stale={3 * 24 * 3600} />}
        <Spacer />
        <Button asChild size="sm" variant="ghost" icon="table"><Link to="/data/catalog?cut=nav">Catalogue</Link></Button>
        <Button asChild size="sm" variant="ghost" icon="settings"><Link to="/settings/connections">Archive login</Link></Button>
      </DataBar>
      {totals.error && !t && <Callout tone="bad" title="Totals did not load"><span className="dt-pre">{totals.error.message}</span></Callout>}
      {noCatalog ? (
        <EmptyState icon="database" title="No synchronised star catalogue"
          action={<Button asChild variant="primary"><Link to="/data/catalog">Open the catalogue</Link></Button>}>
          The navigator reads the FASRC catalogue mirror — pull it on the Catalog tab{online ? "" : ` (${OFFLINE_HINT})`}.
        </EmptyState>
      ) : (
        <div className="dt-split">
          <Card className="dt-viewer-card">
            <CardBody>
              <CurrentStar id={current} />
              <ImageViewer collection="cutouts" urlKey="cut" onReady={(a) => { api.current = a; }}
                onState={(s: ViewerState) => setCurrent(s.id)} />
              {!online && <p className="dt-note">FASRC offline: stars whose cutouts are not cached yet cannot be shown.</p>}
            </CardBody>
          </Card>
          <CutoutGallery onPick={pick} />
        </div>
      )}
      <Section title="Download Euclid star cutouts (FASRC)" collapsible open={steps} onOpenChange={setSteps}>
        <StepById stepId="download_euclid_cutouts" />
      </Section>
    </Page>
  );
}
