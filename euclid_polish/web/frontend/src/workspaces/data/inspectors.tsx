/* Data workspace inspector kinds (registered by ./register.ts):
 *   star:<id>                   a star of the FASRC-mirror catalogue
 *   truth:<split>/<index>/<row> a synthetic truth source of a training record
 *   psf:<cluster index>         a spatial ePSF cluster
 *   tng:<subhalo id>            a TNG50 SKIRT-atlas galaxy */
import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useResource } from "../../api/query";
import { openInspector, type InspectorProps } from "../../app/inspector";
import { formatDeg, formatNumber, formatRaDec } from "../../format";
import { Badge, Button, CopyButton, DefList, EmptyState, JsonTree, Section, Skeleton, Tooltip } from "../../ui";
import { URLS, usePsfInventory, useStars, useTngProperties, type SourceDetail } from "./api";
import { LoadState, SkyButton } from "./common";
import {
  BAND_STATE_HELP, BAND_STATE_TONE, BANDS, bandShort, bandState, clusterObjectId, decodeStars, decodeTng,
  formatCompact, parseClusterId, parseTruthId, recordObjectId, tngValue,
} from "./model";
import "./data.css";

const sci = (v: number | null | undefined) => (v == null ? "—" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0)
  ? v.toExponential(3) : formatNumber(v, { digits: 3 }));

/* ── star ──────────────────────────────────────────────────────────────── */

export function StarInspector({ id }: InspectorProps) {
  const stars = useStars();
  const star = useMemo(() => decodeStars(stars.data).find((s) => String(s.id) === id) ?? null, [stars.data, id]);
  if (stars.loading && !stars.data) return <Skeleton lines={6} />;
  if (!star) {
    return (
      <LoadState loading={false} error={stars.error} onRetry={() => void stars.reload()}>
        <EmptyState compact icon="info" title={`Star ${id} is not in the synchronised catalogue`} />
      </LoadState>
    );
  }
  return (
    <div className="dt-insp">
      <div className="dt-insp__head">
        <span className="dt-insp__title">Star {star.id}</span>
        {star.field && <Badge size="sm">{star.field}</Badge>}
        {star.nav && <Badge size="sm" tone="good">navigator</Badge>}
      </div>
      <DefList dense items={[
        ["RA, Dec", <span key="p" className="mono">{formatRaDec(star.ra, star.dec)}</span>],
        ["degrees", <span key="d" className="mono">{formatDeg(star.ra, 6)} {formatDeg(star.dec, 6, { signed: true })}</span>],
        ["VIS mag", formatNumber(star.mag, { digits: 3 })],
        ["PSF flux", star.flux != null ? `${formatNumber(star.flux, { digits: 2 })} ± ${formatNumber(star.fluxErr, { digits: 3 })} µJy` : "—"],
      ]} />
      <Section title="Cutouts per band">
        <DefList dense items={BANDS.map((b) => {
          const band = star.bands[b];
          const st = bandState(band);
          return [bandShort(b), (
            <span key={b} className="dt-chips">
              <Tooltip content={BAND_STATE_HELP[st]}><span tabIndex={0}><Badge size="sm" tone={BAND_STATE_TONE[st]}>{st}</Badge></span></Tooltip>
              {band?.sizes.length ? <span className="mono">{band.sizes.join(", ")} px</span> : null}
              {band?.corrupted && st === "valid" && <Badge size="sm" tone="warn">also rejected once</Badge>}
            </span>
          )];
        })} />
      </Section>
      <div className="dt-insp__actions">
        {star.nav && (
          <Button asChild size="sm" icon="image"><Link to={`/data/cutouts?v.cut.id=${star.id}`}>Cutouts</Link></Button>
        )}
        <SkyButton ra={star.ra} dec={star.dec} fov={0.02} layers={["stars", "q1-tiles:0.3"]} />
        <CopyButton value={() => `${star.ra} ${star.dec}`} label="Copy the coordinates" />
      </div>
    </div>
  );
}

/* ── synthetic truth source ────────────────────────────────────────────── */

const KEY_FIELDS: [string, string, (v: unknown) => string][] = [
  ["x_pix", "x (HR px)", (v) => formatNumber(v as number, { digits: 2 })],
  ["y_pix", "y (HR px)", (v) => formatNumber(v as number, { digits: 2 })],
  ["flux_vis_e", "VIS e⁻", (v) => (typeof v === "number" ? formatCompact(v) : "—")],
  ["flux_y_e", "Y e⁻", (v) => (typeof v === "number" ? formatCompact(v) : "—")],
  ["flux_j_e", "J e⁻", (v) => (typeof v === "number" ? formatCompact(v) : "—")],
  ["flux_h_e", "H e⁻", (v) => (typeof v === "number" ? formatCompact(v) : "—")],
  ["mag_vis", "VIS mag", (v) => formatNumber(v as number, { digits: 3 })],
  ["achieved_vis_2fwhm_mag", "VIS 2FWHM mag", (v) => formatNumber(v as number, { digits: 3 })],
  ["target_vis_mag", "target VIS mag", (v) => formatNumber(v as number, { digits: 3 })],
  ["re_arcsec", "Rₑ ″", (v) => formatNumber(v as number, { digits: 4 })],
  ["theta_E_arcsec", "θE ″", (v) => formatNumber(v as number, { digits: 3 })],
  ["z", "z", (v) => formatNumber(v as number, { digits: 3 })],
  ["logmass", "log M★", (v) => formatNumber(v as number, { digits: 2 })],
  ["sfr_class", "SFR class", (v) => String(v ?? "—")],
  ["temperature_k", "T (K)", (v) => formatNumber(v as number, { digits: 0 })],
];

export function TruthInspector({ id }: InspectorProps) {
  const target = parseTruthId(id);
  const res = useResource<SourceDetail>(target ? URLS.source(target.split, target.index, target.row) : null, [id]);
  if (!target) return <EmptyState compact icon="info" title={`Bad truth-source id "${id}"`} />;
  const d = res.data;
  return (
    <LoadState loading={res.loading && !d} error={res.error} onRetry={() => void res.reload()} lines={8}>
      {d && (
        <div className="dt-insp">
          <div className="dt-insp__head">
            <Badge tone={d.source.type === "star" ? "warn" : d.source.type === "lens" ? "accent" : "info"}>{d.source.type}</Badge>
            <span className="dt-insp__title">{target.split} · record {target.index} · #{target.row}</span>
            {d.source.off_field && <Badge size="sm">off-field</Badge>}
            {d.source.render && <Badge size="sm">{d.source.render}</Badge>}
          </div>
          <DefList dense items={KEY_FIELDS.filter(([k]) => d.values[k] != null).map(([k, label, f]) => [label, f(d.values[k])])} />
          <div className="dt-insp__actions">
            <Button asChild size="sm" icon="image">
              <Link to={`/data/records?split=${target.split}&v.rec.id=${encodeURIComponent(recordObjectId(target.split, target.index))}`}>Record</Link>
            </Button>
            {d.source.subhalo_id && (
              <Button size="sm" variant="ghost" onClick={() => openInspector({ kind: "tng", id: String(Number(d.source.subhalo_id)) })}>
                TNG {d.source.subhalo_id}
              </Button>
            )}
          </div>
          <Section title="All columns" collapsible defaultOpen={false}>
            <JsonTree data={d.values} expandDepth={1} />
          </Section>
        </div>
      )}
    </LoadState>
  );
}

/* ── PSF cluster ───────────────────────────────────────────────────────── */

export function PsfInspector({ id }: InspectorProps) {
  const inv = usePsfInventory();
  const index = parseClusterId(id);
  const cluster = inv.data?.clusters.find((c) => c.index === index) ?? null;
  if (inv.loading && !inv.data) return <Skeleton lines={5} />;
  if (!cluster) {
    return (
      <LoadState loading={false} error={inv.error} onRetry={() => void inv.reload()}>
        <EmptyState compact icon="info" title={`No PSF cluster ${id} in the cache`} />
      </LoadState>
    );
  }
  return (
    <div className="dt-insp">
      <div className="dt-insp__head"><span className="dt-insp__title">{clusterObjectId(cluster.index)}</span></div>
      <DefList dense items={[
        ["RA, Dec", <span key="p" className="mono">{formatRaDec(cluster.ra, cluster.dec, { mode: "both", digits: 4 })}</span>],
        ["stars", formatNumber(cluster.n_stars, { digits: 0 })],
        ...BANDS.map((b): [string, string] => [`FWHM ${bandShort(b)}`, cluster.fwhm_by_band[b] != null
          ? `${formatNumber(cluster.fwhm_by_band[b], { digits: 3 })}″` : "—"]),
      ]} />
      <div className="dt-insp__actions">
        <Button asChild size="sm" icon="image"><Link to={`/data/psfs?v.psf.id=${clusterObjectId(cluster.index)}`}>View</Link></Button>
        <SkyButton ra={cluster.ra} dec={cluster.dec} fov={0.6} layers={["psf-clusters", "q1-tiles:0.3"]} />
      </div>
    </div>
  );
}

/* ── TNG galaxy ────────────────────────────────────────────────────────── */

export function TngInspector({ id }: InspectorProps) {
  const props = useTngProperties();
  const row = useMemo(() => decodeTng(props.data).find((r) => String(r.id) === id) ?? null, [props.data, id]);
  if (props.loading && !props.data) return <Skeleton lines={6} />;
  if (!row) {
    return (
      <LoadState loading={false} error={props.error} onRetry={() => void props.reload()}>
        <EmptyState compact icon="info" title={`TNG galaxy ${id} is not in the local property cache`} />
      </LoadState>
    );
  }
  const views = props.data?.orientations[id] ?? [];
  return (
    <div className="dt-insp">
      <div className="dt-insp__head">
        <span className="dt-insp__title">TNG50 subhalo {row.id}</span>
        {row.sfr === 0 && <Badge size="sm">quenched</Badge>}
        {row.local > 0 && <Badge size="sm" tone="info">{row.local} local frames</Badge>}
      </div>
      <DefList dense items={[
        ["SFR", `${sci(row.sfr)} M☉/yr`],
        ["sSFR", `${sci(tngValue(row, "ssfr"))} /yr`],
        ["M★", `${sci(row.mass_stars)} M☉`],
        ["M halo (bound)", `${sci(row.m_halo)} M☉`],
        ["r½ (catalogue)", `${formatNumber(row.reff, { digits: 3 })} kpc`],
        ["Rₑ measured", row.re_kpc != null ? `${formatNumber(row.re_kpc, { digits: 3 })} kpc (mean of ${row.n_orient})` : "—"],
      ]} />
      {views.length > 0 && (
        <Section title="Viewpoints">
          <DefList dense items={views.map(([o, px, kpc]) => [`view ${o ?? "?"}`, (
            <span key={String(o)} className="dt-chips">
              <span className="mono">Rₑ {formatNumber(px, { digits: 1 })} px · {formatNumber(kpc, { digits: 3 })} kpc</span>
              {row.local > 0 && o != null && (
                <Button size="sm" variant="ghost" icon="fileSearch"
                  onClick={() => openInspector({ kind: "fits", id: `data/tng_skirt/${row.id}/TNG${row.id}_O${o}_Euclid_VIS.fits` })}>
                  VIS
                </Button>
              )}
            </span>
          )])} />
        </Section>
      )}
    </div>
  );
}
