/* Home's small static sky: an all-sky Mollweide outline (RA grows to the
 * left, the galactic plane dashed) with the Euclid Q1 deep-field cones
 * (/api/sky/layer/q1-fields) and the NEXUS mosaic
 * (/api/sky/layer/nexus-footprint), plus a legend of the same targets. No
 * Aladin here: every shape and legend row links into the Sky atlas centred on
 * it (`/sky/atlas?ra&dec&fov`). Colours are tokens (home.css). */
import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useResource } from "../../api/query";
import { formatDeg } from "../../format";
import { Callout, Skeleton } from "../../ui";
import {
  SKY_H, SKY_W, circlePath, fieldLabel, galacticPlane, graticule, mollweide, polygonPath, toSvg,
} from "./skyProjection";

type Circle = { id: string; ra: number; dec: number; radius_deg: number; props?: { name?: string } };
type Polygon = { id: string; polygon: number[][]; props?: { tiles?: number; filter?: string | null } & Record<string, unknown> };
type CirclesLayer = { features?: Circle[] };
type PolygonLayer = { features?: Polygon[] };

export const atlasLink = (ra: number, dec: number, fov: number) =>
  `/sky/atlas?ra=${Number(ra.toFixed(4))}&dec=${Number(dec.toFixed(4))}&fov=${fov}`;

function centroid(polygon: number[][]): [number, number] | null {
  const pts = polygon.filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (!pts.length) return null;
  return [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length];
}

const coords = (ra: number, dec: number) => `${formatDeg(ra, 1)}, ${formatDeg(dec, 1, { signed: true })}`;

export function SkyOverview() {
  const fields = useResource<CirclesLayer>("/api/sky/layer/q1-fields", [], { ttl: 10 * 60_000 });
  const nexus = useResource<PolygonLayer>("/api/sky/layer/nexus-footprint", [], { ttl: 10 * 60_000 });
  const grid = useMemo(() => graticule(), []);
  const plane = useMemo(() => galacticPlane(), []);
  const circles = fields.data?.features ?? [];
  const hull = (nexus.data?.features ?? []).find((f) => !String(f.id).endsWith(":mosaic")) ?? null;
  const nexusCentre = hull ? centroid(hull.polygon) : null;
  const nexusTiles = typeof hull?.props?.tiles === "number" ? hull.props.tiles : null;

  if (fields.loading && !fields.data) return <Skeleton height={170} />;
  if (fields.error && !fields.data) {
    return <Callout tone="bad" title="Sky layers unavailable">{fields.error.message}</Callout>;
  }
  const nexusSvg = nexusCentre ? toSvg(mollweide(nexusCentre[0], nexusCentre[1])) : null;
  return (
    <figure className="home-sky">
      <svg viewBox={`0 0 ${SKY_W} ${SKY_H}`} className="home-sky__svg" role="img"
        aria-label={`All-sky map: ${circles.map((c) => c.props?.name ?? c.id).join(", ") || "no fields"}${nexusCentre ? ", NEXUS" : ""}`}>
        <path d={grid.outline} className="home-sky__outline" />
        {grid.meridians.map((d, i) => <path key={`m${i}`} d={d} className="home-sky__grid" />)}
        {grid.parallels.map((d, i) => <path key={`p${i}`} d={d} className="home-sky__grid" />)}
        <path d={plane} className="home-sky__plane"><title>Galactic plane</title></path>
        {circles.map((c) => {
          const name = c.props?.name ?? c.id;
          const label = fieldLabel(c.ra, c.dec, c.radius_deg);
          return (
            <Link key={c.id} to={atlasLink(c.ra, c.dec, Math.max(2, 2.5 * c.radius_deg))} className="home-sky__field"
              aria-label={`${name} on the sky atlas (RA ${formatDeg(c.ra, 2)}, Dec ${formatDeg(c.dec, 2, { signed: true })})`}>
              <title>{`${name} · RA ${c.ra.toFixed(1)}°, Dec ${c.dec.toFixed(1)}°`}</title>
              <path d={circlePath(c.ra, c.dec, c.radius_deg)} />
              <text x={label.x} y={label.y} textAnchor={label.anchor} className="home-sky__label">{name}</text>
            </Link>
          );
        })}
        {hull && nexusCentre && nexusSvg && (
          <Link to={atlasLink(nexusCentre[0], nexusCentre[1], 0.6)} className="home-sky__nexus"
            aria-label="NEXUS F200W mosaic on the sky atlas">
            <title>NEXUS F200W mosaic (JWST × Euclid tiles)</title>
            <path d={polygonPath(hull.polygon)} />
            <circle cx={nexusSvg.x} cy={nexusSvg.y} r={3.4} />
            {/* inside EDF-N, whose name sits on the other side of the cone */}
            <text x={nexusSvg.x + (nexusSvg.x >= SKY_W / 2 ? -7 : 7)} y={nexusSvg.y + 4}
              textAnchor={nexusSvg.x >= SKY_W / 2 ? "end" : "start"} className="home-sky__label home-sky__label--nexus">NEXUS</text>
          </Link>
        )}
      </svg>
      <ul className="home-sky__legend" aria-label="Sky targets">
        {circles.map((c) => (
          <li key={c.id}>
            <Link to={atlasLink(c.ra, c.dec, Math.max(2, 2.5 * c.radius_deg))} className="home-sky__item">
              <span className="home-sky__swatch" data-kind="field" aria-hidden="true" />
              <span className="home-sky__name">{c.props?.name ?? c.id}</span>
              <span className="home-sky__coord">{coords(c.ra, c.dec)}</span>
            </Link>
          </li>
        ))}
        {nexusCentre && (
          <li>
            <Link to={atlasLink(nexusCentre[0], nexusCentre[1], 0.6)} className="home-sky__item">
              <span className="home-sky__swatch" data-kind="nexus" aria-hidden="true" />
              <span className="home-sky__name">NEXUS</span>
              <span className="home-sky__coord">{nexusTiles != null ? `${nexusTiles} tiles · ` : ""}{coords(nexusCentre[0], nexusCentre[1])}</span>
            </Link>
          </li>
        )}
      </ul>
    </figure>
  );
}
