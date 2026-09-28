/* "What covers this point" (`source:at/<ra>,<dec>`, `GET /api/sky/at`):
 * the Q1 verdict and containing MER tiles, the real tiles / NEXUS tiles /
 * pairs / discovered JWST footprints there, and the actions that make
 * something new at this position. */
import { openInspector } from "../../../../app/inspector";
import { useResource } from "../../../../api/query";
import { formatNumber } from "../../../../format";
import { Badge, Button, Callout, DefList, Section, Skeleton, type MenuItem, type Tone } from "../../../../ui";
import { cacheTileAt, discoverJwst, downloadPair } from "../actions";
import { useFlyTo } from "../engineHooks";
import { atlasTarget, type RawInspect } from "../layerModel";
import { fmtCoord } from "../urlState";
import { CardActions, MoreMenu, PositionValue, positionMenuItems } from "./common";

type Q1Tile = { tile: string; field?: string | null; region?: string | null; levels_e?: number[] | null; rejected?: string | null; margin_arcsec?: number };
type Hit = { source?: string; id?: string; ref?: string; label?: string; state?: string; has_jwst?: boolean; inspect?: RawInspect };
type Footprint = { obs_id: string; instrument?: string; filters?: string; target?: string };

export type AtResponse = {
  ra: number; dec: number; field: string | null; in_q1: boolean; q1_observed: boolean;
  q1_verdict: "observed" | "unobserved" | "outside"; q1_tiles: Q1Tile[]; best_tile: string | null;
  real_tiles: Hit[]; nexus: Hit[]; pairs: Hit[]; jwst: Footprint[]; jwst_discovered: boolean;
};

const VERDICT_TONE: Record<string, Tone> = { observed: "good", unobserved: "warn", outside: "neutral" };

function HitList({ hits }: { hits: Hit[] }) {
  return (
    <ul className="sky-card__hits">
      {hits.map((h, i) => {
        const target = atlasTarget(h.inspect ?? (h.ref ? { kind: "realtile", id: h.ref } : null));
        return (
          <li key={h.ref ?? i}>
            <button type="button" className="sky-card__hit" disabled={!target} onClick={() => target && openInspector(target)}>
              <span>{h.label ?? h.ref}</span>
              {h.state && <Badge size="sm" tone={h.state === "current" ? "good" : h.state === "stale" ? "warn" : "neutral"}>{h.state}</Badge>}
              {h.has_jwst && <Badge size="sm" tone="info">JWST</Badge>}
            </button>
          </li>
        );
      })}
    </ul>
  );
}

export function PointCard({ ra, dec }: { ra: number; dec: number }) {
  const at = useResource<AtResponse>(`/api/sky/at?ra=${fmtCoord(ra)}&dec=${fmtCoord(dec)}`, [], { ttl: 30_000 });
  const fly = useFlyTo();
  if (at.loading) return <Skeleton lines={5} />;
  const d = at.data;
  if (!d) {
    return (
      <Callout tone="bad" title="Could not check this position" action={<Button size="sm" onClick={at.reload}>Retry</Button>}>
        {at.error?.message ?? "No data."}
      </Callout>
    );
  }
  const more: MenuItem[] = [
    { label: "Cache a tile + run production & mean…", disabled: d.q1_verdict === "outside", onSelect: () => { void cacheTileAt(ra, dec, { run: true }); } },
    { label: "Discover JWST around here…", onSelect: () => { void discoverJwst({ region: `${fmtCoord(ra)},${fmtCoord(dec)},0.2`, label: "a 0.2° cone here" }); } },
    { type: "separator" },
    ...positionMenuItems(ra, dec),
  ];
  const tiles = [...d.real_tiles, ...d.nexus.filter((n) => !d.real_tiles.some((r) => r.ref === n.ref))];
  return (
    <div className="sky-card">
      <div className="sky-card__badges">
        <Badge tone={VERDICT_TONE[d.q1_verdict] ?? "neutral"} dot>Q1 {d.q1_verdict}</Badge>
        {d.field && <Badge>{d.field}</Badge>}
        {d.best_tile && <Badge>tile {d.best_tile}</Badge>}
      </div>
      <CardActions>
        <Button size="sm" icon="globe" onClick={() => fly({ ra, dec })}>Show on sky</Button>
        <Button size="sm" variant="primary" disabled={d.q1_verdict === "outside"} onClick={() => { void cacheTileAt(ra, dec); }}>Cache a 25.6″ tile</Button>
        <Button size="sm" onClick={() => { void downloadPair({ ra, dec }); }}>JWST pair</Button>
        <MoreMenu items={more} />
      </CardActions>
      <DefList dense items={[["position", <PositionValue ra={ra} dec={dec} />]]} />
      <Section title="Q1 MER tiles" sub={String(d.q1_tiles.length)} collapsible defaultOpen>
        {d.q1_tiles.length === 0 ? <p className="muted">Outside every committed Q1 polygon.</p> : (
          <ul className="sky-card__models">
            {d.q1_tiles.map((t) => (
              <li key={t.tile}>
                <Badge size="sm" tone={t.rejected ? "warn" : "good"} dot>{t.rejected ? "rejected" : "observed"}</Badge>
                <code className="mono">{t.tile}</code>
                <span className="muted">{t.field ?? t.region ?? ""}{t.levels_e ? ` · VIS sky ${formatNumber(t.levels_e[0], { digits: 1 })} e⁻` : ""}{t.margin_arcsec != null ? ` · ${formatNumber(t.margin_arcsec, { digits: 0 })}″ from edge` : ""}</span>
              </li>
            ))}
          </ul>
        )}
      </Section>
      <Section title="Real tiles here" sub={String(tiles.length + d.pairs.length)} collapsible defaultOpen>
        {tiles.length + d.pairs.length === 0 ? <p className="muted">No cached tile, NEXUS tile or pair covers this point.</p> : <HitList hits={[...tiles, ...d.pairs]} />}
      </Section>
      <Section title="JWST" sub={d.jwst_discovered ? `${d.jwst.length} footprints` : "not discovered"} collapsible defaultOpen={d.jwst.length > 0}>
        {!d.jwst_discovered ? (
          <p className="muted">No JWST discovery has been run here (JWST tools › Discover).</p>
        ) : d.jwst.length === 0 ? <p className="muted">No discovered JWST imaging contains this point.</p> : (
          <ul className="sky-card__models">
            {d.jwst.map((j) => (
              <li key={j.obs_id}>
                <code className="mono">{j.obs_id}</code>
                <span className="muted">{[j.instrument, j.filters, j.target].filter(Boolean).join(" · ")}</span>
                <Button size="sm" variant="ghost" onClick={() => { void downloadPair({ obs_id: j.obs_id }); }}>Pair</Button>
              </li>
            ))}
          </ul>
        )}
      </Section>
    </div>
  );
}
