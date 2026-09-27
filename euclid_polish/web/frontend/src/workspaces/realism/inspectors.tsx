/* Realism inspector kinds (registered by ./register.ts). */
import { Link } from "react-router-dom";
import type { InspectorProps } from "../../app/inspector";
import { useFasrcStatus } from "../../app/status";
import { bandColor } from "../../colors";
import { formatDeg, formatNumber, formatRaDec } from "../../format";
import { Badge, Button, DefList, EmptyState, JsonTree, Section, Skeleton } from "../../ui";
import { ImageViewer } from "../../viewer";
import { useArchiveMeta, useNoisePosition, useOverview, type OverviewItem } from "./api";
import { LoadState, SkyLink, StateDot, STATE_TONE, Swatch } from "./common";
import { keyForUrl, offlinePolicy, runAction, useRealismJob, type JobKey } from "./jobs";
import "./realism.css";

const bandShort = (band: string) => band.replace("_E", "");

function scalarFacts(facts: Record<string, unknown>): [string, string][] {
  return Object.entries(facts)
    .filter(([, v]) => v == null || ["string", "number", "boolean"].includes(typeof v))
    .map(([k, v]) => [k.replace(/_/g, " "), v == null ? "—" : typeof v === "number" ? formatNumber(v) : String(v)]);
}

function ItemFix({ item }: { item: OverviewItem }) {
  const action = item.action!;
  const job = useRealismJob(keyForUrl(action.url) as JobKey);
  const fasrc = useFasrcStatus().data;
  const policy = offlinePolicy(action, fasrc ? !fasrc.ssh_connected : false);
  return (
    <Button size="sm" loading={job.busy} disabled={policy.disabled} title={policy.hint ?? undefined}
      onClick={() => void runAction(action)}>
      {policy.disabled ? `${action.label} (FASRC offline)` : action.label}
    </Button>
  );
}

export function ReadinessInspector({ id }: InspectorProps) {
  const overview = useOverview();
  const item = overview.data?.items.find((i) => i.id === id);
  if (overview.loading && !overview.data) return <Skeleton lines={5} />;
  if (!item) {
    return (
      <LoadState loading={false} error={overview.error} onRetry={overview.reload}>
        <EmptyState compact icon="info" title={`No readiness item "${id}"`} />
      </LoadState>
    );
  }
  const nested = Object.fromEntries(Object.entries(item.facts).filter(([, v]) => v != null && typeof v === "object"));
  return (
    <div className="rl-insp">
      <div className="rl-insp__head">
        <StateDot state={item.state} />
        <div>
          <div className="rl-check__label">{item.label}</div>
          <div className="rl-insp__title">{item.title}</div>
        </div>
        <Badge size="sm" tone={STATE_TONE[item.state]}>{item.state}</Badge>
      </div>
      {item.detail && <p className="rl-insp__detail">{item.detail}</p>}
      <div className="rl-row">
        {item.action && <ItemFix item={item} />}
        {item.to && <Button asChild size="sm" variant="ghost" iconRight="chevronRight"><Link to={item.to}>Open</Link></Button>}
      </div>
      <Section title="Facts">
        <DefList dense items={scalarFacts(item.facts)} />
        {Object.keys(nested).length > 0 && <JsonTree data={nested} expandDepth={1} />}
      </Section>
    </div>
  );
}

/** A 4×4 sub-tile grid, each cell shaded by its level relative to the grid's
 *  median (token colour mixed over the surface). */
function SubGrid({ band, values, side }: { band: string; values: (number | null)[]; side: number }) {
  const finite = values.filter((v): v is number => v != null && Number.isFinite(v)).sort((a, b) => a - b);
  const median = finite.length ? finite[Math.floor(finite.length / 2)] : 1;
  const spread = Math.max(1e-9, Math.max(...finite.map((v) => Math.abs(v / median - 1)), 0.05));
  return (
    <div className="rl-subgrid" style={{ gridTemplateColumns: `repeat(${side}, 1fr)`, ["--sw" as string]: bandColor(band) }}
      role="table" aria-label={`${bandShort(band)} sub-tile levels`}>
      {values.map((v, i) => {
        const rel = v == null ? null : v / median - 1;
        const pct = rel == null ? 0 : Math.round(12 + 70 * Math.min(1, Math.abs(rel) / spread));
        return (
          <span key={i} role="cell" className="rl-subgrid__cell" data-sign={rel != null && rel < 0 ? "neg" : "pos"}
            style={{ ["--mix" as string]: `${pct}%` }} title={v == null ? "no data" : `${formatNumber(v, { digits: 2 })} e⁻`}>
            {v == null ? "—" : formatNumber(v, { digits: v >= 10 ? 1 : 2 })}
          </span>
        );
      })}
    </div>
  );
}

export function NoisePositionInspector({ id }: InspectorProps) {
  const position = useNoisePosition(id);
  const p = position.data;
  return (
    <LoadState loading={position.loading && !p} error={position.error} onRetry={position.reload}>
      {p && (
        <div className="rl-insp">
          <DefList dense items={[
            ["field", p.field],
            ["tile", <span className="rl-mono" key="t">{p.tile}</span>],
            ["RA, Dec", formatRaDec(p.ra, p.dec, { mode: "both" })],
            ["noise model", <span className="rl-mono" key="m">{p.noise_model}</span>],
          ]} />
          <div className="rl-levels" aria-label="Band levels">
            {p.bands.map((band) => {
              const step = p.steps[band];
              return (
                <div key={band} className="rl-levels__band">
                  <div className="rl-levels__head">
                    <Swatch color={bandColor(band)} />
                    <strong>{bandShort(band)}</strong>
                    <span className="rl-num">{formatNumber(p.levels_e[band], { digits: 2 })} e⁻</span>
                    {step && (
                      <Badge size="sm" tone={step.seam ? "warn" : undefined}
                        title={`largest straight-line step ×${step.step.toFixed(2)}; uniformity ${step.scatter.toFixed(2)}`}>
                        {step.seam ? "seam " : "step "}×{step.step.toFixed(2)}
                      </Badge>
                    )}
                  </div>
                  {p.sub_levels_e?.[band] && p.grid_side && (
                    <SubGrid band={band} values={p.sub_levels_e[band]} side={p.grid_side} />
                  )}
                </div>
              );
            })}
          </div>
          <div className="rl-row">
            <SkyLink layers={["q1-tiles:0.3", "noise-positions"]} ra={p.ra} dec={p.dec} fov={0.5}
              inspect={`source:noise-positions/${p.tile}`}>Open on sky</SkyLink>
            <span className="rl-faint">{formatDeg(p.ra, 4)}, {formatDeg(p.dec, 4, { signed: true })}</span>
          </div>
        </div>
      )}
    </LoadState>
  );
}

export function ArchiveFieldInspector({ id }: InspectorProps) {
  const meta = useArchiveMeta();
  const index = meta.data?.objects?.findIndex((o) => (o.id ?? String(o.sample_id)) === id) ?? -1;
  const object = index >= 0 ? meta.data?.objects?.[index] : undefined;
  return (
    <LoadState loading={meta.loading && !meta.data} error={meta.error} onRetry={meta.reload}>
      {!object ? <EmptyState compact icon="image" title={`Archive field ${id} is not in the local collection`} /> : (
        <div className="rl-insp">
          <DefList dense items={[
            ["field", object.field],
            ["position", object.position_name],
            ["parent pointing", <span className="rl-mono" key="p">{object.parent_id}</span>],
            ["sample", `${object.sample_id + 1} · source pointing ${object.source_sample_id + 1}`],
            ["RA, Dec", formatRaDec(object.ra, object.dec, { mode: "both" })],
          ]} />
          <div className="rl-insp__viewer">
            <ImageViewer collection="archive-fields" initialIndex={index} id={`archivefield-${id}`}
              toolbar="compact" nav={false} />
          </div>
          <div className="rl-row">
            <Button asChild size="sm" variant="ghost" icon="image">
              <Link to={`/realism/visual?v.real.id=${encodeURIComponent(object.id ?? String(object.sample_id))}`}>Open in Visual</Link>
            </Button>
            <SkyLink layers={["archive-fields"]} ra={object.ra} dec={object.dec} fov={0.3}>Open on sky</SkyLink>
          </div>
        </div>
      )}
    </LoadState>
  );
}
