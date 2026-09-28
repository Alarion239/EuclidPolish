/* The `figure:<result id>` inspector card: a saved viewer result — every
 * panel it can render, its source and crop geometry, WCS and sky position,
 * and its actions (source viewer, sheet, sky, Files, FITS, rename, delete). */
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useResource } from "../../api/query";
import { formatBytes, formatDateTime, formatNumber, formatRaDec } from "../../format";
import { Badge, Button, DefList, EmptyState, Menu, Section, Skeleton } from "../../ui";
import { confirmDeleteResults, RenameDialog } from "./actions";
import { URLS, fitsUrl, panelUrl, type SavedResult } from "./api";
import { RegimeBadge, ServerImage, WcsBadge } from "./common";
import { cropSideArcsec, gridHref, inspectLink, normalizeIndex, recipeLabel, resultRegime, skyLink, sourceLabel, viewerLink } from "./model";
import "./figures.css";

export default function FigureInspector({ id }: { id: string }) {
  const res = useResource<{ result: unknown }>(URLS.result(id), [id], { ttl: 15_000 });
  const [recipe, setRecipe] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  const navigate = useNavigate();
  const result: SavedResult | undefined = res.data ? normalizeIndex({ results: [res.data.result] }).results[0] : undefined;
  if (res.loading && !res.data) return <Skeleton lines={6} />;
  if (res.error || !result) {
    return (
      <EmptyState icon="image" compact title="Saved result unavailable"
        action={<Button size="sm" onClick={() => void res.reload()}>Retry</Button>}>
        {res.error?.message ?? "The server sent no result."}
      </EmptyState>
    );
  }
  const shown = recipe && result.recipes.includes(recipe as never) ? recipe : result.thumbnail ?? result.recipes[0] ?? null;
  const link = viewerLink(result);
  const sky = skyLink(result);
  const regime = resultRegime(result) ?? "real";
  const side = cropSideArcsec(result);
  const obj = result.source?.object ?? {};
  const display = result.display ?? {};
  const tiers = result.logical_tiers;
  return (
    <div className="fig-insp">
      <div className="fig-insp__hero">
        <ServerImage src={shown ? panelUrl(result.id, shown, 512) : null} alt={`${result.label} · ${shown ? recipeLabel(shown) : ""}`}
          className="fig-insp__image" minHeight={200} paper={false} />
        <div className="fig-insp__title">
          <strong className="fig-ellipsis" title={result.label}>{result.label}</strong>
          <span className="fig-insp__badges">
            <RegimeBadge regime={resultRegime(result)} />
            <Badge size="sm" title={`viewer collection ${result.source?.collection ?? "—"}`}>{sourceLabel(result)}</Badge>
            <WcsBadge result={result} />
          </span>
        </div>
      </div>
      {result.recipes.length > 1 && (
        <div className="fig-insp__panels" role="group" aria-label="Panels">
          {result.recipes.map((k) => (
            <button key={k} type="button" className="fig-insp__panel" data-on={k === shown} aria-pressed={k === shown}
              title={recipeLabel(k)} onClick={() => setRecipe(k)}>
              <img src={panelUrl(result.id, k, 96)} alt="" loading="lazy" />
              <span>{recipeLabel(k)}</span>
            </button>
          ))}
        </div>
      )}
      <div className="fig-insp__actions">
        {link && <Button asChild size="sm" variant="primary"><Link to={link.to}>{link.label}</Link></Button>}
        <Button asChild size="sm" icon="columns"><Link to={gridHref([result.id], regime)}>Sheet</Link></Button>
        {sky && <Button asChild size="sm" icon="globe"><Link to={sky}>Sky</Link></Button>}
        <Menu label="FITS" items={tiers.flatMap((t) => [
          { label: `Open ${t}.fits in Files`, disabled: !inspectLink(result, t), onSelect: () => { const to = inspectLink(result, t); if (to) navigate(to); } },
          { label: `Download ${t}.fits`, onSelect: () => window.location.assign(fitsUrl(result.id, t)) },
        ])} trigger={<Button size="sm" icon="download">FITS</Button>} />
        <Button size="sm" variant="ghost" onClick={() => setRenaming(true)}>Rename</Button>
        <Button size="sm" variant="ghost" onClick={() => void confirmDeleteResults([result])}>Delete</Button>
      </div>
      <Section title="Source">
        <DefList dense items={[
          ["collection", result.source?.collection ?? "—"],
          !!obj.id && ["object", <span className="mono" key="o">{obj.id}</span>],
          !!obj.label && obj.label !== result.label && ["object label", obj.label],
          result.source?.index != null && ["index", String(result.source.index)],
          result.source?.params && Object.keys(result.source.params).length > 0
            && ["params", <span className="mono" key="p">{Object.entries(result.source.params).map(([k, v]) => `${k}=${v}`).join(" ")}</span>],
          ["saved", formatDateTime(result.created_utc)],
          ["id", <span className="mono" key="i">{result.id}</span>],
        ]} />
      </Section>
      <Section title="Crop">
        <DefList dense items={[
          side != null && ["side", `${formatNumber(side, { digits: 3 })}″`],
          result.center && ["centre", formatRaDec(result.center.ra, result.center.dec, { mode: "both" })],
          !!result.selection?.source_tier && ["anchored on", result.selection.source_tier],
          ...tiers.map((t) => {
            const f = result.files?.[t];
            const shape = f?.shape_hwc ? `${f.shape_hwc[1]}×${f.shape_hwc[0]}×${f.shape_hwc[2]}` : "";
            const scale = result.pixscale_arcsec[t];
            return [t, `${f?.source_label ?? f?.source_tier ?? t} · ${shape}${scale ? ` · ${formatNumber(scale, { digits: 3 })}″/px` : ""}${f?.wcs ? " · WCS" : ""}`] as [string, string];
          }),
          result.bytes != null && ["size", formatBytes(result.bytes)],
        ]} />
      </Section>
      <Section title="Display at save" collapsible defaultOpen={false}>
        <DefList dense items={Object.entries(display).filter(([k]) => k !== "transfers").map(([k, v]) => [k, String(v)] as [string, string])} />
      </Section>
      <RenameDialog result={result} open={renaming} onOpenChange={setRenaming} />
    </div>
  );
}
