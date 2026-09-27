/* NEXUS × Euclid comparison plates: render (a local job — tiles, band or
 * Temp colour, model spec) and browse the runs in output/nexus_comparisons
 * with their provenance (model identity and state, tile positions). The
 * run / render / enlarged tile live in the URL. */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useJob } from "../../../api/jobs";
import { invalidate, useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { formatBytes, formatDateTime, formatDeg, formatRaDec, formatRelative } from "../../../format";
import { useUrlState } from "../../../hooks/useUrlState";
import { useSelected } from "../../../state/selection";
import {
  Badge, Button, Callout, DefList, Dialog, EmptyState, Field, IconButton, Input, JobProgress, Segmented,
  Select, Skeleton, Tooltip, confirm, toast, type SelectOption,
} from "../../../ui";
import { URLS, deletePlateRun, plateFileUrl, type PlateRender, type PlatesIndex, type PlateTileRecord } from "../api";
import { ServerImage } from "../common";
import {
  findRender, parseTileList, plateCoverage, renderKey, renderTitle, specsCoveringAll, type NexusTile,
} from "../model";

type ModelRow = { spec: string; label: string; available?: boolean; reason?: string | null; kind?: string };
type ModelsPayload = { models: ModelRow[] };
type NexusList = { tiles: (NexusTile & { label?: string; ra?: number; dec?: number })[] };

const BAND_OPTIONS = [
  { value: "VIS", label: "VIS" }, { value: "Y_E", label: "Y" }, { value: "J_E", label: "J" },
  { value: "H_E", label: "H" }, { value: "temp", label: "Temp", title: "The viewer's temperature colour (LR and SR); NEXUS stays grey" },
];

function experimentsHref(refs: readonly string[], spec: string): string {
  const q = new URLSearchParams({ tiles: refs.join(","), models: spec });
  return `/sky/experiments?${q.toString()}`;
}

function stateTone(state: string | null | undefined): "good" | "warn" | "neutral" {
  return state === "current" ? "good" : state === "stale" ? "warn" : "neutral";
}

export function NexusPlates() {
  const plates = useResource<PlatesIndex>(URLS.plates, [], { ttl: 30_000 });
  const models = useResource<ModelsPayload>(URLS.models, [], { ttl: 120_000 });
  const nexus = useResource<NexusList>(URLS.nexusTiles, [], { ttl: 60_000 });
  const [run, setRun] = useUrlState<string>("run", "");
  const [render, setRender] = useUrlState<string>("render", "");
  const [tile, setTile] = useUrlState<string>("tile", "");
  const [tilesText, setTilesText] = useUrlState<string>("tiles", "");
  const [band, setBand] = useUrlState<string>("band", "VIS");
  const [spec, setSpec] = useUrlState<string>("model", "");
  const [tag, setTag] = useState("");
  const job = useJob("figures:nexus-plates");
  const selectedTiles = useSelected("tile");

  const runs = plates.data?.runs ?? [];
  const defaults = plates.data?.defaults;
  const text = tilesText || (defaults?.tiles ?? [40, 42, 70, 178]).join(", ");
  const tokens = useMemo(() => parseTileList(text), [text]);
  const nexusTiles = useMemo(() => nexus.data?.tiles ?? [], [nexus.data]);
  const specs = useMemo(() => (models.data?.models ?? []).map((m) => m.spec), [models.data]);
  const coverAll = useMemo(() => {
    const cov = plateCoverage(tokens, nexusTiles, "");
    return specsCoveringAll(cov.tiles, specs);
  }, [tokens, nexusTiles, specs]);
  const chosen = spec || coverAll[0] || "production";
  const coverage = useMemo(() => plateCoverage(tokens, nexusTiles, chosen), [tokens, nexusTiles, chosen]);
  const maxTiles = defaults?.max_tiles ?? 24;
  const tooMany = tokens.length > maxTiles;
  const canRender = !!tokens.length && !tooMany && !coverage.unknown.length && !coverage.missing.length && !job.busy;
  const nexusSelection = selectedTiles.filter((r) => r.startsWith("nexus/")).map((r) => r.slice("nexus/".length));

  const modelOptions: SelectOption[] = (models.data?.models ?? []).map((m) => {
    const n = plateCoverage(tokens, nexusTiles, m.spec);
    const have = n.tiles.length - n.missing.length;
    return { value: m.spec, label: `${m.label}${tokens.length ? ` · ${have}/${n.tiles.length}` : ""}`, hint: m.spec };
  });
  if (!modelOptions.some((o) => o.value === chosen)) modelOptions.unshift({ value: chosen, label: chosen });

  const current = runs.find((r) => r.tag === run) ?? runs[0];
  const shown: PlateRender | undefined = findRender(current, render);
  const enlarged = shown?.tiles.find((t) => String(t.index) === tile);

  const submit = () => {
    if (!canRender) return;
    void job.run(URLS.plates, { tiles: coverage.tiles.map((t) => t.id).join(","), band, model: chosen, tag: tag.trim() }, {
      onDone: (j) => {
        invalidate(URLS.plates);
        const result = j.result as { tag?: string; band?: string; model?: string } | null;
        if (j.status === "done" && result?.tag) {
          setRun(result.tag);
          setRender(`${result.band}~${result.model}`);
          toast.success(`Plates rendered into ${result.tag}`);
        } else if (j.status === "failed") toast.error(`Plates failed: ${String(j.error ?? "").split("\n")[0]}`);
      },
    });
  };
  const removeRun = async () => {
    if (!current) return;
    if (!(await confirm({ title: `Delete plate run “${current.tag}”?`, message: `Its ${current.files.length} PNG file(s) are removed from ${plates.data?.root ?? "output/nexus_comparisons"}.`, tone: "danger", confirmLabel: "Delete" }))) return;
    try {
      await deletePlateRun(current.tag);
      setRun("");
      setRender("");
      toast.success("Plate run deleted");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  };

  usePageActions([
    { id: "fig-nexus-render", label: "Render NEXUS comparison plates", group: "Plates", disabled: !canRender, run: submit },
    { id: "fig-nexus-selection", label: `Use the Sky tile selection for the plates (${nexusSelection.length})`, group: "Plates",
      disabled: !nexusSelection.length, run: () => setTilesText(nexusSelection.join(", ")) },
    { id: "fig-nexus-refresh", label: "Refresh the NEXUS plate runs", group: "Plates", run: () => void plates.reload() },
  ]);

  const runOptions: SelectOption[] = runs.map((r) => ({ value: r.tag, label: r.tag, hint: formatRelative(r.updated) }));
  const renderOptions: SelectOption[] = (current?.renders ?? []).map((r) => ({ value: renderKey(r), label: renderTitle(r) }));

  return (
    <div className="fig-nexus">
      <section className="fig-panel fig-nexus__form" aria-labelledby="fig-nexus-form">
        <header className="fig-panel__head"><h3 id="fig-nexus-form">Render plates</h3></header>
        <Field label="Tiles" description={tooMany ? undefined : `${coverage.tiles.length} NEXUS tile${coverage.tiles.length === 1 ? "" : "s"}`}
          error={tooMany ? `At most ${maxTiles} tiles` : coverage.unknown.length ? `Unknown: ${coverage.unknown.join(", ")}` : undefined}>
          <Input value={text} onChange={setTilesText} placeholder="40, 42, 70, 178" onEnter={submit} />
        </Field>
        <div className="fig-nexus__quick">
          <Tooltip content="The NEXUS tiles selected in Sky (atlas / real results)">
            <span tabIndex={nexusSelection.length ? -1 : 0}>
              <Button size="sm" variant="ghost" disabled={!nexusSelection.length} onClick={() => setTilesText(nexusSelection.join(", "))}>
                Sky selection ({nexusSelection.length})
              </Button>
            </span>
          </Tooltip>
          {shown && <Button size="sm" variant="ghost" onClick={() => setTilesText(shown.tiles.map((t) => t.index).join(", "))}>Shown run's tiles</Button>}
          <Button size="sm" variant="ghost" onClick={() => setTilesText("")}>Defaults</Button>
        </div>
        <Field label="Colour">
          <Segmented size="sm" value={band} onChange={setBand} aria-label="Band" options={BAND_OPTIONS} />
        </Field>
        <Field label="Model" description={models.error ? models.error.message : coverAll.length ? `On every tile: ${coverAll.slice(0, 4).join(", ")}${coverAll.length > 4 ? "…" : ""}` : "No model has run on every tile yet"}>
          <Select searchable value={chosen} onChange={setSpec} options={modelOptions} aria-label="Model" />
        </Field>
        <Field label="Tag" description="Output folder (default: model-date)">
          <Input value={tag} onChange={setTag} placeholder={`${chosen.replace(":", "-")}-…`} />
        </Field>
        {coverage.missing.length > 0 && (
          <Callout tone="warn" title={`${chosen} has not run on ${coverage.missing.length} tile${coverage.missing.length === 1 ? "" : "s"}`}
            action={<Button asChild size="sm"><Link to={experimentsHref(coverage.missing.map((id) => `nexus/${id}`), chosen)}>Run in Experiments</Link></Button>}>
            {coverage.missing.slice(0, 6).join(", ")}{coverage.missing.length > 6 ? "…" : ""}
          </Callout>
        )}
        {coverage.stale.length > 0 && !coverage.missing.length && (
          <p className="fig-note"><Badge size="sm" tone="warn" dot>stale</Badge> {coverage.stale.length} tile output(s) predate the current {chosen}.</p>
        )}
        <Button variant="primary" icon="image" loading={job.busy} disabled={!canRender} onClick={submit}>Render</Button>
        <JobProgress job={job.job} error={job.error} />
      </section>

      <section className="fig-panel fig-nexus__runs" aria-labelledby="fig-nexus-runs">
        <header className="fig-panel__head">
          <h3 id="fig-nexus-runs">Runs</h3>
          <span className="muted">{plates.data?.root ?? ""}</span>
          <span className="fig-bar__spacer" />
          {runOptions.length > 0 && <Select size="sm" value={current?.tag ?? ""} onChange={(v) => { setRun(v); setRender(""); setTile(""); }} options={runOptions} aria-label="Run" />}
          {renderOptions.length > 1 && <Select size="sm" value={shown ? renderKey(shown) : ""} onChange={(v) => { setRender(v); setTile(""); }} options={renderOptions} aria-label="Render" />}
          {current && <IconButton icon="close" size="sm" label={`Delete run ${current.tag}`} onClick={() => void removeRun()} />}
          <IconButton icon="reset" size="sm" label="Refresh runs" onClick={() => void plates.reload()} />
        </header>
        {plates.loading && !plates.data ? <Skeleton lines={5} />
          : plates.error ? <Callout tone="bad" title="Plate runs did not load" action={<Button size="sm" onClick={() => void plates.reload()}>Retry</Button>}>{plates.error.message}</Callout>
            : !current || !shown ? (
              <EmptyState compact icon="image" title="No plates rendered yet">Pick tiles and a model, then Render.</EmptyState>
            ) : (
              <>
                <DefList dense items={[
                  ["model", <span key="m">{shown.model_label ?? shown.model ?? "—"}{shown.legacy && <> <Badge size="sm">legacy</Badge></>}</span>],
                  !!shown.model_fingerprint && ["fingerprint", <span className="mono" key="f">{shown.model_fingerprint.slice(0, 12)}</span>],
                  ["band", shown.band === "temp" ? "temperature colour" : shown.band],
                  !!shown.field_id && ["field", <span className="mono" key="fi">{shown.field_id}</span>],
                  !!shown.created && ["rendered", formatDateTime(shown.created)],
                  ["tiles", String(shown.tiles.length)],
                ]} />
                {shown.sheet && (
                  <figure className="fig-nexus__sheet">
                    <ServerImage src={plateFileUrl(current.tag, shown.sheet, { thumb: 1400 })} alt={`Contact sheet ${renderTitle(shown)}`} minHeight={260} paper={false} />
                    <figcaption>
                      <span className="mono fig-ellipsis">{shown.sheet}</span>
                      <Button size="sm" href={plateFileUrl(current.tag, shown.sheet)} target="_blank" rel="noreferrer">Full size</Button>
                      <Button size="sm" icon="download" href={plateFileUrl(current.tag, shown.sheet, { download: true })}>PNG</Button>
                    </figcaption>
                  </figure>
                )}
                <ul className="fig-nexus__tiles" aria-label="Tile plates">
                  {shown.tiles.map((t) => (
                    <li key={t.file}>
                      <button type="button" className="fig-nexus__tile" onClick={() => setTile(String(t.index))}
                        aria-label={`Enlarge tile ${t.index}`}
                        title={`Tile ${t.index}${t.ra_deg != null ? ` · ${formatRaDec(t.ra_deg, t.dec_deg ?? NaN)}` : ""}`}>
                        <img src={plateFileUrl(current.tag, t.file, { thumb: 480 })} alt="" loading="lazy" />
                        <span className="fig-nexus__tilecap">
                          <span>tile {t.index}</span>
                          {t.model_state && <Badge size="sm" dot tone={stateTone(t.model_state)}>{t.model_state}</Badge>}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
                <p className="fig-note muted">{current.files.length} file(s) · {formatBytes(current.files.reduce((n, f) => n + f.size, 0))} · updated {formatRelative(current.updated)}</p>
              </>
            )}
      </section>

      <TileDialog tag={current?.tag} tile={enlarged} onClose={() => setTile("")} />
    </div>
  );
}

function TileDialog({ tag, tile, onClose }: { tag?: string; tile?: PlateTileRecord; onClose: () => void }) {
  if (!tag || !tile) return null;
  return (
    <Dialog open onOpenChange={(o) => { if (!o) onClose(); }} title={`NEXUS tile ${tile.index}`} size="xl"
      description={tile.ra_deg != null ? `${formatDeg(tile.ra_deg, 5)} ${formatDeg(tile.dec_deg ?? NaN, 5, { signed: true })}` : undefined}
      footer={<>
        {tile.ref && (
          <Button asChild size="sm">
            <Link to={`/sky/results?${new URLSearchParams({ inspect: `realtile:${tile.ref}` }).toString()}`} onClick={onClose}>Open the real tile</Link>
          </Button>
        )}
        <Button size="sm" icon="download" href={plateFileUrl(tag, tile.file, { download: true })}>PNG</Button>
        <Button size="sm" variant="primary" onClick={onClose}>Close</Button>
      </>}>
      <ServerImage src={plateFileUrl(tag, tile.file)} alt={`Tile ${tile.index} plate`} minHeight={240} paper={false} />
      <DefList dense items={[
        !!tile.id && ["tile", <span className="mono" key="t">{tile.id}</span>],
        !!tile.sr_label && ["SR", tile.sr_label],
        !!tile.model_state && ["state", <Badge key="s" size="sm" dot tone={stateTone(tile.model_state)}>{tile.model_state}</Badge>],
        ["file", <span className="mono" key="f">{tile.file}</span>],
      ]} />
    </Dialog>
  );
}
