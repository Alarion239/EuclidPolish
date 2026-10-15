/* Models › Knees › New run: the collapsible run panel. Pick multi-knee models
   (the catalogue's GET /api/knee-attribution/models; unavailable ones listed
   with their reason, the newest universal models preselected), the number of
   synthetic test fields, real tiles (Sky's tile catalogue: the poster galaxy,
   the NEXUS tiles and the cached Q1 tiles, with a filter box), samples per
   class and the seed. Run states the cost ("local job: N models × M fields,
   ~X min") and asks first (jobs.ts startKneeAttribution).
   The panel's body is mounted only while it is open, so a closed panel reads
   nothing; opening it only GETs. The form state lives in the caller, so it
   survives closing the panel. */
import { useMemo, type Dispatch, type SetStateAction } from "react";
import { formatCount } from "../../../format";
import { Button, Caption, Checkbox, Field, Input, NumberField, Section, Skeleton } from "../../../ui";
import {
  KNEE_TILE_SOURCES, useKneeModels, useTileList, useTileSources, type KneeModelChoice, type TileList,
} from "../kneesApi";
import {
  checkRunForm, defaultModelPick, estimateText, kneeLabel, MAX_PER_CLASS, MAX_SYNTHETIC, modelShort, normaliseModels, runFormData,
  type RunForm,
} from "../knees";
import { startKneeAttribution } from "../jobs";

/** The panel's form state (`models: null` = the default pick until the user
 *  changes it). */
export type RunDraft = Omit<RunForm, "models"> & { models: string[] | null; filter: string };
export const EMPTY_DRAFT: RunDraft = { models: null, synthetic: "20", tiles: [], perClass: "24", seed: "0", filter: "" };

const plural = (n: number, one: string, many = `${one}s`) => `${formatCount(n)} ${n === 1 ? one : many}`;
const MAX_TILE_ROWS = 300;

export function RunPanel({ open, onOpenChange, draft, setDraft, busy, onStarted }: {
  open: boolean; onOpenChange: (open: boolean) => void;
  draft: RunDraft; setDraft: Dispatch<SetStateAction<RunDraft>>;
  busy: boolean; onStarted: (jobId: string) => void;
}) {
  return (
    <Section collapsible open={open} onOpenChange={onOpenChange} className="kn-run" title="New run"
      sub="Local job: attributes each sampled pixel to the model's input knees">
      <RunBody draft={draft} setDraft={setDraft} busy={busy} onStarted={onStarted} />
    </Section>
  );
}

function RunBody({ draft, setDraft, busy, onStarted }: {
  draft: RunDraft; setDraft: Dispatch<SetStateAction<RunDraft>>; busy: boolean; onStarted: (jobId: string) => void;
}) {
  const modelsRes = useKneeModels();
  const models = useMemo(() => normaliseModels(modelsRes.data), [modelsRes.data]);
  // a pick the catalogue no longer runs is dropped (the POST would refuse it)
  const picked = (draft.models ?? defaultModelPick(models))
    .filter((spec) => !models.length || models.some((m) => m.spec === spec && m.available));
  const set = (patch: Partial<RunDraft>) => setDraft((d) => ({ ...d, ...patch }));
  const toggleModel = (spec: string, on: boolean) =>
    set({ models: on ? [...picked.filter((m) => m !== spec), spec] : picked.filter((m) => m !== spec) });
  const toggleTile = (ref: string, on: boolean) =>
    setDraft((d) => ({ ...d, tiles: on ? [...d.tiles.filter((t) => t !== ref), ref] : d.tiles.filter((t) => t !== ref) }));

  const form: RunForm = { models: picked, synthetic: draft.synthetic, tiles: draft.tiles, perClass: draft.perClass, seed: draft.seed };
  const check = checkRunForm(form);
  const nFields = (check.synthetic ?? 0) + draft.tiles.length;
  const estimate = estimateText(picked.length, nFields);
  const run = async () => {
    if (!check.ok || check.synthetic == null || check.perClass == null || check.seed == null) return;
    const id = await startKneeAttribution(runFormData({
      models: picked, synthetic: check.synthetic, tiles: draft.tiles, perClass: check.perClass, seed: check.seed,
    }), estimate);
    if (id) onStarted(id);
  };

  return (
    <div className="kn-run__body">
      <fieldset className="kn-run__group">
        <legend className="kn-run__legend">Models <span className="mdl-faint">{picked.length} picked</span></legend>
        {modelsRes.loading ? <Skeleton lines={3} /> : modelsRes.error ? (
          <p className="mdl-note mdl-warn">Could not read the models: {modelsRes.error.message}</p>
        ) : models.length ? (
          <div className="kn-pick" role="group" aria-label="Models to run">
            {models.map((m) => <ModelRow key={m.spec} m={m} on={picked.includes(m.spec)} onChange={(on) => toggleModel(m.spec, on)} />)}
          </div>
        ) : <p className="mdl-note">No multi-knee model in the catalogue (a universal model or a member trained at several knees).</p>}
        {check.errors.models && models.length > 0 && <p className="mdl-note mdl-warn" role="alert">{check.errors.models}</p>}
      </fieldset>
      <div className="kn-run__numbers">
        <Field label="Synthetic test fields" description={`0–${MAX_SYNTHETIC}, the universal scores' test fields`} error={check.errors.synthetic}>
          <NumberField size="sm" value={draft.synthetic} onChange={(v) => set({ synthetic: v })} min={0} max={MAX_SYNTHETIC} step={1} />
        </Field>
        <Field label="Samples per class" description="per field and band" error={check.errors.perClass}>
          <NumberField size="sm" value={draft.perClass} onChange={(v) => set({ perClass: v })} min={1} max={MAX_PER_CLASS} step={1} />
        </Field>
        <Field label="Seed" error={check.errors.seed}>
          <NumberField size="sm" value={draft.seed} onChange={(v) => set({ seed: v })} min={0} step={1} />
        </Field>
      </div>
      <TilePicker selected={draft.tiles} filter={draft.filter} onFilter={(v) => set({ filter: v })} onToggle={toggleTile}
        onClear={() => set({ tiles: [] })} />
      <div className="kn-run__foot">
        <Button variant="primary" icon="activity" loading={busy} disabled={!check.ok || busy} onClick={() => { void run(); }}>
          Run…
        </Button>
        {check.ok ? <span className="mdl-note">{estimate}</span>
          : check.errors.fields ? <span className="mdl-note mdl-warn">{check.errors.fields}</span> : null}
      </div>
    </div>
  );
}

function ModelRow({ m, on, onChange }: { m: KneeModelChoice; on: boolean; onChange: (on: boolean) => void }) {
  const knees = m.knees.length ? `${m.knees.length} knees, ${kneeLabel(Math.min(...m.knees))}–${kneeLabel(Math.max(...m.knees))}` : "";
  const facts = [m.kind === "universal" ? "universal" : "member", m.depth != null ? `depth ${m.depth}` : null, knees].filter(Boolean).join(" · ");
  return (
    <div className="kn-pick__row" data-off={!m.available || undefined}>
      <Checkbox checked={on && m.available} disabled={!m.available} onChange={onChange}
        title={m.available ? m.label : m.reason ?? "unavailable"}>
        <span className="mdl-member">{modelShort(m.spec)}</span>
      </Checkbox>
      <span className="kn-pick__meta">{m.available ? facts : m.reason ?? "unavailable"}</span>
    </div>
  );
}

/** Real tiles from Sky's catalogue: poster, NEXUS and cached tiles (the
 *  sources that hold any), filtered by ref or label. */
function TilePicker({ selected, filter, onFilter, onToggle, onClear }: {
  selected: string[]; filter: string; onFilter: (v: string) => void; onToggle: (ref: string, on: boolean) => void; onClear: () => void;
}) {
  const sources = useTileSources(true);
  const has = (id: string) => (sources.data?.sources ?? []).some((s) => s.id === id && s.count > 0);
  const [poster, nexus, cached] = KNEE_TILE_SOURCES;
  const p = useTileList(has(poster) ? poster : null);
  const n = useTileList(has(nexus) ? nexus : null);
  const c = useTileList(has(cached) ? cached : null);
  const lists = [p, n, c];
  const tiles = useMemo(() => ([p.data, n.data, c.data].filter(Boolean) as TileList[])
    .flatMap((l) => (l.tiles ?? []).map((t) => ({ ref: t.ref, label: t.label, source: l.label || l.source }))), [p.data, n.data, c.data]);
  const q = filter.trim().toLowerCase();
  const shown = q ? tiles.filter((t) => `${t.ref} ${t.label} ${t.source}`.toLowerCase().includes(q)) : tiles;
  const loading = sources.loading || lists.some((l) => l.loading);
  return (
    <fieldset className="kn-run__group">
      <legend className="kn-run__legend">
        Real tiles <span className="mdl-faint">{selected.length ? `${plural(selected.length, "tile")} picked` : "none picked"}</span>
      </legend>
      <div className="mdl-row">
        <Input size="sm" value={filter} onChange={onFilter} icon="search" clearable placeholder="Filter by ref or label"
          aria-label="Filter real tiles" className="kn-run__filter" />
        {selected.length > 0 && <Button size="sm" variant="ghost" onClick={onClear}>Clear</Button>}
      </div>
      {loading && !tiles.length ? <Skeleton lines={2} /> : sources.error ? (
        <p className="mdl-note mdl-warn">Could not read the real tiles: {sources.error.message}</p>
      ) : tiles.length ? (
        <div className="kn-tiles" role="group" aria-label="Real tiles">
          {shown.slice(0, MAX_TILE_ROWS).map((t) => (
            <Checkbox key={t.ref} checked={selected.includes(t.ref)} onChange={(on) => onToggle(t.ref, on)} title={t.label}>
              <span className="mdl-mono">{t.ref}</span> <span className="mdl-faint">{t.source}</span>
            </Checkbox>
          ))}
          {!shown.length && <p className="mdl-note">No tile matches “{filter}”.</p>}
        </div>
      ) : <p className="mdl-note">No real tiles here yet (the poster galaxy, NEXUS tiles or cached Q1 tiles: Sky › Targets).</p>}
      {shown.length > MAX_TILE_ROWS && <Caption>Showing {MAX_TILE_ROWS} of {formatCount(shown.length)}: filter to find the rest.</Caption>}
    </fieldset>
  );
}
