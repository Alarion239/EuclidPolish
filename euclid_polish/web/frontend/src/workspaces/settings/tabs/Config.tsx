/* settings/config (spec §8.8): the universal JobConfig editor.
 *
 * GET /api/config gives the values, their `version`, the `defaults`, the
 * `types`, and which FASRC steps each field is injected into (`used_by`,
 * `steps`). Save posts ONLY the edited fields with the loaded `version` as
 * `base_version`; a field changed server-side since then (another tab, a
 * galaxy-calibration activation) is refused with 409 `config_conflict` and
 * shown, and "Take server values" rebases (the server's value for those
 * fields, every other edit kept). Hints live in popovers; every filter is in
 * the URL (?q, ?group, ?changed, ?edited). ⌘/Ctrl-S saves. */
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, apiPost } from "../../../api/client";
import { invalidate, useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { useShortcut } from "../../../hooks/useShortcut";
import { useUrlState } from "../../../hooks/useUrlState";
import {
  Badge, Button, Callout, Card, CardBody, CardHead, Chip, EmptyState, Field, IconButton, Input, NumberField,
  Page, PageHead, Select, Skeleton, Tooltip, toast,
} from "../../../ui";
import { GROUPS, type FieldMeta, type GroupId } from "../configFields";
import {
  commonSteps, dirtyFields, fieldError, filterFields, isDefault, orderedFields, rebase, saveBody, toForm,
  type ConfigValues, type Conflict, type FormState,
} from "../configModel";
import "../settings.css";

type ConfigResp = {
  ok: boolean; config: ConfigValues; version?: string;
  defaults?: ConfigValues; types?: Record<string, string>;
  used_by?: Record<string, string[]>; steps?: Record<string, string>;
};
type SaveResp = { ok: boolean; config: ConfigValues; version?: string; note?: string | null; error?: string };
type ConflictBody = { conflicts?: Conflict["fields"]; config?: ConfigValues; version?: string };

const GROUP_OPTIONS = [{ value: "all", label: "All groups" }, ...GROUPS.map((g) => ({ value: g.id, label: g.title }))];
const GROUP_IDS = new Set<string>(GROUPS.map((g) => g.id));
/** `?group=` → a known group id, or undefined (→ "all") for anything else. */
const parseGroup = (raw: string): string | undefined => (GROUP_IDS.has(raw) ? raw : undefined);

/** A default for display: 6 significant digits (1.3888888888888888 → 1.38889). */
const shortNumber = (v: number) => (Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(6))));

/** "Injected into" chips: each links to Ops › FASRC. */
function StepChips({ ids, steps, label }: { ids: string[]; steps: Record<string, string>; label: string }) {
  if (!ids.length) return null;
  return (
    <span className="cfg-field__uses" aria-label={label}>
      {ids.map((step) => (
        <Tooltip key={step} content={`Injected into the ${steps[step] ?? step} FASRC step`}>
          <Link to={`/ops/fasrc?view=steps&step=${encodeURIComponent(step)}`} className="cfg-chip">{steps[step] ?? step}</Link>
        </Tooltip>
      ))}
    </span>
  );
}

function FieldCell({ meta, value, loaded, def, error, usedBy, steps, onChange }: {
  meta: FieldMeta; value: string; loaded: string; def: ConfigValues[string]; error: string | null;
  usedBy: string[]; steps: Record<string, string>; onChange: (v: string) => void;
}) {
  const dirty = value !== loaded;
  const atDefault = def === undefined || isDefault(meta.name, value, { [meta.name]: def });
  // aria-label = the label alone: the unit sits inside the <label> and would
  // otherwise become part of the control's name ("Train scenesscenes").
  const control = meta.kind === "choice" ? (
    <Select value={value} onChange={onChange} options={meta.choices ?? []} aria-label={meta.label} />
  ) : meta.kind === "text" ? (
    <Input value={value} onChange={onChange} aria-label={meta.label} />
  ) : (
    <NumberField value={value} onChange={onChange} min={meta.min} max={meta.max} aria-label={meta.label}
      step={meta.step ?? (meta.kind === "int" ? 1 : "any")} unit={meta.unit} />
  );
  const defText = def === undefined ? null
    : meta.kind === "choice" ? (meta.choices?.find((c) => c.value === String(def))?.label ?? String(def))
      : typeof def === "number" ? shortNumber(def) : String(def);
  return (
    <div className="cfg-field" data-dirty={dirty || undefined} data-changed={!atDefault || undefined}>
      <Field label={meta.label} hint={meta.hint} error={error}>{control}</Field>
      {/* Only what differs is shown: the default when the value is not it,
          the steps this field feeds beyond its group's (the group head lists those). */}
      {((defText != null && !atDefault) || meta.unused || usedBy.length > 0) && (
        <div className="cfg-field__meta">
          {defText != null && !atDefault && (
            <span className="cfg-field__def">
              default <code title={String(def)}>{defText}</code>
              <IconButton icon="reset" size="sm" label={`Reset ${meta.label} to ${defText}`}
                onClick={() => onChange(String(def))} />
            </span>
          )}
          {meta.unused && <Badge size="sm" tone="warn" title="No job or page reads this field">unused</Badge>}
          <StepChips ids={usedBy} steps={steps} label={`${meta.label} is also injected into`} />
        </div>
      )}
    </div>
  );
}

export default function Config() {
  const cfg = useResource<ConfigResp>("/api/config", [], { ttl: 0 });
  const [loaded, setLoaded] = useState<FormState | null>(null);
  const [form, setForm] = useState<FormState | null>(null);
  const [version, setVersion] = useState<string | null>(null);
  const [conflict, setConflict] = useState<Conflict | null>(null);
  const [saving, setSaving] = useState(false);
  const [q, setQ] = useUrlState("q", "");
  const [group, setGroup] = useUrlState<string>("group", "all", { parse: parseGroup });
  const [onlyChanged, setOnlyChanged] = useUrlState("changed", false);
  const [onlyEdited, setOnlyEdited] = useUrlState("edited", false);

  // Seed the editable copy from the first load (a later refetch never
  // overwrites edits: use "Reload" / the conflict flow for that).
  useEffect(() => {
    if (cfg.data?.config && !loaded) {
      const f = toForm(cfg.data.config);
      setLoaded(f); setForm(f); setVersion(cfg.data.version ?? null);
    }
  }, [cfg.data, loaded]);

  const data = cfg.data;
  const defaults = useMemo(() => data?.defaults ?? {}, [data]);
  const types = useMemo(() => data?.types ?? {}, [data]);
  const fields = useMemo(() => orderedFields(Object.keys(data?.config ?? {}), data?.types ?? {}), [data]);
  const dirty = form && loaded ? dirtyFields(form, loaded) : [];
  const errors = useMemo(() => {
    const out: Record<string, string | null> = {};
    if (form) for (const f of fields) out[f.name] = fieldError(f.name, form[f.name] ?? "", types[f.name]);
    return out;
  }, [form, fields, types]);
  const invalid = dirty.filter((k) => errors[k]);
  const changedCount = form ? fields.filter((f) => !isDefault(f.name, form[f.name] ?? "", defaults)).length : 0;

  const set = (name: string, value: string) => setForm((prev) => (prev ? { ...prev, [name]: value } : prev));

  async function save() {
    if (!form || !loaded || !dirty.length || invalid.length || saving) return;
    setSaving(true);
    setConflict(null);
    try {
      const r = await apiPost<SaveResp>("/api/config/save", saveBody(form, loaded, version));
      if (!r.ok) { toast.error("Config not saved", { description: r.error ?? "refused" }); return; }
      const next = toForm(r.config);
      setLoaded(next); setForm(next); setVersion(r.version ?? null);
      toast.success(`Saved ${dirty.length} field${dirty.length === 1 ? "" : "s"}`, r.note ? { description: r.note } : undefined);
      void invalidate("/api/config");
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.code === "config_conflict") {
        const b = (e.body ?? {}) as ConflictBody;
        setConflict({ fields: b.conflicts ?? {}, config: b.config ?? {}, version: b.version ?? null });
        return;
      }
      toast.error("Config not saved", { description: e instanceof Error ? e.message : String(e) });
    } finally {
      setSaving(false);
    }
  }

  function takeServerValues() {
    if (!conflict || !form || !loaded) return;
    const next = rebase(form, loaded, conflict);
    setLoaded(next.loaded); setForm(next.form); setVersion(next.version); setConflict(null);
    toast.info("Server values loaded", { description: "Your other edits are kept — review and save again." });
  }

  function discard() {
    if (loaded) setForm(loaded);
    setConflict(null);
  }

  function resetAllToDefaults() {
    if (!form) return;
    const next = { ...form };
    for (const f of fields) if (defaults[f.name] !== undefined) next[f.name] = String(defaults[f.name]);
    setForm(next);
    toast.info("Every field set to its default", { description: "Nothing is saved until you press Save." });
  }

  async function reload() {
    setLoaded(null); setForm(null); setConflict(null);
    await cfg.reload();
  }

  useShortcut("$mod+s", () => { void save(); return true; }, { description: "Save the job config", scope: "Settings", allowInInputs: true });
  usePageActions([
    { id: "config:save", label: "Save the job config", group: "Settings", keywords: ["config", "save"],
      disabled: !dirty.length || invalid.length > 0, run: () => { void save(); } },
    { id: "config:discard", label: "Discard config edits", group: "Settings", disabled: !dirty.length, run: discard },
    { id: "config:defaults", label: "Set every config field to its default", group: "Settings", run: resetAllToDefaults },
    { id: "config:changed", label: "Show only fields changed from default", group: "Settings",
      run: () => setOnlyChanged(true) },
  ]);

  const shown = form && loaded ? filterFields(fields, {
    q, group: group as GroupId | "all", changed: onlyChanged, edited: onlyEdited,
  }, form, loaded, defaults) : [];
  const groups = GROUPS.map((g) => ({ ...g, fields: shown.filter((f) => f.group === g.id) })).filter((g) => g.fields.length);

  return (
    <Page className="settings-config">
      <PageHead eyebrow="settings · config" title="Job config"
        sub="Shared knobs the pipeline injects into its jobs (~/.euclid_polish/job_config.json)." />

      <div className="settings-bar" role="toolbar" aria-label="Config toolbar">
        <Input value={q} onChange={setQ} icon="search" clearable placeholder="Filter fields…" aria-label="Filter fields"
          className="settings-bar__search" />
        <Select value={group} onChange={setGroup} options={GROUP_OPTIONS} aria-label="Group" size="sm" />
        <Chip on={onlyChanged} onClick={() => setOnlyChanged(!onlyChanged)}>Changed from default · {changedCount}</Chip>
        <Chip on={onlyEdited} onClick={() => setOnlyEdited(!onlyEdited)} disabled={!dirty.length && !onlyEdited}>
          Unsaved · {dirty.length}
        </Chip>
        <span className="settings-bar__spacer" />
        {invalid.length > 0 && <Badge tone="bad">{invalid.length} invalid</Badge>}
        <Button size="sm" variant="ghost" onClick={discard} disabled={!dirty.length}>Discard</Button>
        <Button size="sm" variant="primary" onClick={() => void save()} loading={saving}
          disabled={!dirty.length || invalid.length > 0}>
          Save{dirty.length ? ` ${dirty.length}` : ""}
        </Button>
        <IconButton icon="reset" size="sm" label="Reload from the server (drops unsaved edits)" onClick={() => void reload()} />
      </div>

      {conflict && form && (
        <Callout tone="warn" title="The config changed since you loaded it"
          action={<Button size="sm" variant="primary" onClick={takeServerValues}>Take server values</Button>}>
          Not saved — changed elsewhere:
          <ul className="mono cfg-conflicts">
            {Object.entries(conflict.fields).map(([f, c]) => (
              <li key={f}>{f}: yours {form[f] ?? "—"} · now {String(c.current ?? "—")}</li>
            ))}
          </ul>
        </Callout>
      )}

      {cfg.loading && !form && <Card><CardBody><Skeleton lines={6} /></CardBody></Card>}
      {cfg.error && !form && (
        <Callout tone="bad" title="Could not load the job config"
          action={<Button size="sm" onClick={() => void cfg.reload()}>Retry</Button>}>
          {cfg.error.message}
        </Callout>
      )}

      {form && loaded && groups.length === 0 && (
        <EmptyState icon="filter" title="No field matches"
          action={<Button size="sm" onClick={() => { setQ(""); setGroup("all"); setOnlyChanged(false); setOnlyEdited(false); }}>Clear filters</Button>} />
      )}

      {form && loaded && groups.length > 0 && (
        <div className="cfg-groups">
          {groups.map((g) => {
            const usedBy = data?.used_by ?? {};
            // the steps of the WHOLE group (not just the filtered fields)
            const shared = commonSteps(fields.filter((f) => f.group === g.id), usedBy);
            return (
              <Card key={g.id} aria-label={g.title}>
                <CardHead title={g.title} sub={g.sub}
                  right={<StepChips ids={shared} steps={data?.steps ?? {}} label={`${g.title} is injected into`} />} />
                <CardBody>
                  <div className="cfg-grid">
                    {g.fields.map((meta) => (
                      <FieldCell key={meta.name} meta={meta} value={form[meta.name] ?? ""} loaded={loaded[meta.name] ?? ""}
                        def={defaults[meta.name]} error={form[meta.name] !== loaded[meta.name] ? errors[meta.name] : null}
                        usedBy={(usedBy[meta.name] ?? []).filter((step) => !shared.includes(step))} steps={data?.steps ?? {}}
                        onChange={(v) => set(meta.name, v)} />
                    ))}
                  </div>
                </CardBody>
              </Card>
            );
          })}
          <div className="cfg-foot">
            <Button size="sm" variant="ghost" onClick={resetAllToDefaults}>Set every field to its default…</Button>
          </div>
        </div>
      )}
    </Page>
  );
}
