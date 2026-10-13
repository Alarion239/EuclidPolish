/* The model-catalogue picker (spec §7.3, GET /api/models): specs grouped
 * production / universal models / mean / gate variants / members, each with
 * its availability — an unavailable spec is disabled and says why. The
 * universal models (standalone single-image multi-knee models, and the active
 * members that are universal too — those stay in Members as well) list their
 * depth, steps, knee and training time. The legacy RBF combiner sits behind a
 * "Legacy RBF" toggle (hidden by default, shown while one is picked), so it
 * can still be benchmarked on real tiles. Quick picks for the common
 * comparisons; a filter for the long lists (members, universal models). */
import { useMemo, useState } from "react";
import { useResource } from "../../../api/query";
import { formatDateTime } from "../../../format";
import { Button, Callout, Checkbox, Input, Skeleton, Tooltip } from "../../../ui";
import { URLS, type ModelSpecRow, type ModelsPayload } from "./api";
import { groupModels, membersText, productionMembersText, specShort, universalDetails, universalDetailText } from "./model";
import "./results.css";

export function useModels() {
  return useResource<ModelsPayload>(URLS.models, [], { ttl: 60_000 });
}

function detail(m: ModelSpecRow): string {
  // a universal model (or a member that is one): depth, steps, knee, hours — not "1 member"
  if (universalDetails(m)) return universalDetailText(m);
  const d = m.details ?? {};
  const bits: string[] = [];
  // "6 of 20 members" for a pruned gate; production says how many it runs.
  const members = m.kind === "production" ? productionMembersText(m) : membersText(m);
  if (members) bits.push(members);
  if (typeof d.mix_space === "string") bits.push(`${d.mix_space} mix`);
  if (d.use_lr === true) bits.push("LR input");
  if (typeof d.fitted_at === "string") bits.push(`fitted ${formatDateTime(d.fitted_at)}`);
  return bits.join(", ");
}

/** The groups long enough to filter (more than 8 rows), with the filter's words. */
const FILTERED: Record<string, { placeholder: string; none: string }> = {
  member: { placeholder: "Filter members", none: "No member matches." },
  universal: { placeholder: "Filter universal models", none: "No universal model matches." },
};

export function ModelPicker({ value, onChange, compact = false }: {
  value: readonly string[]; onChange: (specs: string[]) => void; compact?: boolean;
}) {
  const models = useModels();
  const [q, setQ] = useState<Record<string, string>>({});
  const [legacyOpen, setLegacyOpen] = useState(false);
  const offered = useMemo(() => models.data?.models ?? [], [models.data]);
  const selected = new Set(value);
  const legacy = offered.filter((m) => m.kind === "rbf");
  const showLegacy = legacyOpen || legacy.some((m) => selected.has(m.spec));
  const groups = useMemo(() => groupModels(offered).filter((g) => g.id !== "rbf" || showLegacy), [offered, showLegacy]);
  const available = offered.filter((m) => m.available);
  // the universal group (kind universal + the members that are universal): its runnable specs
  const universalGroup = groups.find((g) => g.id === "universal");
  const universal = (universalGroup?.items ?? []).filter((m) => m.available).map((m) => m.spec);
  const set = (specs: Iterable<string>) => {
    const want = new Set(specs);
    onChange(offered.map((m) => m.spec).filter((s) => want.has(s)));
  };
  const toggle = (spec: string, on: boolean) => {
    const next = new Set(selected);
    if (on) next.add(spec); else next.delete(spec);
    set(next);
  };
  if (models.loading) return <Skeleton lines={4} />;
  if (!models.data) {
    return (
      <Callout tone="bad" title="Could not load the model catalogue"
        action={<Button size="sm" onClick={models.reload}>Retry</Button>}>
        {models.error?.message ?? "No data."}
      </Callout>
    );
  }
  const pick = (kinds: string[]) => set([...selected, ...available.filter((m) => kinds.includes(String(m.kind))).map((m) => m.spec)]);
  const production = available.filter((m) => m.kind === "production").map((m) => m.spec);
  return (
    <div className="res-models" data-compact={compact || undefined}>
      <div className="res-models__quick" role="group" aria-label="Quick picks">
        <Button size="sm" variant="subtle" onClick={() => set(available.filter((m) => m.kind === "production" || m.kind === "mean").map((m) => m.spec))}>
          Production + mean
        </Button>
        {universalGroup && (
          <Button size="sm" variant="subtle" disabled={!universal.length} onClick={() => set([...production, ...universal])}>
            Production + universal
          </Button>
        )}
        <Button size="sm" variant="subtle" onClick={() => pick(["gate"])}>+ gate variants</Button>
        {universalGroup && (
          <Button size="sm" variant="subtle" disabled={!universal.length} onClick={() => set([...selected, ...universal])}>
            + all universal
          </Button>
        )}
        <Button size="sm" variant="subtle" onClick={() => pick(["member"])}>+ all members</Button>
        <Button size="sm" variant="ghost" disabled={!value.length} onClick={() => onChange([])}>Clear</Button>
        {legacy.length > 0 && (
          <Tooltip content="The retired RBF combiner, kept to benchmark it against the gate on real tiles">
            <span><Button size="sm" variant="ghost" aria-pressed={showLegacy}
              disabled={legacy.some((m) => selected.has(m.spec))} onClick={() => setLegacyOpen(!legacyOpen)}>
              {showLegacy ? "Hide legacy RBF" : "Legacy RBF"}
            </Button></span>
          </Tooltip>
        )}
        <span className="res-models__count muted">{value.length} selected</span>
      </div>
      {groups.map((g) => {
        const filter = FILTERED[g.id];
        const needle = (q[g.id] ?? "").trim().toLowerCase();
        const items = filter && needle
          ? g.items.filter((m) => `${m.spec} ${m.label} ${detail(m)}`.toLowerCase().includes(needle)) : g.items;
        const grid = g.id === "member";
        return (
          <fieldset key={g.id} className="res-models__group" data-group={g.id}>
            <legend>
              {g.id === "rbf" ? "Legacy RBF combiner" : g.label}
              {filter && <span className="muted"> ({g.items.length})</span>}
            </legend>
            {filter && g.items.length > 8 && (
              <Input size="sm" value={q[g.id] ?? ""} onChange={(v) => setQ({ ...q, [g.id]: v })} icon="search" clearable
                placeholder={filter.placeholder} aria-label={filter.placeholder} className="res-models__filter" />
            )}
            <div className={grid ? "res-models__grid" : "res-models__list"}>
              {items.map((m) => {
                const line = (
                  <Checkbox checked={selected.has(m.spec)} disabled={!m.available}
                    onChange={(on) => toggle(m.spec, on)}>
                    <span className="res-models__name mono">{grid ? specShort(m.spec) : m.spec}</span>
                    {!grid && !compact && <span className="res-models__detail muted">{m.available ? detail(m) : m.reason}</span>}
                  </Checkbox>
                );
                const tip = [m.label, m.available ? detail(m) : `unavailable: ${m.reason ?? "?"}`].filter(Boolean).join(" — ");
                return (
                  <Tooltip key={m.spec} content={tip}>
                    <span className="res-models__item" data-unavailable={!m.available || undefined}>{line}</span>
                  </Tooltip>
                );
              })}
              {!items.length && <span className="muted">{filter?.none ?? "Nothing to pick."}</span>}
            </div>
          </fieldset>
        );
      })}
    </div>
  );
}
