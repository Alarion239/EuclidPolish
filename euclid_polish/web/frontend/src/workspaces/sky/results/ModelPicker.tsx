/* The model-catalogue picker (spec §7.3, GET /api/models): specs grouped
 * production / mean / rbf / gate variants / members, each with its
 * availability — an unavailable spec is disabled and says why. Quick picks
 * for the common comparisons; a filter for the member list. */
import { useMemo, useState } from "react";
import { useResource } from "../../../api/query";
import { formatDateTime } from "../../../format";
import { Button, Callout, Checkbox, Input, Skeleton, Tooltip } from "../../../ui";
import { URLS, type ModelSpecRow, type ModelsPayload } from "./api";
import { groupModels, membersText, productionMembersText, specShort } from "./model";
import "./results.css";

export function useModels() {
  return useResource<ModelsPayload>(URLS.models, [], { ttl: 60_000 });
}

function detail(m: ModelSpecRow): string {
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

export function ModelPicker({ value, onChange, compact = false }: {
  value: readonly string[]; onChange: (specs: string[]) => void; compact?: boolean;
}) {
  const models = useModels();
  const [q, setQ] = useState("");
  const groups = useMemo(() => groupModels(models.data?.models ?? []), [models.data]);
  const selected = new Set(value);
  const available = (models.data?.models ?? []).filter((m) => m.available);
  const set = (specs: Iterable<string>) => {
    const want = new Set(specs);
    onChange((models.data?.models ?? []).map((m) => m.spec).filter((s) => want.has(s)));
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
  const needle = q.trim().toLowerCase();
  return (
    <div className="res-models" data-compact={compact || undefined}>
      <div className="res-models__quick" role="group" aria-label="Quick picks">
        <Button size="sm" variant="subtle" onClick={() => set(available.filter((m) => m.kind === "production" || m.kind === "mean").map((m) => m.spec))}>
          Production + mean
        </Button>
        <Button size="sm" variant="subtle" onClick={() => pick(["gate"])}>+ gate variants</Button>
        <Button size="sm" variant="subtle" onClick={() => pick(["member"])}>+ all members</Button>
        <Button size="sm" variant="ghost" disabled={!value.length} onClick={() => onChange([])}>Clear</Button>
        <span className="res-models__count muted">{value.length} selected</span>
      </div>
      {groups.map((g) => {
        const items = g.id === "member" && needle
          ? g.items.filter((m) => `${m.spec} ${m.label}`.toLowerCase().includes(needle)) : g.items;
        return (
          <fieldset key={g.id} className="res-models__group" data-group={g.id}>
            <legend>
              {g.label}
              {g.id === "member" && <span className="muted"> ({g.items.length})</span>}
            </legend>
            {g.id === "member" && g.items.length > 8 && (
              <Input size="sm" value={q} onChange={setQ} icon="search" clearable placeholder="Filter members"
                aria-label="Filter members" className="res-models__filter" />
            )}
            <div className={g.id === "member" ? "res-models__grid" : "res-models__list"}>
              {items.map((m) => {
                const line = (
                  <Checkbox checked={selected.has(m.spec)} disabled={!m.available}
                    onChange={(on) => toggle(m.spec, on)}>
                    <span className="res-models__name mono">{g.id === "member" ? specShort(m.spec) : m.spec}</span>
                    {g.id !== "member" && !compact && <span className="res-models__detail muted">{m.available ? detail(m) : m.reason}</span>}
                  </Checkbox>
                );
                const tip = [m.label, m.available ? detail(m) : `unavailable: ${m.reason ?? "?"}`].filter(Boolean).join(" — ");
                return (
                  <Tooltip key={m.spec} content={tip}>
                    <span className="res-models__item" data-unavailable={!m.available || undefined}>{line}</span>
                  </Tooltip>
                );
              })}
              {!items.length && <span className="muted">No member matches.</span>}
            </div>
          </fieldset>
        );
      })}
    </div>
  );
}
