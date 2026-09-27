/* "Run models…": a popover with the model picker that starts an experiment
 * on the given tiles (POST /api/experiments, confirm first). */
import { useState, type ReactElement } from "react";
import type { Job } from "../../../api/jobs";
import { Button, Input, Popover } from "../../../ui";
import { runModels } from "./actions";
import { ModelPicker } from "./ModelPicker";

export function RunModelsPopover({ refs, trigger, defaultSpecs = ["production", "mean"], onStarted, onDone, open: openProp, onOpenChange }: {
  refs: readonly string[];
  trigger?: ReactElement;
  defaultSpecs?: string[];
  onStarted?: (experimentId: string | null) => void;
  onDone?: (job: Job, experimentId: string | null) => void;
  /** Controlled open state (e.g. opened from the command palette). */
  open?: boolean; onOpenChange?: (open: boolean) => void;
}) {
  const [inner, setInner] = useState(false);
  const open = openProp ?? inner;
  const setOpen = (v: boolean) => { setInner(v); onOpenChange?.(v); };
  const [specs, setSpecs] = useState<string[]>(defaultSpecs);
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    try {
      const r = await runModels(refs, specs, { label, onDone });
      if (r) { setOpen(false); onStarted?.(r.experimentId); }
    } finally { setBusy(false); }
  };
  return (
    <Popover open={open} onOpenChange={setOpen} label="Run models" width={420} align="start"
      trigger={trigger ?? (
        <Button size="sm" variant="primary" icon="activity" disabled={!refs.length}>
          Run models{refs.length > 1 ? ` (${refs.length})` : ""}…
        </Button>
      )}>
      <div className="res-run">
        <div className="res-run__head">
          <strong>Run on {refs.length === 1 ? refs[0] : `${refs.length} tiles`}</strong>
        </div>
        <ModelPicker value={specs} onChange={setSpecs} compact />
        <div className="res-run__foot">
          <Input size="sm" value={label} onChange={setLabel} placeholder="Label (optional)" aria-label="Experiment label" />
          <Button size="sm" variant="primary" loading={busy} disabled={!specs.length || !refs.length} onClick={run}>
            Run {specs.length || ""}
          </Button>
        </div>
      </div>
    </Popover>
  );
}
