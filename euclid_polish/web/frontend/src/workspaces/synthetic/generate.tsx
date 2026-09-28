/* The synthetic_generate FASRC step, shared by Synthetic › Status (the gate's
   "Generate validate+test on FASRC" dialog) and Records (the "Generate and
   sync" drawer): which splits to rebuild from zero (none = resume: complete
   splits are reused, incomplete ones resume), then the step card, whose
   submit is confirmed. The card prefills from the last run; a
   --regenerate-splits / --force left in its extra flags would silently turn a
   resume into a rebuild, so they are dropped (dataModel.ts resumeSafeStep). */
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { StepById, StepCard, useStepsStatus } from "../../fasrc";
import { Badge, Button, Chip, Tooltip } from "../../ui";
import { SPLITS, type Split } from "./dataApi";
import { resumeSafeStep } from "./dataModel";

export function GeneratePanel({ defaultSplits = [] }: { defaultSplits?: Split[] }) {
  const [splits, setSplits] = useState<Split[]>(defaultSplits);
  const steps = useStepsStatus();
  const toggle = (s: Split) => setSplits((c) => (c.includes(s) ? c.filter((x) => x !== s) : [...c, s]));
  const found = steps.data?.steps?.find((st) => st.step_id === "synthetic_generate");
  const safe = useMemo(() => (found ? resumeSafeStep(found) : null), [found]);
  const extraParams = splits.length ? { regenerate_splits: splits.join(",") } : undefined;
  const hideParams = splits.length ? ["force", "regenerate_splits"] : ["regenerate_splits"];
  return (
    <div className="dt-gen">
      <div className="dt-gen__splits" role="group" aria-label="Splits to rebuild">
        <span className="dt-label">Rebuild</span>
        {SPLITS.map((s) => (
          <Chip key={s} on={splits.includes(s)} onClick={() => toggle(s)}>{s}</Chip>
        ))}
        <Button size="sm" variant="ghost" onClick={() => setSplits(["validate", "test"])}>validate + test</Button>
        {splits.length > 0 && <Button size="sm" variant="ghost" onClick={() => setSplits([])}>resume instead</Button>}
        {splits.length ? <Badge size="sm" tone="warn">rebuild {splits.join(" + ")}</Badge> : <Badge size="sm">resume</Badge>}
        <Tooltip content={splits.length
          ? "The selected splits are deleted at job start and rebuilt from zero; the others are left untouched."
          : "Complete splits are reused and incomplete ones resume."}>
          <span className="dt-help" tabIndex={0} aria-label="About the rebuild">?</span>
        </Tooltip>
        {safe && safe.dropped.length > 0 && (
          <Tooltip content={`The last run's extra flags carried ${safe.dropped.join(" ")}; it was left out of this form so a resume stays a resume. Pick the splits above to rebuild.`}>
            <span tabIndex={0}><Badge size="sm" tone="info">dropped {safe.dropped.join(" ")}</Badge></span>
          </Tooltip>
        )}
      </div>
      {safe ? (
        <div className="ops-step-inline">
          <StepCard key={safe.step.step_id} step={safe.step} sshConnected={!!steps.data?.ssh_connected} embedded
            extraParams={extraParams} hideParams={hideParams} />
        </div>
      ) : (
        <StepById stepId="synthetic_generate" embedded extraParams={extraParams} hideParams={hideParams} />
      )}
    </div>
  );
}

/** "Edit in Config": the one editor of the generation knobs. */
export function EditInConfig() {
  return (
    <Button asChild size="sm" variant="ghost" iconRight="chevronRight">
      <Link to="/system/config">Edit in Config</Link>
    </Button>
  );
}
