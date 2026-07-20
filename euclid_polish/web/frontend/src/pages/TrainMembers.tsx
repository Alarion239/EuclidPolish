/* Train members — its own page under Model. Submits an ensemble_train SLURM
   job. The heart is a PER-MEMBER row editor (add/fork): one row per new model,
   each with its own loss / depth / knee / noise / bootstrap / ICNR / regime /
   seed, exactly like the classic card. The rows are the source of truth — they
   build the positional `member_spec` JSON (+ `count`) the backend consumes
   (build_command → --member-spec). Continue mode selects one or more existing
   members and extends each by the same number of steps. Row/continue prefill
   arrives via ?mode=&member= (the Ensemble members table's ▶continue / ⑂fork
   buttons link here). */
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { StepById } from "../fasrc";
import { LOSS_COLOR } from "../colors";
import { useResource } from "../hooks";
import {
  Badge, Button, Card, CardBody, CardHead, Checkbox, Empty, Field, Input,
  NumberField, Page, PageHead, Segmented, Select, Spinner,
} from "../ui";

type Mode = "add" | "continue" | "fork";
const LOSSES = ["l1", "l2", "l3"];

type ContinueMember = {
  name: string; step?: number | null; loss?: string; blocks?: number | null;
  starless?: boolean;
};
type ContinueStatus = { members?: ContinueMember[] };

/* One new member's knobs. Blank fields fall back to the train-wide defaults. */
type SpecRow = {
  loss: string; blocks: string; knee: string; noise: string;
  boot: string; icnr: boolean; starless: boolean; seed: string;
};
const newRow = (starless: boolean, from?: SpecRow): SpecRow => ({
  loss: from?.loss ?? "l1",
  blocks: from?.blocks ?? "32",
  knee: from?.knee ?? "100",
  noise: from?.noise ?? "0",
  boot: from?.boot ?? "",
  icnr: from?.icnr ?? true,
  starless: from?.starless ?? starless,
  seed: "",                       // never cloned — two members with one seed are one model
});

/* Rows → the positional per-member override list (only non-default keys, so an
   untouched row is `{loss}`; mirrors the classic sync()). */
function buildSpec(rows: SpecRow[], mode: Mode): Record<string, unknown>[] {
  return rows.map((r) => {
    const o: Record<string, unknown> = { loss: r.loss };
    if (!r.starless) o.starless = false;                 // default starless=true
    const noise = parseFloat(r.noise); if (noise > 0) o.noise_aug = noise;
    const boot = parseFloat(r.boot); if (boot > 0 && boot < 1) o.bootstrap = boot;
    const knee = parseFloat(r.knee); if (knee > 0 && knee !== 100) o.asinh_knee = knee;
    if (r.seed.trim() !== "" && Number.isFinite(+r.seed)) o.seed = parseInt(r.seed, 10);
    if (mode === "add") {                                // fork inherits depth + init
      const b = parseInt(r.blocks, 10); if (b > 0) o.num_res_blocks = b;
      if (r.icnr) o.icnr = true;
    }
    return o;
  });
}

export default function TrainMembersPage() {
  const [sp] = useSearchParams();
  const mode0 = (sp.get("mode") as Mode) || "add";
  const prefilledMember = sp.get("member") ?? "";
  const [mode, setMode] = useState<Mode>(
    ["add", "continue", "fork"].includes(mode0) ? mode0 : "add");
  const [member, setMember] = useState(prefilledMember);
  const [selectedMembers, setSelectedMembers] = useState<string[]>(
    prefilledMember ? [prefilledMember] : []);
  const [starlessDefault, setStarlessDefault] = useState(true);
  const [forwardOtf, setForwardOtf] = useState(true);   // run-wide forward model
  const [batchSize, setBatchSize] = useState("1");
  const [hrCropSize, setHrCropSize] = useState("510");
  const [cropsPerField, setCropsPerField] = useState("1");
  const [psfSubset, setPsfSubset] = useState("64");
  const [psfWarpProb, setPsfWarpProb] = useState("1");
  const [psfWarpAlphaMax, setPsfWarpAlphaMax] = useState("20");
  const [psfWarpSigma, setPsfWarpSigma] = useState("3");
  const [saturationMaskProb, setSaturationMaskProb] = useState("0.2");
  const [rows, setRows] = useState<SpecRow[]>([newRow(true)]);
  const [steps, setSteps] = useState("60000");
  const [extraSteps, setExtraSteps] = useState("50000");

  const setRow = (i: number, patch: Partial<SpecRow>) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  const addRow = () => setRows((rs) => [...rs, newRow(starlessDefault, rs[rs.length - 1])]);
  const delRow = (i: number) => setRows((rs) => rs.length > 1 ? rs.filter((_, j) => j !== i) : rs);

  const continueStatus = useResource<ContinueStatus>(
    mode === "continue" ? "/ensemble/status.json" : null, [mode]);
  const continueMembers = useMemo(
    () => continueStatus.data?.members ?? [], [continueStatus.data?.members]);

  // A ?member= deep-link preselects that row. Once the live active-member list
  // arrives, discard stale/archived names so an invisible selection can never
  // be submitted accidentally.
  useEffect(() => {
    if (!continueStatus.data) return;
    const available = new Set(continueMembers.map((m) => m.name));
    setSelectedMembers((current) => {
      const next = current.filter((name) => available.has(name));
      return next.length === current.length ? current : next;
    });
  }, [continueMembers, continueStatus.data]);

  const toggleContinueMember = (name: string, checked: boolean) =>
    setSelectedMembers((current) => checked
      ? (current.includes(name) ? current : [...current, name])
      : current.filter((value) => value !== name));

  const selectedMemberNames = useMemo(() => {
    const selected = new Set(selectedMembers);
    return continueMembers.filter((m) => selected.has(m.name)).map((m) => m.name);
  }, [continueMembers, selectedMembers]);

  const extra = useMemo<Record<string, string>>(() => {
    const geometry = {
      batch_size: batchSize,
      forward_onthefly: forwardOtf ? "1" : "0",
      hr_crop_size: hrCropSize,
      crops_per_field: cropsPerField,
      psf_subset: psfSubset,
      psf_warp_prob: psfWarpProb,
      psf_warp_alpha_max: psfWarpAlphaMax,
      psf_warp_sigma: psfWarpSigma,
      saturation_mask_prob: saturationMaskProb,
    };
    if (mode === "continue")
      return { mode, members: selectedMemberNames.join(","), extra_steps: extraSteps,
        ...geometry };
    const p: Record<string, string> = {
      mode, count: String(rows.length), steps,
      member_spec: JSON.stringify(buildSpec(rows, mode)),
      ...geometry,
    };
    if (mode === "fork") p.fork_from = member.trim();
    return p;
  }, [mode, member, selectedMemberNames, extraSteps, rows, steps, forwardOtf,
    batchSize, hrCropSize, cropsPerField, psfSubset, psfWarpProb,
    psfWarpAlphaMax, psfWarpSigma, saturationMaskProb]);

  const showDepth = mode === "add";   // fork inherits its source's depth + init
  const extraStepCount = Number(extraSteps);
  const extraStepsValid = Number.isInteger(extraStepCount) && extraStepCount > 0;
  const continueSubmitDisabled = mode === "continue"
    && (selectedMemberNames.length === 0 || !extraStepsValid);
  const geometryValid = Number.isInteger(Number(batchSize)) && Number(batchSize) > 0
    && Number.isInteger(Number(hrCropSize)) && Number(hrCropSize) > 0
    && Number(hrCropSize) % 2 === 0
    && Number.isInteger(Number(cropsPerField)) && Number(cropsPerField) > 0
    && Number(psfWarpProb) >= 0 && Number(psfWarpProb) <= 1
    && Number(saturationMaskProb) >= 0 && Number(saturationMaskProb) <= 0.5;
  const selectedSummary = selectedMemberNames.length === 0 ? "…"
    : selectedMemberNames.length <= 3 ? selectedMemberNames.join(", ")
      : `${selectedMemberNames.length} selected members`;

  return (
    <Page>
      <PageHead eyebrow="model · training" title="Train members"
        sub="Submit an ensemble_train SLURM job. Add fresh members with per-model knobs, or continue / fork an existing one."
        right={<Segmented<Mode> value={mode} onChange={setMode}
          options={[{ value: "add", label: "add" }, { value: "continue", label: "continue" }, { value: "fork", label: "fork" }]} />} />

      {mode === "continue" ? (
        <Card>
          <CardHead title="Continue members"
            sub="select one or more active models; one FASRC job trains them sequentially for the same number of additional steps"
            right={<div className="row" style={{ gap: 6 }}>
              <Badge>{selectedMemberNames.length} selected</Badge>
              <Button size="sm" variant="ghost"
                onClick={() => setSelectedMembers(continueMembers.map((m) => m.name))}
                disabled={!continueMembers.length}>select all</Button>
              <Button size="sm" variant="ghost" onClick={() => setSelectedMembers([])}
                disabled={!selectedMemberNames.length}>clear</Button>
            </div>} />
          <CardBody>
            <div className="fasrc-step__res" style={{ marginBottom: "var(--s3)" }}>
              <NumberField label="extra steps" value={extraSteps} onChange={setExtraSteps} min={1000} max={500000} step={1000} />
              <span className="ui-field__hint">Applied to every selected member from its current checkpoint.</span>
            </div>
            {continueStatus.loading ? <Empty><Spinner /> loading active members…</Empty>
              : continueStatus.error ? <Empty>could not load active ensemble members</Empty>
                : !continueMembers.length ? <Empty>no active members are available to continue</Empty>
                  : <div className="continue-members">
                    {continueMembers.map((m) => {
                      const checked = selectedMemberNames.includes(m.name);
                      return <label key={m.name} className="continue-member" data-selected={checked}>
                        <input type="checkbox" checked={checked}
                          onChange={(event) => toggleContinueMember(m.name, event.target.checked)} />
                        <span className="continue-member__body">
                          <span className="continue-member__name mono">{m.name}</span>
                          <span className="continue-member__meta">
                            <Badge>{m.starless ? "starless" : "starfull"}</Badge>
                            <span>{(m.loss ?? "l1").toUpperCase()}</span>
                            {m.blocks != null && <span>{m.blocks} blocks</span>}
                          </span>
                        </span>
                        <span className="continue-member__step mono">
                          {m.step != null ? m.step.toLocaleString() : "—"} steps
                        </span>
                      </label>;
                    })}
                  </div>}
          </CardBody>
        </Card>
      ) : (
        <Card>
          <CardHead
            title={mode === "fork" ? "Fork into new members" : "New members"}
            sub={mode === "fork"
              ? "seed each new member from an existing one; per-model knobs below (depth + init inherited)"
              : "one row per model — each trains with its own knobs for ensemble diversity"}
            right={<div className="row" style={{ gap: 8 }}>
              <Checkbox checked={starlessDefault} onChange={setStarlessDefault}>starless default</Checkbox>
              <Button size="sm" onClick={addRow}>+ member</Button>
            </div>} />
          <CardBody>
            {mode === "fork" && (
              <div className="fasrc-step__res" style={{ marginBottom: "var(--s3)" }}>
                <Field label="fork from"><Input value={member} onChange={setMember} placeholder="member_02" /></Field>
                <NumberField label="steps" value={steps} onChange={setSteps} min={1000} max={500000} step={1000} />
              </div>
            )}
            {mode === "add" && (
              <div className="fasrc-step__res" style={{ marginBottom: "var(--s3)" }}>
                <NumberField label="steps" value={steps} onChange={setSteps} min={1000} max={500000} step={1000} />
              </div>
            )}

            <div className="ens-spec">
              {rows.map((r, i) => (
                <div key={i} className="ens-spec__row">
                  <span className="ens-spec__idx mono">#{i + 1}</span>
                  <Field label="loss">
                    <Select value={r.loss} onChange={(v) => setRow(i, { loss: v })}
                      options={LOSSES.map((v) => ({ value: v, label: v }))} />
                    <span className="ens-spec__dot" style={{ background: LOSS_COLOR[r.loss] }} />
                  </Field>
                  {showDepth && (
                    <Field label="depth"><Input type="number" value={r.blocks} min={4} max={64}
                      onChange={(v) => setRow(i, { blocks: v })} /></Field>
                  )}
                  <Field label="knee [e⁻]"><Input type="number" value={r.knee} min={1} max={100000} step={10}
                    onChange={(v) => setRow(i, { knee: v })} /></Field>
                  <Field label="noise aug"><Input type="number" value={r.noise} min={0} max={5} step={0.25}
                    onChange={(v) => setRow(i, { noise: v })} /></Field>
                  <Field label="bootstrap"><Input type="number" value={r.boot} min={0} max={1} step={0.05}
                    placeholder="off" onChange={(v) => setRow(i, { boot: v })} /></Field>
                  <Field label="regime">
                    <Select value={r.starless ? "1" : "0"} onChange={(v) => setRow(i, { starless: v === "1" })}
                      options={[{ value: "1", label: "starless" }, { value: "0", label: "starfull" }]} />
                  </Field>
                  <Field label="seed"><Input type="number" value={r.seed} placeholder="auto"
                    onChange={(v) => setRow(i, { seed: v })} /></Field>
                  {showDepth && <Checkbox checked={r.icnr} onChange={(v) => setRow(i, { icnr: v })}>ICNR</Checkbox>}
                  <button type="button" className="ens-spec__del" title="remove this model"
                    onClick={() => delRow(i)} disabled={rows.length <= 1}>×</button>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>
      )}

      <Card style={{ marginTop: "var(--s4)" }}>
        <CardHead title="Training geometry"
          sub="Run-wide controls. 510 HR / 255 LR with one crop trains on the complete generated field; loss and validation cover the complete output." />
        <CardBody>
          <div className="fasrc-step__res" style={{ marginBottom: "var(--s3)" }}>
            <NumberField label="batch size" value={batchSize} onChange={setBatchSize}
              min={1} max={64} step={1} />
            <NumberField label="HR example side" value={hrCropSize} onChange={setHrCropSize}
              min={2} max={510} step={2} />
            <NumberField label="examples / field" value={cropsPerField}
              onChange={setCropsPerField} min={1} max={25} step={1} />
            <Checkbox checked={forwardOtf} onChange={setForwardOtf}>on-the-fly forward</Checkbox>
          </div>
          <div className="fasrc-step__res">
            <NumberField label="PSF clusters / member" value={psfSubset}
              onChange={setPsfSubset} min={1} max={10000} step={1} />
            <NumberField label="PSF warp probability" value={psfWarpProb}
              onChange={setPsfWarpProb} min={0} max={1} step={0.05} />
            <NumberField label="PSF warp alpha max" value={psfWarpAlphaMax}
              onChange={setPsfWarpAlphaMax} min={0} max={100} step={1} />
            <NumberField label="PSF warp sigma [HR px]" value={psfWarpSigma}
              onChange={setPsfWarpSigma} min={0.1} max={100} step={0.5} />
            <NumberField label="saturation mask probability" value={saturationMaskProb}
              onChange={setSaturationMaskProb} min={0} max={0.5} step={0.05} />
          </div>
        </CardBody>
      </Card>

      <Card style={{ marginTop: "var(--s4)" }}>
        <CardHead title="Submit"
          sub={mode === "continue"
            ? `continue ${selectedSummary} for ${extraSteps} more steps each`
            : `${rows.length} member${rows.length > 1 ? "s" : ""} · ${steps} steps each`} />
        <CardBody>
          <StepById stepId="ensemble_train" extraParams={extra}
            submitDisabled={continueSubmitDisabled || !geometryValid}
            submitDisabledHint={!geometryValid
              ? "enter a positive batch, an even HR side, positive examples per field, and valid probabilities"
              : selectedMemberNames.length === 0
              ? "select at least one member above"
              : "enter a positive whole number of extra steps"} />
        </CardBody>
      </Card>
    </Page>
  );
}
