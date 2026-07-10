/* Train members — its own page under Model. Submits an ensemble_train SLURM
   job. The heart is a PER-MEMBER row editor (add/fork): one row per new model,
   each with its own loss / depth / knee / noise / bootstrap / ICNR / regime /
   seed, exactly like the classic card. The rows are the source of truth — they
   build the positional `member_spec` JSON (+ `count`) the backend consumes
   (build_command → --member-spec). Continue mode just extends an existing
   member. Row/continue prefill arrives via ?mode=&member= (the Ensemble members
   table's ▶continue / ⑂fork buttons link here). */
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { StepById } from "../fasrc";
import { LOSS_COLOR } from "../colors";
import {
  Button, Card, CardBody, CardHead, Checkbox, Field, Input, NumberField,
  Page, PageHead, Segmented, Select,
} from "../ui";

type Mode = "add" | "continue" | "fork";
const LOSSES = ["l1", "l2", "l3"];

/* One new member's knobs. Blank fields fall back to the train-wide defaults. */
type SpecRow = {
  loss: string; blocks: string; knee: string; noise: string;
  boot: string; icnr: boolean; starless: boolean; seed: string;
};
const newRow = (starless: boolean, from?: SpecRow): SpecRow => ({
  loss: from?.loss ?? "l1",
  blocks: from?.blocks ?? "16",
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
  const [mode, setMode] = useState<Mode>(
    ["add", "continue", "fork"].includes(mode0) ? mode0 : "add");
  const [member, setMember] = useState(sp.get("member") ?? "");
  const [starlessDefault, setStarlessDefault] = useState(true);
  const [rows, setRows] = useState<SpecRow[]>([newRow(true)]);
  const [steps, setSteps] = useState("50000");
  const [extraSteps, setExtraSteps] = useState("50000");

  const setRow = (i: number, patch: Partial<SpecRow>) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)));
  const addRow = () => setRows((rs) => [...rs, newRow(starlessDefault, rs[rs.length - 1])]);
  const delRow = (i: number) => setRows((rs) => rs.length > 1 ? rs.filter((_, j) => j !== i) : rs);

  const extra = useMemo<Record<string, string>>(() => {
    if (mode === "continue")
      return { mode, members: member.trim(), extra_steps: extraSteps };
    const p: Record<string, string> = {
      mode, count: String(rows.length), steps,
      member_spec: JSON.stringify(buildSpec(rows, mode)),
    };
    if (mode === "fork") p.fork_from = member.trim();
    return p;
  }, [mode, member, extraSteps, rows, steps]);

  const showDepth = mode === "add";   // fork inherits its source's depth + init

  return (
    <Page>
      <PageHead eyebrow="model · training" title="Train members"
        sub="Submit an ensemble_train SLURM job. Add fresh members with per-model knobs, or continue / fork an existing one."
        right={<Segmented<Mode> value={mode} onChange={setMode}
          options={[{ value: "add", label: "add" }, { value: "continue", label: "continue" }, { value: "fork", label: "fork" }]} />} />

      {mode === "continue" ? (
        <Card>
          <CardHead title="Continue a member" sub="train an existing member for more steps (its knobs are unchanged)" />
          <CardBody>
            <div className="fasrc-step__res">
              <Field label="member"><Input value={member} onChange={setMember} placeholder="member_00" /></Field>
              <NumberField label="extra steps" value={extraSteps} onChange={setExtraSteps} min={1000} max={500000} step={1000} />
            </div>
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
        <CardHead title="Submit"
          sub={mode === "continue"
            ? `continue ${member || "…"} for ${extraSteps} steps`
            : `${rows.length} member${rows.length > 1 ? "s" : ""} · ${steps} steps each`} />
        <CardBody>
          <StepById stepId="ensemble_train" extraParams={extra} />
        </CardBody>
      </Card>
    </Page>
  );
}
