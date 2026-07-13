import { useMemo, useState } from "react";
import { postForm } from "../api";
import { asArray } from "../data";
import { ConnectionBar, StepById } from "../fasrc";
import { useResource } from "../hooks";
import { JobProgressView, useJob } from "../jobs";
import { useThemeValue } from "../theme";
import {
  Badge, Button, Card, CardBody, CardHead, Checkbox, Field, Input,
  NumberField, Page, PageHead, Stat,
} from "../ui";
import {
  CombinerCard, DisagreementCard, Evaluations, TrainingCurves,
  type Combiner, type Curve, type Evals, type Member,
} from "./Ensemble";

type SplitStatus = {
  present: boolean;
  count?: number | null;
  files?: { dirty?: boolean; lens?: boolean; sources?: boolean };
};

type EvalSummary = {
  ensemble_psnr?: number;
  mean_member_psnr?: number;
  ensemble_gain_db?: number;
  combiner_psnr?: number;
  combiner_vs_mean_db?: number;
};

type Status = {
  root: string;
  records: {
    present: boolean;
    dataset?: { counts?: Record<string, number> };
    splits?: Record<"train" | "validate" | "test", SplitStatus>;
  };
  ensemble: { present: boolean; members: Member[] };
  evaluation: { present: boolean; metrics?: Record<string, { auc?: number }> };
  members?: Member[];
  test_present?: boolean;
  validate_present?: boolean;
  evaluations_ready?: boolean;
  eval_summary?: EvalSummary | null;
  eval_summary_stale?: boolean;
};

const splitLabel = (split?: SplitStatus) => split?.present
  ? `${split.count ?? "?"} fields`
  : "not synced";

export default function LensIsolationPage() {
  const theme = useThemeValue();
  const status = useResource<Status>("/api/lens-isolation/status");
  const evals = useResource<Evals>("/api/lens-isolation/ensemble/evals.json");
  const combiner = useResource<Combiner>("/api/lens-isolation/ensemble/combiner.json");
  const curves = useResource<{ members?: Curve[] }>(
    "/api/lens-isolation/ensemble/training-curves.json",
  );
  const evalJob = useJob();
  const fitJob = useJob();

  const [ntrain, setNtrain] = useState("6400");
  const [nvalid, setNvalid] = useState("100");
  const [ntest, setNtest] = useState("100");
  const [sources, setSources] = useState("");
  const [steps, setSteps] = useState("50000");
  const [batch, setBatch] = useState("16");
  const [evalFields, setEvalFields] = useState("100");
  const [syncTrain, setSyncTrain] = useState(false);
  const [syncValidate, setSyncValidate] = useState(true);
  const [syncTest, setSyncTest] = useState(true);
  const [syncEvaluation, setSyncEvaluation] = useState(true);
  const [syncEnsemble, setSyncEnsemble] = useState(false);
  const [syncMessage, setSyncMessage] = useState("");

  const generation = useMemo(
    () => ({ ntrain, nvalid, ntest }),
    [ntrain, nvalid, ntest],
  );
  const training = useMemo(() => ({
    sources: sources.trim(), steps, batch_size: batch, evaluate_every: "500",
  }), [sources, steps, batch]);

  const reloadEvaluation = () => {
    status.reload();
    evals.reload();
    combiner.reload();
    curves.reload();
  };

  async function sync() {
    const subsets = [
      syncTrain && "train",
      syncValidate && "validate",
      syncTest && "test",
    ].filter(Boolean).join(",");
    setSyncMessage("syncing selected artifacts…");
    try {
      const result = await postForm<{ synced: string[] }>("/api/lens-isolation/sync", {
        subsets,
        ensemble: syncEnsemble ? "1" : "0",
        evaluation: syncEvaluation ? "1" : "0",
      });
      setSyncMessage(result.synced.length ? `synced ${result.synced.join(", ")}` : "nothing selected");
      reloadEvaluation();
    } catch (error) {
      setSyncMessage(error instanceof Error ? error.message : String(error));
    }
  }

  const records = status.data?.records;
  const splits = records?.splits;
  const members = asArray<Member>(status.data?.members ?? status.data?.ensemble?.members);
  const ensembleMetric = status.data?.evaluation?.metrics?.ensemble;
  const summary = status.data?.eval_summary;
  const canEvaluate = Boolean(status.data?.test_present && members.length);

  return (
    <Page>
      <PageHead
        eyebrow="experiment · selective reconstruction"
        title="Lens-system isolation"
        sub="Keep the deflector and lensed source. Training, validation, test, checkpoints, and every local evaluation artifact remain in the sealed experiment namespace."
        right={<ConnectionBar />}
      />

      <Card style={{ marginBottom: "var(--s4)" }}>
        <CardHead title="The reconstruction contract" sub="The target is a physical layer, not a segmentation mask." />
        <CardBody>
          <div className="grid" style={{ gridTemplateColumns: "1fr auto 1fr", alignItems: "stretch", gap: "var(--s3)" }}>
            <div style={{ borderLeft: "4px solid var(--series-mean)", padding: "14px 16px", background: "var(--surface-2)" }}>
              <div className="eyebrow">dirty LR input</div>
              <b>ordinary TNG galaxies + every lens + stars</b>
            </div>
            <div className="mono" style={{ alignSelf: "center", fontSize: 24 }}>→</div>
            <div style={{ borderLeft: "4px solid var(--series-baseline)", padding: "14px 16px", background: "var(--surface-2)" }}>
              <div className="eyebrow">lens HR target</div>
              <b>deflector + lensed source only</b>
            </div>
          </div>
        </CardBody>
      </Card>

      <div className="grid" style={{ gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: "var(--s3)", marginBottom: "var(--s4)" }}>
        <Card><CardBody><Stat k="train records" v={splitLabel(splits?.train)} /></CardBody></Card>
        <Card><CardBody><Stat k="validate records" v={splitLabel(splits?.validate)} /></CardBody></Card>
        <Card><CardBody><Stat k="test records" v={splitLabel(splits?.test)} /></CardBody></Card>
        <Card><CardBody><Stat k="trained members" v={members.length} /></CardBody></Card>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <Card>
          <CardHead title="1 · Generate paired normal fields" sub="Pure TNG ordinary galaxies · 20 lenses / arcmin² · CPU" right={<Badge>CPU</Badge>} />
          <CardBody>
            <div className="fasrc-step__res" style={{ marginBottom: "var(--s3)" }}>
              <NumberField label="train" value={ntrain} onChange={setNtrain} min={1} step={1} />
              <NumberField label="validate" value={nvalid} onChange={setNvalid} min={1} step={1} />
              <NumberField label="test" value={ntest} onChange={setNtest} min={1} step={1} />
            </div>
            <StepById stepId="lens_isolation_generate" extraParams={generation} />
          </CardBody>
        </Card>

        <Card>
          <CardHead title="2 · Fork and train fixed records" sub="Normal dirty LR → lens-only HR record-mode training · GPU" right={<Badge>GPU</Badge>} />
          <CardBody>
            <div className="fasrc-step__res" style={{ marginBottom: "var(--s3)" }}>
              <Field label="source members"><Input value={sources} onChange={setSources} placeholder="member_01,member_04" /></Field>
              <NumberField label="steps" value={steps} onChange={setSteps} min={1} />
              <NumberField label="batch size" value={batch} onChange={setBatch} min={1} />
            </div>
            {!sources.trim() && <p className="ui-field__hint">Enter at least one existing production member before submitting.</p>}
            <StepById stepId="lens_isolation_train" extraParams={training} />
          </CardBody>
        </Card>

        <Card>
          <CardHead title="3 · Evaluate detection behavior" sub="Random block-aligned cutouts grouped by observed target flux · GPU" right={<Badge>GPU</Badge>} />
          <CardBody>
            <StepById stepId="lens_isolation_evaluate" />
            {ensembleMetric?.auc != null && <Stat k="cached ensemble AUC" v={ensembleMetric.auc.toFixed(4)} />}
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Local artifact mirror" sub={status.data?.root ?? "data/experiments/lens_isolation"} />
          <CardBody>
            <p className="muted" style={{ marginTop: 0 }}>
              Train records feed member training; validate records fit the combiner; test records feed full-field metrics and the disagreement viewer.
            </p>
            <div className="row" style={{ gap: "var(--s3)", flexWrap: "wrap" }}>
              <Checkbox checked={syncTrain} onChange={setSyncTrain}>train records</Checkbox>
              <Checkbox checked={syncValidate} onChange={setSyncValidate}>validate records</Checkbox>
              <Checkbox checked={syncTest} onChange={setSyncTest}>test records</Checkbox>
              <Checkbox checked={syncEnsemble} onChange={setSyncEnsemble}>checkpoints</Checkbox>
              <Checkbox checked={syncEvaluation} onChange={setSyncEvaluation}>evaluation artifacts</Checkbox>
              <Button variant="primary" onClick={sync}>Sync selected</Button>
              {syncMessage && <span className="muted mono">{syncMessage}</span>}
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHead
            title="4 · Full-field ensemble evaluation"
            sub="The production ensemble metric/cube pipeline, evaluated against lens_test instead of production HR/clean."
            right={status.data?.evaluations_ready
              ? <Badge tone={status.data.eval_summary_stale ? "warn" : "good"}>{status.data.eval_summary_stale ? "stale" : "current"}</Badge>
              : <Badge>not run</Badge>}
          />
          <CardBody>
            <div className="row" style={{ alignItems: "flex-end", gap: "var(--s3)", flexWrap: "wrap" }}>
              <NumberField label="test fields" value={evalFields} onChange={setEvalFields} min={1} max={2000} />
              <Button variant="primary" disabled={evalJob.busy || !canEvaluate}
                title={canEvaluate ? undefined : "sync test records and experiment checkpoints first"}
                onClick={() => evalJob.run(
                  "/api/lens-isolation/ensemble/evaluate",
                  { num_images: evalFields },
                  { onDone: reloadEvaluation },
                )}>
                Evaluate test set
              </Button>
              {summary?.ensemble_psnr != null && <Stat k="ensemble PSNR" v={`${summary.ensemble_psnr.toFixed(2)} dB`} />}
              {summary?.ensemble_gain_db != null && <Stat k="vs best member" v={`${summary.ensemble_gain_db >= 0 ? "+" : ""}${summary.ensemble_gain_db.toFixed(2)} dB`} />}
              {summary?.combiner_psnr != null && <Stat k="combiner PSNR" v={`${summary.combiner_psnr.toFixed(2)} dB`} />}
            </div>
            <JobProgressView job={evalJob.job} error={evalJob.error} />
          </CardBody>
        </Card>

        <TrainingCurves curves={asArray<Curve>(curves.data?.members)} />
        <Evaluations
          evals={evals.data}
          loading={evals.loading}
          mode="lens-isolation"
          theme={theme}
          viewerCollection="lens-isolation"
          traceBase="/api/lens-isolation/ensemble/pixel-trace.json"
          targetLabel="lens target"
        />
        <CombinerCard
          comb={combiner.data}
          loading={combiner.loading}
          mode="lens-isolation"
          theme={theme}
          fitJob={fitJob}
          onFit={reloadEvaluation}
          evalReady={Boolean(status.data?.evaluations_ready && status.data?.validate_present)}
          fitUrl="/api/lens-isolation/ensemble/combiner/fit"
          title="Lens-isolation combiner"
        />
        <DisagreementCard
          key="lens-isolation"
          mode="lens-isolation"
          members={members}
          collection="lens-isolation"
          targetLabel="lens target"
        />
      </div>
    </Page>
  );
}
