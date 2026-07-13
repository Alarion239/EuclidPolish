import { useMemo, useState } from "react";
import { postForm } from "../api";
import { ConnectionBar, StepById } from "../fasrc";
import { useResource } from "../hooks";
import {
  Badge, Button, Card, CardBody, CardHead, Checkbox, Field, Input,
  NumberField, Page, PageHead, Stat,
} from "../ui";

type Status = {
  root: string;
  records: { present: boolean; dataset?: { counts?: Record<string, number> } };
  ensemble: { present: boolean; members: { name: string; checkpoint: boolean }[] };
  evaluation: { present: boolean; metrics?: Record<string, { auc?: number }> };
};

export default function LensIsolationPage() {
  const status = useResource<Status>("/api/lens-isolation/status");
  const [ntrain, setNtrain] = useState("6400");
  const [nvalid, setNvalid] = useState("100");
  const [ntest, setNtest] = useState("100");
  const [sources, setSources] = useState("");
  const [steps, setSteps] = useState("50000");
  const [batch, setBatch] = useState("16");
  const [includeRecords, setIncludeRecords] = useState(false);
  const [includeEnsemble, setIncludeEnsemble] = useState(false);
  const [syncMessage, setSyncMessage] = useState("");

  const generation = useMemo(() => ({ ntrain, nvalid, ntest }), [ntrain, nvalid, ntest]);
  const training = useMemo(() => ({
    sources: sources.trim(), steps, batch_size: batch, evaluate_every: "500",
  }), [sources, steps, batch]);

  async function sync() {
    setSyncMessage("syncing evaluation…");
    try {
      const result = await postForm<{ synced: string[] }>("/api/lens-isolation/sync", {
        records: includeRecords ? "1" : undefined,
        ensemble: includeEnsemble ? "1" : undefined,
      });
      setSyncMessage(`synced ${result.synced.join(", ")}`);
      status.reload();
    } catch (error) {
      setSyncMessage(error instanceof Error ? error.message : String(error));
    }
  }

  const ensembleMetric = status.data?.evaluation.metrics?.ensemble;
  return (
    <Page>
      <PageHead
        eyebrow="experiment · selective reconstruction"
        title="Lens-system isolation"
        sub="Keep the deflector and lensed source. Every normal pure-TNG field is accepted; records and models stay in a sealed experiment namespace."
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
              <div className="eyebrow">clean HR target</div>
              <b>deflector + lensed source only</b>
            </div>
          </div>
        </CardBody>
      </Card>

      <div className="grid" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: "var(--s3)", marginBottom: "var(--s4)" }}>
        <Card><CardBody><Stat k="paired records" v={status.data?.records.present ? "ready" : "not synced"} /></CardBody></Card>
        <Card><CardBody><Stat k="trained members" v={status.data?.ensemble.members.length ?? 0} /></CardBody></Card>
        <Card><CardBody><Stat k="ensemble AUC" v={ensembleMetric?.auc?.toFixed(4) ?? "—"} /></CardBody></Card>
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
          <CardHead title="3 · Evaluate random held-out cutouts" sub="Random, block-aligned crops; group results after sampling by observed target flux" right={<Badge>GPU</Badge>} />
          <CardBody><StepById stepId="lens_isolation_evaluate" /></CardBody>
        </Card>

        <Card>
          <CardHead title="Local artifact mirror" sub={status.data?.root ?? "data/experiments/lens_isolation"} />
          <CardBody>
            <div className="row" style={{ gap: "var(--s3)", flexWrap: "wrap" }}>
              <Checkbox checked={includeRecords} onChange={setIncludeRecords}>include records</Checkbox>
              <Checkbox checked={includeEnsemble} onChange={setIncludeEnsemble}>include checkpoints</Checkbox>
              <Button variant="primary" onClick={sync}>Sync evaluation</Button>
              {syncMessage && <span className="muted mono">{syncMessage}</span>}
            </div>
          </CardBody>
        </Card>
      </div>
    </Page>
  );
}
