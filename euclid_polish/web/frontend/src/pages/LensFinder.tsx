/* Lens-finder — the SR-vs-LR lens-discovery pipeline (4 FASRC SLURM steps).
   Replicates the POLISH++ evaluation: finetune a Zoobot binary lens classifier
   and compare its recovery on LR vs SR vs HR source-centered stamps. This page
   just mounts the four FASRC step cards in reading order — each self-loads its
   resources form + confirm-guarded submit + live SLURM monitor. Dataset/training
   knobs live on the Config tab. */
import { Link } from "react-router-dom";
import { ConnectionBar, StepById } from "../fasrc";
import { Card, CardBody, CardHead, Page, PageHead } from "../ui";

export default function LensFinderPage() {
  return (
    <Page>
      <PageHead
        eyebrow="eval · lens-finder"
        title="Lens-finder"
        sub="Does super-resolution surface more lenses? Finetune a Zoobot binary classifier and compare TPR-vs-θ_E on LR vs SR vs HR stamps."
        right={<ConnectionBar />}
      />

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <Card>
          <CardBody>
            <p className="muted" style={{ marginTop: 0 }}>
              Run the four FASRC steps below in order:
              {" "}<code className="mono">lensfinder-data</code> (fields →{" "}
              <code className="mono">records_lensfinder</code>) →{" "}
              <code className="mono">lensfinder-sr-infer</code> (SR each field, GPU) →{" "}
              <code className="mono">lensfinder-stamps</code> (4-band LR/SR/HR Lupton-RGB) →{" "}
              <code className="mono">lensfinder-train</code> (3 Zoobot heads). The final
              TPR-vs-θ<sub>E</sub> plot is produced locally by{" "}
              <code className="mono">scripts/lensfinder_evaluate.py</code> once the heads are synced down.
            </p>
            <p className="hint" style={{ marginBottom: 0 }}>
              Field count / size and the Zoobot training knobs live on the{" "}
              <Link to="/config">Config</Link> tab ("Lens-finder dataset" and
              "Lens-finder training" sections).
            </p>
          </CardBody>
        </Card>

        <Card>
          <CardHead title="1 · Generate lens-finder fields"
            sub="TFRecords + Zenodo lens catalog → records_lensfinder" />
          <CardBody><StepById stepId="lensfinder_generate" /></CardBody>
        </Card>

        <Card>
          <CardHead title="2 · SR inference (GPU)"
            sub="Super-resolve each field → sr_ records" />
          <CardBody><StepById stepId="lensfinder_sr_infer" /></CardBody>
        </Card>

        <Card>
          <CardHead title="3 · Build LR/SR/HR stamps"
            sub="Cut 4-band Lupton-RGB source-centered stamps (CPU)" />
          <CardBody><StepById stepId="lensfinder_build_stamps" /></CardBody>
        </Card>

        <Card>
          <CardHead title="4 · Train the lens-finder (3 heads)"
            sub="Finetune the LR / SR / HR Zoobot binary heads" />
          <CardBody><StepById stepId="lensfinder_train" /></CardBody>
        </Card>
      </div>
    </Page>
  );
}
