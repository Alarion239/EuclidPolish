/* Inference — run SR reconstruction on synthetic pairs or a real Euclid
   cutout, and point at the recent reconstruction PNGs. Two local jobs
   (generate+reconstruct, reconstruct-euclid) each drive a JobProgressView.
   Readiness comes from /api/status; the render-side gallery has no JSON
   listing, so recent outputs are linked to the classic page instead. */
import { useState } from "react";
import { useResource } from "../hooks";
import { useJob, JobProgressView } from "../jobs";
import {
  Badge, Button, Card, CardBody, CardHead, Checkbox, DefList, Empty, Field,
  Input, NumberField, Page, PageHead, Spinner,
} from "../ui";

// ---- API JSON (subset we consume) --------------------------------------
type CheckpointFile = { name: string; rel: string; member: string; size_mb: number };
type CheckpointsStatus = { dir: string; files: CheckpointFile[] };
type TfrecordFile = { name: string; size_mb: number };
type TfrecordsStatus = { dir: string; files: TfrecordFile[] };
type StatusResp = {
  checkpoints: CheckpointsStatus;
  tfrecords: TfrecordsStatus;
};

const CLASSIC = "/inference";

export default function InferencePage() {
  const { data, loading } = useResource<StatusResp>("/api/status");

  const ckpts = data?.checkpoints;
  const recs = data?.tfrecords;
  const hasModel = (ckpts?.files?.length ?? 0) > 0;
  const hasRecords = (recs?.files?.length ?? 0) > 0;

  // --- synthetic generate + reconstruct ---
  const [nPairs, setNPairs] = useState("1");
  const genJob = useJob();
  const runGen = () => {
    const n = Math.max(1, Math.min(8, Math.round(Number(nPairs) || 1)));
    genJob.run("/inference/generate-reconstruct", { n_pairs: n });
  };

  // --- real Euclid cutout ---
  const [ra, setRa] = useState("267.4229");
  const [dec, setDec] = useState("64.8873");
  const [cutout, setCutout] = useState("512");
  const [allBands, setAllBands] = useState(false);
  const euclidJob = useJob();
  const raNum = Number(ra);
  const decNum = Number(dec);
  const raOk = ra.trim() !== "" && Number.isFinite(raNum);
  const decOk = dec.trim() !== "" && Number.isFinite(decNum);
  const euclidOk = raOk && decOk;
  const runEuclid = () => {
    if (!euclidOk) return;
    const size = Math.max(32, Math.min(4096, Math.round(Number(cutout) || 512)));
    euclidJob.run("/inference/reconstruct-euclid", {
      ra: raNum,
      dec: decNum,
      cutout_size: size,
      show_all_bands: allBands ? "1" : undefined,
    });
  };

  const readiness = (
    <div className="row" style={{ gap: 8 }}>
      <Badge tone={hasModel ? "good" : "bad"}>
        {hasModel ? `ensemble · ${ckpts?.files.length} files` : "no active members"}
      </Badge>
      <Badge tone={hasRecords ? "good" : "warn"}>
        {hasRecords ? `records · ${recs?.files.length}` : "no records"}
      </Badge>
    </div>
  );

  return (
    <Page>
      <PageHead
        eyebrow="ops · inference"
        title="Inference"
        sub="Run the ensemble mean on a fresh synthetic scene or a real Euclid cutout."
        right={!loading ? readiness : undefined}
      />

      {loading && <Card><CardBody><Empty><Spinner /> loading status…</Empty></CardBody></Card>}

      {!loading && (
        <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
          <Card>
            <CardHead
              title="Readiness"
              sub="Inference runs the registry-active ensemble members (mean prediction)."
            />
            <CardBody>
              <DefList items={[
                ["model",
                  hasModel
                    ? <span><b>ensemble</b> · {ckpts?.files.length} checkpoint files in <code className="mono">{ckpts?.dir}</code></span>
                    : <span className="muted">no active members — train or pull them on the Ensemble page first</span>],
                ["records",
                  hasRecords
                    ? <span>{recs?.files.length} TFRecords in <code className="mono">{recs?.dir}</code></span>
                    : <span className="muted">none — the synthetic path generates on the fly</span>],
              ]} />
              <p className="muted" style={{ marginTop: "var(--s3)", fontSize: 12 }}>
                HR size + asinh scale come from the Config tab. Manage members on the Ensemble tab.
              </p>
            </CardBody>
          </Card>

          <Card>
            <CardHead
              title="Generate + Reconstruct"
              sub="Synthetic scenes generated + reconstructed in one run (ensemble mean)."
            />
            <CardBody>
              <NumberField
                label="Number of pairs"
                value={nPairs}
                onChange={setNPairs}
                min={1}
                max={8}
                step={1}
                hint="1–8 independent scenes per run."
              />
              <div className="row" style={{ marginTop: "var(--s3)" }}>
                <Button variant="primary" disabled={genJob.busy} onClick={runGen}>
                  Generate & Reconstruct
                </Button>
              </div>
              <div style={{ marginTop: "var(--s3)" }}>
                <JobProgressView job={genJob.job} error={genJob.error} />
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead
              title="Reconstruct — real Euclid cutout"
              sub="Download a cutout at RA/Dec and reconstruct it (+ disagreement cubes when >1 member)."
            />
            <CardBody>
              <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "var(--s3)" }}>
                <Field label="RA (deg)">
                  <Input value={ra} onChange={setRa} onEnter={runEuclid} placeholder="267.4229" />
                </Field>
                <Field label="Dec (deg)">
                  <Input value={dec} onChange={setDec} onEnter={runEuclid} placeholder="64.8873" />
                </Field>
                <NumberField
                  label="Cutout size (VIS px)"
                  value={cutout}
                  onChange={setCutout}
                  min={32}
                  max={4096}
                  step={1}
                />
              </div>
              <div style={{ marginTop: "var(--s3)" }}>
                <Checkbox checked={allBands} onChange={setAllBands}>
                  Show all 4 bands separately (VIS, Y_E, J_E, H_E)
                </Checkbox>
              </div>
              {!euclidOk && (
                <p className="muted" style={{ marginTop: "var(--s2)", fontSize: 12 }}>
                  RA and Dec must both be valid numbers (degrees).
                </p>
              )}
              <div className="row" style={{ marginTop: "var(--s3)" }}>
                <Button variant="primary" disabled={euclidJob.busy || !euclidOk} onClick={runEuclid}>
                  Download & Reconstruct
                </Button>
              </div>
              <div style={{ marginTop: "var(--s3)" }}>
                <JobProgressView job={euclidJob.job} error={euclidJob.error} />
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHead title="Recent outputs" sub="Reconstruction PNGs + inspectable SR FITS." />
            <CardBody>
              <Empty>
                <span>
                  Recent reconstruction PNGs are served under <code className="mono">/vis/reconstruction/</code>{" "}
                  (with per-scene SR FITS sidecars). The full gallery + FITS inspector links live on the{" "}
                  <a href={CLASSIC}>classic Inference page</a>.
                </span>
              </Empty>
            </CardBody>
          </Card>
        </div>
      )}
    </Page>
  );
}
