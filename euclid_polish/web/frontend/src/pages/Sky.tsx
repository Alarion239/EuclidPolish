/* Sky — browse the sky TFRecords (LR/HR/SR tiers) in the cutout viewer, sync the
   records from FASRC, and generate SR locally. Ported from the classic sky.html.
   Follows Git.tsx (postForm + status card) & Ensemble.tsx (Segmented + Stats). */
import { useState } from "react";
import { postForm } from "../api";
import { useResource, usePolling } from "../hooks";
import { useJob, JobProgressView } from "../jobs";
import { StepById } from "../fasrc";
import { CutoutViewer } from "../legacy";
import {
  Badge, Button, Card, CardBody, CardHead, Checkbox, Empty, LogTail, Page,
  PageHead, Segmented, Spinner, Stat,
} from "../ui";

type Subset = "test" | "validate" | "train";
const GENERATION_SPLITS: Subset[] = ["train", "validate", "test"];

interface SrStatus {
  records: boolean;
  checkpoint: boolean;
  can_generate: boolean;
  subsets?: Subset[];
  sr: { test: number; validate: number; train: number };
}

interface SyncFileResult {
  ok: boolean;
  size_bytes?: number;
  error?: string;
}
interface SyncResult {
  ok: boolean;
  files: Record<string, SyncFileResult>;
  include_train?: boolean;
  include_validate?: boolean;
}

export default function SkyPage() {
  const [subset, setSubset] = useState<Subset>("test");
  const [viewerKey, setViewerKey] = useState(0);
  const [regenerateSplits, setRegenerateSplits] = useState<Subset[]>([]);

  function toggleRegenerationSplit(split: Subset, checked: boolean) {
    setRegenerateSplits((current) => GENERATION_SPLITS.filter((candidate) =>
      candidate === split ? checked : current.includes(candidate)));
  }

  const { data: status, reload: reloadStatus } =
    useResource<SrStatus>("/api/sky/sr-status");

  // --- Generate SR (local {job_id} job) ---------------------------------
  const { job, error: jobError, busy: jobBusy, run: runJob } = useJob();
  // While a generate job runs, keep the counts fresh.
  usePolling(reloadStatus, 3000, jobBusy);

  function generateSr() {
    runJob("/api/sky/generate-sr", {}, {
      onDone: () => {
        reloadStatus();
        setViewerKey((k) => k + 1); // re-mount the viewer so the SR tier unlocks
      },
    });
  }

  // --- Sync records from FASRC (plain POST) -----------------------------
  const [includeTrain, setIncludeTrain] = useState(false);
  const [syncBusy, setSyncBusy] = useState(false);
  const [syncNote, setSyncNote] = useState<{ ok: boolean; text: string } | null>(null);

  async function sync() {
    setSyncBusy(true);
    setSyncNote(null);
    try {
      const r = await postForm<SyncResult>("/api/sky/sync", {
        include_train: includeTrain ? 1 : 0,
      });
      if (r.ok) {
        const n = Object.values(r.files).filter((f) => f.ok).length;
        setSyncNote({ ok: true, text: `✓ synced ${n} shard(s).` });
        reloadStatus();
        setViewerKey((k) => k + 1);
      } else {
        setSyncNote({
          ok: false,
          text: "✗ sync failed (nothing pulled). Generate on FASRC first?",
        });
      }
    } catch (e) {
      setSyncNote({ ok: false, text: `✗ ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      setSyncBusy(false);
    }
  }

  const sr = status?.sr;

  const genHint = !status
    ? ""
    : !status.records
      ? "sync sky records first"
      : !status.checkpoint
        ? "no active ensemble — train or pull members on the Ensemble page"
        : "runs the model over the downloaded records";

  return (
    <Page>
      <PageHead
        eyebrow="data · sky"
        title="Sky records"
        sub="Browse the sky TFRecords (LR / HR / SR tiers), sync them from FASRC, and generate SR locally."
      />

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <Card>
          <CardHead
            title="Cutout viewer"
            sub="multi-band sky TFRecords · LR / HR / SR tiers"
            right={
              <Segmented<Subset>
                value={subset}
                onChange={setSubset}
                options={[
                  { value: "test", label: "test" },
                  { value: "validate", label: "validate" },
                  { value: "train", label: "train" },
                ]}
              />
            }
          />
          <CardBody>
            <CutoutViewer
              key={`${subset}-${viewerKey}`}
              collection="sky"
              params={{ subset }}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHead
            title="Generate SR"
            sub="run the model over the downloaded records — needs records + a checkpoint"
            right={
              status &&
              (status.can_generate ? (
                <Badge tone="good">ready</Badge>
              ) : (
                <Badge tone="warn">{status.records ? "no checkpoint" : "no records"}</Badge>
              ))
            }
          />
          <CardBody>
            {!status ? (
              <Empty><Spinner /> loading…</Empty>
            ) : (
              <>
                <div className="row" style={{ gap: "var(--s6)" }}>
                  <Stat k="SR · test" v={sr?.test ?? "—"} />
                  <Stat k="SR · validate" v={sr?.validate ?? "—"} />
                  <Stat k="SR · train" v={sr?.train ?? "—"} />
                </div>
                <div className="row" style={{ marginTop: "var(--s4)", gap: "var(--s3)", alignItems: "center" }}>
                  <Button variant="primary" disabled={!status.can_generate || jobBusy} onClick={generateSr}>
                    ✨ Generate SR
                  </Button>
                  <span className="muted">— {genHint}</span>
                </div>
                {(job || jobError) && (
                  <div style={{ marginTop: "var(--s3)" }}>
                    <JobProgressView job={job} error={jobError} />
                  </div>
                )}
              </>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHead
            title="Sync records from FASRC"
            sub="pulls held-out test and validation splits together; only the large train split is opt-in"
          />
          <CardBody>
            <div className="row" style={{ gap: "var(--s6)", flexWrap: "wrap" }}>
              <Checkbox checked={includeTrain} onChange={setIncludeTrain} disabled={syncBusy}>
                also pull train shards
              </Checkbox>
            </div>
            <div className="row" style={{ marginTop: "var(--s4)", gap: "var(--s3)", alignItems: "center" }}>
              <Button variant="primary" disabled={syncBusy} onClick={sync}>
                ⤓ Sync records from FASRC
              </Button>
              {syncBusy && <span className="muted"><Spinner /> syncing… (large shards may take a while)</span>}
            </div>
            {syncNote && (
              <div className={`job-panel job-panel--${syncNote.ok ? "done" : "err"}`} style={{ marginTop: "var(--s3)" }}>
                <LogTail text={syncNote.text} />
              </div>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHead
            title="Generate synthetic training pairs"
            sub="resume incomplete data normally, or explicitly replace only selected splits"
            right={regenerateSplits.length
              ? <Badge tone="warn">targeted rebuild</Badge>
              : <Badge>resume mode</Badge>}
          />
          <CardBody>
            <div className="row" style={{ gap: "var(--s5)", flexWrap: "wrap", alignItems: "center" }}>
              <span className="eyebrow">regenerate only</span>
              {GENERATION_SPLITS.map((split) => (
                <Checkbox key={split} checked={regenerateSplits.includes(split)}
                  onChange={(checked) => toggleRegenerationSplit(split, checked)}>
                  {split}
                </Checkbox>
              ))}
              <Button size="sm" variant="ghost"
                onClick={() => setRegenerateSplits(["validate", "test"])}>
                validate + test
              </Button>
              {!!regenerateSplits.length && (
                <Button size="sm" variant="ghost" onClick={() => setRegenerateSplits([])}>
                  clear
                </Button>
              )}
            </div>
            {regenerateSplits.length ? (
              <div className="job-panel job-panel--warn" style={{ marginTop: "var(--s3)" }}>
                Selected splits are deleted at job start and rebuilt from zero.
                Unselected splits are left untouched, even if incomplete.
              </div>
            ) : (
              <div className="muted" style={{ marginTop: "var(--s3)" }}>
                No target selected: complete splits are reused and incomplete splits resume.
              </div>
            )}
            <div style={{ marginTop: "var(--s4)" }}>
              <StepById stepId="synthetic_generate" embedded
                extraParams={{
                  // Keep the structured value for history/new backends, and
                  // also use the long-standing extra_flags path so a resident
                  // pre-upgrade Flask process still forwards the new CLI flag.
                  regenerate_splits: regenerateSplits.join(","),
                  extra_flags: regenerateSplits.length
                    ? `--regenerate-splits=${regenerateSplits.join(",")}` : "",
                }} />
            </div>
          </CardBody>
        </Card>
      </div>
    </Page>
  );
}
