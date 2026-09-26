/* TNG atlas — manage the IllustrisTNG API token (written to ~/.tng_api_key on
   FASRC), run the TNG50-1 SKIRT atlas download step, and view the derived
   infographics: locally-rendered property histograms, the FASRC grid render,
   and the stacked-FITS download. Ported from the classic tng.html/tng.py. */
import { useEffect, useRef, useState } from "react";
import { postForm } from "../api";
import { useJobsStore } from "../api/jobs";
import { useResource } from "../hooks";
import { useJob } from "../jobs";
import { StepById } from "../fasrc";
import {
  Badge, Button, Card, CardBody, CardHead, ConnBadge, Field, Input, JobProgressView, LogTail,
  Page, PageHead, PngFigure,
} from "../ui";

interface TngAuth {
  present: boolean;
  connected: boolean;
  chars?: number;
}
interface TngSaveResp {
  ok: boolean;
  error?: string;
  chars?: number;
}
interface TngRadiusStatus {
  valid: boolean;
  connected?: boolean;
  expected_count?: number;
  valid_count?: number;
  failed_count?: number;
  reasons?: string[];
  /** The cached validation is missing or old: POST /api/tng/radii/refresh. */
  stale?: boolean;
  cached?: boolean;
  /** Id of a running validation job (started here or elsewhere), else null. */
  refresh_job?: string | null;
}

export default function TngPage() {
  const { data: auth, loading: authLoading, reload: reloadAuth } =
    useResource<TngAuth>("/tng-auth/status");
  const [radiusPoll, setRadiusPoll] = useState(false);
  const { data: radius, loading: radiusLoading, reload: reloadRadius } =
    useResource<TngRadiusStatus>("/api/tng/radii/status", [], { poll: radiusPoll ? 3_000 : undefined });

  // The status GET is read-only and answers from the cache at once; when that
  // cache is stale and FASRC is up, re-validate in a background job (once per
  // visit) and show it. A validation already running (refresh_job) is shown
  // too, and the status is polled until it ends.
  const refresh = useJob("tng:radii-refresh");
  const autoRefreshed = useRef(false);
  const runningId = radius?.refresh_job ?? null;
  const running = useJobsStore((st) => (runningId ? st.jobs[runningId] ?? null : null));
  useEffect(() => {
    if (!radius?.stale || !radius.connected || radius.refresh_job || refresh.busy || autoRefreshed.current) return;
    autoRefreshed.current = true;
    void refresh.run("/api/tng/radii/refresh", {}, { onDone: () => reloadRadius() });
  }, [radius, refresh, reloadRadius]);
  useEffect(() => { setRadiusPoll(!!runningId && !refresh.busy); }, [runningId, refresh.busy]);

  const [token, setToken] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveNote, setSaveNote] = useState<{ ok: boolean; text: string } | null>(null);

  const [showHist, setShowHist] = useState(false);
  const [showGrid, setShowGrid] = useState(false);

  async function saveToken() {
    if (!token.trim()) return;
    setSaving(true); setSaveNote(null);
    try {
      const r = await postForm<TngSaveResp>("/tng-auth/save", { tng_token: token });
      if (r.ok) {
        setSaveNote({ ok: true, text: `token saved to FASRC (${r.chars ?? "?"} chars)` });
        setToken("");
      } else {
        setSaveNote({ ok: false, text: r.error || "failed" });
      }
    } catch (e) {
      setSaveNote({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setSaving(false);
      reloadAuth();
    }
  }

  const authBadge = authLoading ? (
    <Badge>…</Badge>
  ) : !auth?.connected ? (
    <ConnBadge ok={false} labels={["FASRC connected", "FASRC not connected"]} />
  ) : auth.present ? (
    <Badge tone="good">✓ saved ({auth.chars ?? 0} chars)</Badge>
  ) : (
    <Badge tone="warn">not set</Badge>
  );

  return (
    <Page>
      <PageHead eyebrow="data · tng" title="TNG atlas"
        sub="Manage the TNG API token, run the TNG50-1 SKIRT atlas download, and view the results." />

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        {/* API token */}
        <Card>
          <CardHead title="TNG API token" right={authBadge} />
          <CardBody>
            <p className="muted" style={{ marginTop: 0 }}>
              Saved to <code className="mono">~/.tng_api_key</code> on FASRC (mode 600) — the
              same file the download job reads on the compute node. Requires an active FASRC
              connection; the token is sent over the SSH channel as file content, never as a
              command argument and never stored on this machine. Get it from{" "}
              <a href="https://www.tng-project.org/users/profile/" target="_blank" rel="noopener">
                tng-project.org → API Token</a>.
            </p>
            <div className="row" style={{ gap: "var(--s2)", alignItems: "flex-end" }}>
              <Field label="API token">
                <Input type="password" value={token} onChange={setToken}
                  placeholder="paste your tng-project.org API token"
                  disabled={saving} onEnter={saveToken} />
              </Field>
              <Button variant="primary" onClick={saveToken}
                disabled={saving || !token.trim() || !auth?.connected}>
                {saving ? "saving…" : "Save token to FASRC"}
              </Button>
            </div>
            {saveNote && (
              <div className={`job-panel job-panel--${saveNote.ok ? "done" : "err"}`}
                style={{ marginTop: "var(--s3)" }}>
                <LogTail text={saveNote.text} />
              </div>
            )}
          </CardBody>
        </Card>

        {/* Download job (FASRC step) */}
        <Card>
          <CardHead title="Download TNG50-1 SKIRT atlas"
            sub="Bulk-fetch ~1153 galaxies as dusty Euclid VIS + NISP cutouts (20 FITS/galaxy)." />
          <CardBody>
            <p className="muted" style={{ marginTop: 0 }}>
              Runs as a CPU SLURM job with one download thread per allocated CPU. Each finished
              galaxy gets a <code className="mono">.done</code> marker, so a re-submit after a
              time-limit only fills the gaps. Set your API token above first.
            </p>
            <StepById stepId="download_tng_skirt" />
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Measured effective radii"
            sub="required for COSMOS-conditioned population generation"
            right={radiusLoading ? <Badge>…</Badge> : radius?.valid ?
              <Badge tone="good">✓ valid</Badge> : <Badge tone="warn">recalculation required</Badge>} />
          <CardBody>
            <p className="muted" style={{ marginTop: 0 }}>
              Each downloaded TNG VIS frame is measured independently. The
              fitted COSMOS circularized <i>R</i><sub>e</sub> is matched to that
              measurement; the atlas stamp side never enters the scale.
            </p>
            {radius && (
              <p className="mono" style={{ fontSize: "var(--text-sm)" }}>
                {radius.valid_count ?? 0}/{radius.expected_count ?? 0} frames valid
                {radius.failed_count ? ` · ${radius.failed_count} failed` : ""}
              </p>
            )}
            {!radius?.valid && radius?.reasons?.length ? (
              <div className="job-panel job-panel--err"><LogTail text={radius.reasons.join("\n")} /></div>
            ) : null}
            {radius?.stale && !radius.connected && !refresh.job && (
              <p className="muted">This validation is out of date; connect to FASRC to re-check it.</p>
            )}
            <JobProgressView job={refresh.job ?? running} error={refresh.error} />
            <div className="row" style={{ gap: "var(--s2)" }}>
              <StepById stepId="measure_tng_radii" />
              <Button onClick={reloadRadius}>Refresh status</Button>
            </div>
          </CardBody>
        </Card>

        {/* Property histograms (rendered locally) */}
        <Card>
          <CardHead title="Property histograms" sub="rendered locally" />
          <CardBody>
            <p className="muted" style={{ marginTop: 0 }}>
              SFR · stellar mass · total mass · effective radius, over the downloaded galaxies.
              The id list is pulled from FASRC and properties come from the TNG API (cached
              locally, ~30 s for the whole atlas the first time, instant after).
            </p>
            <div className="row" style={{ marginBottom: "var(--s3)" }}>
              <Button onClick={() => setShowHist(true)}>Render histograms</Button>
            </div>
            {showHist && (
              <PngFigure srcFor={() => `/tng/histograms.png?t=${Date.now()}`}
                alt="property histograms" />
            )}
          </CardBody>
        </Card>

        {/* 5×5 image grid job + result */}
        <Card>
          <CardHead title="5×5 image grid"
            sub="5 random galaxies × 5 viewpoints — submit the job, then load the result." />
          <CardBody>
            <p className="muted" style={{ marginTop: 0 }}>
              Pick band (VIS / NISP / RGB) and downsample in the job form; a blank seed re-rolls
              5 random galaxies each submit.
            </p>
            <StepById stepId="tng_grid" />
            <div className="row" style={{ margin: "var(--s3) 0" }}>
              <Button onClick={() => setShowGrid(true)}>Load result grid</Button>
            </div>
            {showGrid && (
              <PngFigure srcFor={() => `/tng/result/grid.png?t=${Date.now()}`}
                alt="5×5 galaxy grid" />
            )}
          </CardBody>
        </Card>

        {/* Stacked-FITS job + download */}
        <Card>
          <CardHead title="Stacked FITS (5 viewpoints of one band)"
            sub="Bundles the 5 viewpoint frames into one multi-extension FITS (~51 MB)." />
          <CardBody>
            <p className="muted" style={{ marginTop: 0 }}>
              Run the job, then download the latest result (the 50 MB UI cap doesn't apply to
              this download).
            </p>
            <StepById stepId="tng_stack" />
            <div className="row" style={{ marginTop: "var(--s3)" }}>
              <Button onClick={() => { window.location.href = "/tng/result/stack.fits"; }}>
                Download latest stacked FITS
              </Button>
            </div>
          </CardBody>
        </Card>
      </div>
    </Page>
  );
}
