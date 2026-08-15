/* Evaluation — browse real+synthetic reconstructions, run the grouped analysis
   locally, and inspect the pixel-level summary figures. */
import { useState } from "react";
import { postForm } from "../api";
import { useResource } from "../hooks";
import { useJob, JobProgressView } from "../jobs";
import { CutoutViewer } from "../legacy";
import {
  Badge, Button, Card, CardBody, CardHead, Checkbox, Empty, Field, Input,
  LogTail, NumberField, Page, PageHead, PngFigure, Spinner, Table, type Column,
} from "../ui";

type RunRow = {
  ok: boolean; id?: string; out_subdir?: string;
  flux_ratio_sr_over_lr?: number | string | null; grade?: string;
};

const num = (v: unknown): number | null => {
  const n = typeof v === "number" ? v : v == null ? NaN : Number(v);
  return Number.isFinite(n) ? n : null;
};
type RunsResp = { n: number; n_ok: number; run?: string; rows: RunRow[] };

const FIGURES: { key: string; label: string; url: string }[] = [
  { key: "transformation", label: "SR→HR recovery", url: "/api/evaluation/transformation" },
  { key: "angular-power-spectrum", label: "Angular power spectrum", url: "/api/evaluation/angular-power-spectrum" },
];

const RUN_COLS: Column<RunRow>[] = [
  { header: "id", cell: (r) => <code className="mono">{r.id ?? r.out_subdir ?? "—"}</code> },
  { header: "grade", cell: (r) => r.grade ? <Badge tone={r.grade === "A" ? "good" : r.grade === "C" ? "warn" : undefined}>{r.grade}</Badge> : "—" },
  { header: "flux SR/LR", cell: (r) => { const f = num(r.flux_ratio_sr_over_lr); return f != null ? f.toFixed(3) : "—"; }, align: "right" },
];

export default function EvaluationPage() {
  const runs = useResource<RunsResp>("/api/evaluation/runs");
  const job = useJob();
  const [nPer, setNPer] = useState("8");
  const [synthetic, setSynthetic] = useState(true);
  const [nGal, setNGal] = useState("50");
  const [regen, setRegen] = useState(false);
  const [user, setUser] = useState("");
  const [pw, setPw] = useState("");
  const [loggedIn, setLoggedIn] = useState(false);
  const [figNonce, setFigNonce] = useState<Record<string, number>>({});
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const onDone = () => runs.reload();

  async function plainPost(url: string, body?: Record<string, string>) {
    setBusy(url); setNote(null);
    try {
      const r = await postForm<{ ok?: boolean; error?: string; stdout?: string; rows?: number }>(url, body ?? {});
      setNote(r.error ? `✗ ${r.error}` : `✓ ${r.stdout ?? (r.rows != null ? `${r.rows} rows` : "done")}`);
      runs.reload();
    } catch (e) { setNote(`✗ ${e instanceof Error ? e.message : String(e)}`); }
    finally { setBusy(null); }
  }

  async function login() {
    try {
      const r = await postForm<{ ok: boolean; error?: string }>("/auth/login", { username: user, password: pw });
      if (r.ok) { setLoggedIn(true); setNote("✓ logged in to the Euclid archive"); }
      else setNote(`✗ ${r.error ?? "login failed"}`);
    } catch (e) { setNote(`✗ ${e instanceof Error ? e.message : String(e)}`); }
  }

  return (
    <Page>
      <PageHead eyebrow="model · evaluation" title="Evaluation"
        sub="How well SR recovers real + synthetic sources: browse reconstructions, run the analysis pipeline, read the summary figures." />

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <Card>
          <CardHead title="Reconstruction browser" sub="LR · SR · HR · stdSR — and the disagreement movie" />
          <CardBody><CutoutViewer collection="evaluation" /></CardBody>
        </Card>

        <Card>
          <CardHead title="Analysis pipeline" sub="local jobs — reuse cached inference where possible"
            right={busy || note ? <span className="muted">{busy ? "working…" : ""}</span> : undefined} />
          <CardBody>
            <div className="row" style={{ alignItems: "flex-end", gap: "var(--s4)" }}>
              <NumberField label="N per group" value={nPer} onChange={setNPer} min={1} max={200} />
              <Checkbox checked={synthetic} onChange={setSynthetic}>include synthetic</Checkbox>
              <Button variant="primary" disabled={job.busy}
                onClick={() => job.run("/api/evaluation/run-grouped", { n: nPer, synthetic: synthetic ? 1 : 0 }, { onDone })}>
                Grouped analysis
              </Button>
            </div>
            <JobProgressView job={job.job} error={job.error} />
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Euclid archive" sub="needed to query real galaxies"
            right={<Badge tone={loggedIn ? "good" : undefined}>{loggedIn ? "logged in" : "not logged in"}</Badge>} />
          <CardBody>
            {!loggedIn ? (
              <div className="row" style={{ alignItems: "flex-end", gap: "var(--s3)" }}>
                <Field label="username"><Input value={user} onChange={setUser} /></Field>
                <Field label="password"><Input type="password" value={pw} onChange={setPw} onEnter={login} /></Field>
                <Button onClick={login} disabled={!user || !pw}>Log in</Button>
              </div>
            ) : (
              <Button size="sm" onClick={() => { postForm("/auth/logout"); setLoggedIn(false); }}>Log out</Button>
            )}
            <div className="row" style={{ alignItems: "flex-end", gap: "var(--s3)", marginTop: "var(--s4)" }}>
              <NumberField label="N galaxies" value={nGal} onChange={setNGal} min={1} max={2000} />
              <Checkbox checked={regen} onChange={setRegen}>regenerate</Checkbox>
              <Button disabled={!loggedIn || job.busy}
                onClick={() => job.run("/api/evaluation/query-galaxies", { n_galaxies: nGal, regenerate: regen ? 1 : 0 }, { onDone })}>
                Query galaxies
              </Button>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Data sync" sub="pull results from FASRC and fetch the lens catalog" />
          <CardBody>
            <div className="row" style={{ gap: "var(--s2)" }}>
              <Button onClick={() => plainPost("/api/evaluation/sync")} disabled={busy != null}>⟳ Sync results (FASRC)</Button>
              <Button onClick={() => plainPost("/api/evaluation/fetch-catalog")} disabled={busy != null}>Fetch Q1 lens catalog</Button>
              <Button variant="ghost" size="sm" onClick={() => plainPost("/api/evaluation/rerender")} disabled={busy != null}>drop cached PNGs</Button>
            </div>
            {note && <div className="job-panel" style={{ marginTop: "var(--s3)" }}><LogTail text={note} /></div>}
          </CardBody>
        </Card>

        <div className="row" style={{ gap: "var(--s4)", alignItems: "stretch" }}>
          {FIGURES.map((f) => (
            <Card key={f.key} style={{ flex: "1 1 440px" }}>
              <CardHead title={f.label}
                right={<Button size="sm" variant="ghost" onClick={() => setFigNonce((n) => ({ ...n, [f.key]: (n[f.key] ?? 0) + 1 }))}>↻ regen</Button>} />
              <CardBody>
                <PngFigure srcFor={() => `${f.url}?fresh=1&t=${figNonce[f.key] ?? 0}`} alt={f.label} minHeight={260} />
              </CardBody>
            </Card>
          ))}
        </div>

        <Card>
          <CardHead title="Runs" sub={runs.data ? `${runs.data.n_ok}/${runs.data.n} ok${runs.data.run ? ` · ${runs.data.run}` : ""}` : undefined} />
          <CardBody>
            {runs.loading ? <Empty><Spinner /> loading…</Empty>
              : <Table columns={RUN_COLS} rows={runs.data?.rows ?? []} empty="no eval runs yet — run the grouped analysis"
                  rowKey={(r, i) => r.id ?? r.out_subdir ?? i} />}
          </CardBody>
        </Card>
      </div>
    </Page>
  );
}
