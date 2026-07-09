/* PSFs — empirical ePSF inventory per band, ePSF panel + cluster-map PNGs,
   FASRC sync buttons (plain JSON POSTs, not jobs), and the two extraction
   pipeline steps. Ported from the classic psfs.html/psfs.py. */
import { useState } from "react";
import { postForm } from "../api";
import { StepById } from "../fasrc";
import { useResource } from "../hooks";
import {
  Badge, Button, Card, CardBody, CardHead, Empty, PageHead, Page,
  PngFigure, Spinner, Table, type Column,
} from "../ui";

interface PsfBand {
  name: string;
  fwhm?: number;
  oversampling?: number;
  epsf_pixel_scale?: number;
  empirical: boolean;
  path?: string;
  shape?: [number, number];
  pixel_scale?: number;
  n_psf?: number;
  error?: string;
}
interface StatusResp {
  psfs: { bands: PsfBand[] };
}

interface SyncFile {
  ok: boolean;
  error?: string;
  remote_path?: string;
  size_bytes?: number;
}
interface SyncResp {
  ok: boolean;
  files: Record<string, SyncFile>;
  clusters_meta?: { ok: boolean; n_clusters?: number; error?: string };
}
interface SyncMetaResp {
  ok: boolean;
  n_clusters?: number;
  local_path?: string;
  error?: string;
}

const BAND_CHIPS = ["all", "VIS", "NIR1", "NIR2", "NISP"];

const fmt = (v: number | undefined, d: number): string =>
  typeof v === "number" ? v.toFixed(d) : "—";

const COLS: Column<PsfBand>[] = [
  { header: "band", cell: (b) => <code className="mono">{b.name}</code> },
  { header: "FWHM (\")", cell: (b) => fmt(b.fwhm, 2), align: "right" },
  { header: "oversampling", cell: (b) => (b.oversampling != null ? `${b.oversampling}×` : "—"), align: "right" },
  { header: "ePSF px scale", cell: (b) => `${fmt(b.epsf_pixel_scale, 4)}"`, align: "right" },
  {
    header: "status",
    cell: (b) =>
      b.empirical
        ? <Badge tone="good">empirical</Badge>
        : <span className="muted">Gaussian fallback</span>,
  },
  {
    header: "# PSFs",
    align: "right",
    cell: (b) =>
      b.empirical
        ? <span title="Position-dependent cluster PSFs (HDU0=mean, HDU1..K). Generation draws a random convex blend of these.">{b.n_psf ?? 1}</span>
        : <span className="muted">—</span>,
  },
  {
    header: "saved kernel",
    cell: (b) => {
      if (!b.empirical) return <span className="muted">—</span>;
      if (b.error) return <span className="muted">{b.error}</span>;
      return (
        <span>
          {b.shape && <code className="mono">{b.shape[0]}×{b.shape[1]}</code>}
          {" @ "}{fmt(b.pixel_scale, 4)}"/pix <span className="muted">(mean)</span>
          {b.path && (
            <>
              {" "}
              <a href={`/inspect?fits=${encodeURIComponent(b.path)}`}
                target="_blank" rel="noreferrer" style={{ fontSize: 11 }}
                title="Inspect FITS header + download">inspect</a>
            </>
          )}
        </span>
      );
    },
  },
];

export default function PsfsPage() {
  const { data, loading, reload } = useResource<StatusResp>("/api/status");
  const [band, setBand] = useState("all");
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);

  const bands = data?.psfs?.bands ?? [];

  async function syncEpsfs() {
    setBusy("sync"); setNote(null);
    try {
      const r = await postForm<SyncResp>("/api/euclid-psf/sync");
      const failed = Object.entries(r.files ?? {})
        .filter(([, f]) => !f.ok)
        .map(([b, f]) => `${b}: ${f.error || "not on FASRC"}`);
      if (r.ok && failed.length === 0) {
        setNote({ ok: true, text: "synced — reloading…" });
        reload();
      } else if (r.ok) {
        setNote({ ok: true, text: `synced (some bands skipped: ${failed.join("; ")})` });
        reload();
      } else {
        setNote({ ok: false, text: failed.join("; ") || "sync failed — connect to FASRC first" });
      }
    } catch (e) {
      setNote({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally { setBusy(null); }
  }

  async function syncMeta() {
    setBusy("meta"); setNote(null);
    try {
      const r = await postForm<SyncMetaResp>("/api/euclid-psf/sync-meta");
      if (r.ok) {
        setNote({ ok: true, text: `${r.n_clusters ?? 0} clusters synced — refreshing the map…` });
        reload();
      } else {
        setNote({ ok: false, text: r.error || "metadata sync failed" });
      }
    } catch (e) {
      setNote({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally { setBusy(null); }
  }

  return (
    <Page>
      <PageHead eyebrow="data · psfs" title="PSFs"
        sub="Empirical ePSF inventory, panels, and cluster map — synced from the FASRC extraction." />

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <Card>
          <CardHead title="ePSF panel" sub="server-rendered per-band PSF montage" />
          <CardBody>
            <PngFigure
              srcFor={(a) => `/view/psfs?band=${encodeURIComponent(a || "all")}`}
              toolbar={BAND_CHIPS.map((b) => ({ key: b, label: b }))}
              active={band} onActive={setBand} alt="PSF panel" />
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Inventory" sub="per-band ePSF status from the local cache" />
          <CardBody>
            {loading && <Empty><Spinner /> loading…</Empty>}
            {!loading && (
              <Table columns={COLS} rows={bands} rowKey={(b) => b.name}
                empty="no bands configured" />
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHead title="PSF clusters"
            sub="local sky positions + angular diameter (× = cluster centre, Ø = angular diameter)" />
          <CardBody>
            <PngFigure srcFor={() => "/view/psf-clusters"} alt="PSF clusters" />
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Synchronise from FASRC"
            sub="the page reads the local cache only — pull fresh ePSFs on demand" />
          <CardBody>
            <div className="row" style={{ gap: "var(--s2)" }}>
              <Button variant="primary" onClick={syncEpsfs} disabled={busy != null}>
                {busy === "sync" ? "syncing…" : "Synchronise ePSFs"}
              </Button>
              <Button onClick={syncMeta} disabled={busy != null}
                title="Metadata only: dump per-cluster centroids + star counts from the VIS ePSF headers on FASRC, then rsync the kilobyte JSON down.">
                {busy === "meta" ? "syncing…" : "Cluster metadata only (light)"}
              </Button>
            </div>
            {note && (
              <div className={note.ok ? "muted" : ""} style={{ marginTop: "var(--s3)" }}>
                {note.ok ? "✓ " : "✗ "}{note.text}
              </div>
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHead title="Extract Euclid PSF" sub="FASRC pipeline step" />
          <CardBody><StepById stepId="extract_euclid_psf" /></CardBody>
        </Card>

        <Card>
          <CardHead title="PSF rotation pool" sub="FASRC pipeline step" />
          <CardBody><StepById stepId="psf_rotation_pool" /></CardBody>
        </Card>
      </div>
    </Page>
  );
}
