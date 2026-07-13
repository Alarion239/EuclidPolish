/* Cutouts — per-star 4-band real-Euclid cutout navigator + Euclid archive
   credentials. Ports the classic /cutouts page: shared canvas cutout viewer
   (collection "cutouts"), totals + per-band valid-star table (from /api/status
   catalog summary), and a save-to-FASRC credentials card. */
import { useCallback, useEffect, useState } from "react";
import { getJSON, postForm } from "../api";
import { asArray } from "../data";
import { useResource } from "../hooks";
import { StepById } from "../fasrc";
import { CutoutViewer } from "../legacy";
import {
  Badge, Button, Card, CardBody, CardHead, Empty, Field, Gallery, Input, LogTail,
  Page, PageHead, Segmented, Spinner, Stat, Table, type Column,
} from "../ui";

// ── API shapes ─────────────────────────────────────────────────────────────
interface Totals {
  count: number;
  size: number | null;
}
interface CatalogSummary {
  total?: number;
  valid?: number;
  valid_by_band?: Record<string, number>;
}
interface CatalogStatus {
  present: boolean;
  summary?: CatalogSummary;
  path?: string;
}
interface StatusResp {
  catalog?: CatalogStatus;
}
interface EuclidAuthStatus {
  present: boolean;
  connected: boolean;
  user?: string;
}
interface SaveResp {
  ok: boolean;
  error?: string;
  user?: string;
}

// Band display order matches Config.BANDS.
const BAND_ORDER = ["VIS", "Y_E", "J_E", "H_E"];

interface BandRow {
  name: string;
  valid: number;
}
const BAND_COLS: Column<BandRow>[] = [
  { header: "band", cell: (b) => <code className="mono">{b.name}</code> },
  { header: "valid (catalog)", cell: (b) => b.valid.toLocaleString(), align: "right" },
];

type CutoutList = { band: string; files: string[]; total: number; page: number; n_pages: number; output_dir: string };

export default function CutoutsPage() {
  const { data: totals } = useResource<Totals>("/api/star-cutouts/totals");
  const { data: status, loading: statusLoading } =
    useResource<StatusResp>("/api/status");

  // ── per-band cutout gallery ─────────────────────────────────────────────
  const [galBand, setGalBand] = useState<string>("VIS");
  const [galPage, setGalPage] = useState(1);
  const gal = useResource<CutoutList>(`/api/cutouts/${galBand}/list.json?page=${galPage}`, [galBand, galPage]);
  const galItems = asArray<string>(gal.data?.files).map((f) => ({
    src: `/cutout-image/${galBand}/${encodeURIComponent(f)}?size=170&output_dir=${encodeURIComponent(gal.data?.output_dir ?? "")}`,
    href: `/cutout-image/${galBand}/${encodeURIComponent(f)}?size=768&output_dir=${encodeURIComponent(gal.data?.output_dir ?? "")}`,
    label: f.replace(/^star_/, "").replace(/\.fits$/, ""),
  }));

  const catalog = status?.catalog;
  const validByBand = catalog?.summary?.valid_by_band;
  const bandRows: BandRow[] = validByBand
    ? BAND_ORDER.filter((n) => n in validByBand).map((n) => ({
        name: n,
        valid: validByBand[n],
      }))
    : [];

  // ── Euclid archive credentials ──────────────────────────────────────────
  const [auth, setAuth] = useState<EuclidAuthStatus | null>(null);
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; text: string } | null>(null);

  const loadAuth = useCallback(async () => {
    const r = await getJSON<EuclidAuthStatus>("/euclid-auth/status");
    setAuth(r);
  }, []);
  useEffect(() => {
    loadAuth();
  }, [loadAuth]);

  async function save() {
    setSaving(true);
    setResult(null);
    try {
      const r = await postForm<SaveResp>("/euclid-auth/save", {
        euclid_user: user,
        euclid_password: password,
      });
      setResult({
        ok: !!r.ok,
        text: r.ok
          ? `saved to FASRC${r.user ? ` for ${r.user}` : ""}`
          : r.error || "failed",
      });
      if (r.ok) {
        setUser("");
        setPassword("");
      }
    } catch (e) {
      setResult({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setSaving(false);
      loadAuth();
    }
  }

  function authBadge() {
    if (!auth) return null;
    if (!auth.connected) return <Badge tone="warn">FASRC not connected</Badge>;
    if (auth.present)
      return (
        <Badge tone="good">
          ✓ saved{auth.user ? ` (${auth.user})` : ""}
        </Badge>
      );
    return <Badge tone="warn">not set</Badge>;
  }

  return (
    <Page>
      <PageHead
        eyebrow="data · cutouts"
        title="Star cutouts"
        sub="Real Euclid stars valid in all 4 bands — pulled from FASRC and cached locally."
      />

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        {/* Navigator */}
        <Card>
          <CardHead
            title="Cutout viewer"
            sub="one real-Euclid star per index, browsable across all 4 bands"
          />
          <CardBody>
            <CutoutViewer collection="cutouts" />
          </CardBody>
        </Card>

        {/* Totals */}
        <Card>
          <CardHead title="Totals" sub="valid-in-all-4-bands stars available to the navigator" />
          <CardBody>
            {totals ? (
              <div className="row" style={{ gap: "var(--s3)", flexWrap: "wrap" }}>
                <Stat k="valid stars" v={totals.count.toLocaleString()} />
                <Stat k="cutout size" v={totals.size != null ? `${totals.size} px` : "—"} />
              </div>
            ) : (
              <Empty>
                <Spinner /> loading totals…
              </Empty>
            )}
          </CardBody>
        </Card>

        {/* Per-band gallery */}
        <Card>
          <CardHead title="Cutout gallery" sub={gal.data ? `${galBand} · ${gal.data.total} cutout(s)` : galBand}
            right={
              <div className="row" style={{ gap: 8 }}>
                <Segmented<string> value={galBand} onChange={(b) => { setGalBand(b); setGalPage(1); }}
                  options={BAND_ORDER.map((b) => ({ value: b, label: b }))} />
              </div>
            } />
          <CardBody>
            {gal.loading ? <Empty><Spinner /> loading…</Empty>
              : <Gallery items={galItems} thumb={130} empty={`no ${galBand} cutouts on disk yet`} />}
            {(gal.data?.n_pages ?? 1) > 1 && (
              <div className="row" style={{ marginTop: "var(--s3)", gap: "var(--s2)", alignItems: "center" }}>
                <Button size="sm" disabled={galPage <= 1} onClick={() => setGalPage((p) => p - 1)}>← prev</Button>
                <span className="muted mono">page {gal.data?.page} / {gal.data?.n_pages}</span>
                <Button size="sm" disabled={galPage >= (gal.data?.n_pages ?? 1)} onClick={() => setGalPage((p) => p + 1)}>next →</Button>
              </div>
            )}
          </CardBody>
        </Card>

        {/* Per-band table */}
        <Card>
          <CardHead title="Per-band valid stars" sub="catalog counts, one row per band" />
          <CardBody>
            {statusLoading && !catalog ? (
              <Empty>
                <Spinner /> loading catalog…
              </Empty>
            ) : catalog?.present ? (
              <Table
                columns={BAND_COLS}
                rows={bandRows}
                empty="no per-band counts in catalog summary"
                rowKey={(b) => b.name}
              />
            ) : (
              <Empty>
                no catalog present — open the classic <code className="mono">/catalog</code> page to
                build one
              </Empty>
            )}
          </CardBody>
        </Card>

        {/* Euclid archive credentials */}
        <Card>
          <CardHead title="Euclid archive login" right={authBadge()} />
          <CardBody>
            <div
              className="grid"
              style={{ gridTemplateColumns: "1fr 1fr", gap: "var(--s3)", maxWidth: 520 }}
            >
              <Field label="Euclid username">
                <Input value={user} onChange={setUser} placeholder="username" />
              </Field>
              <Field label="Euclid password">
                <Input
                  value={password}
                  onChange={setPassword}
                  type="password"
                  placeholder="password"
                  onEnter={save}
                />
              </Field>
            </div>
            <div className="row" style={{ marginTop: "var(--s3)", gap: "var(--s2)" }}>
              <Button
                variant="primary"
                disabled={saving || !user.trim() || !password}
                onClick={save}
              >
                {saving ? "saving…" : "Save credentials to FASRC"}
              </Button>
            </div>
            {result && (
              <div
                className={`job-panel job-panel--${result.ok ? "done" : "err"}`}
                style={{ marginTop: "var(--s3)" }}
              >
                <LogTail text={result.text} />
              </div>
            )}
          </CardBody>
        </Card>

        <StepById stepId="download_euclid_cutouts" />
      </div>
    </Page>
  );
}
