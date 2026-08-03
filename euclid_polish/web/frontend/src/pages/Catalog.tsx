/* Catalog — the FASRC star catalog: summary counts, the server-rendered catalog
   views (positions / magnitudes / saturation), the euclid_query build step, and
   Euclid-archive login. Ported from the classic catalog.html + catalog.py. */
import { useState } from "react";
import { postForm } from "../api";
import { useResource } from "../hooks";
import { StepById } from "../fasrc";
import {
  Button, Card, CardBody, CardHead, DefList, Empty, Field, Input, LogTail,
  Page, PageHead, PngFigure, Spinner,
} from "../ui";

type CatalogSummary = {
  total: number;
  valid: number;
  pending: number;
  corrupted: number;
  failed: number;
  mag_min?: number;
  mag_max?: number;
};
type CatalogStatus = {
  present: boolean;
  summary?: CatalogSummary;
  path?: string;
};
type StatusResp = { catalog: CatalogStatus };

type AuthResp = { ok: boolean; error?: string; user?: string };
type AuthStatus = { authenticated: boolean; user?: string };

type View = "positions" | "magnitudes" | "saturation";
const VIEWS: { key: View; label: string }[] = [
  { key: "positions", label: "sky positions" },
  { key: "magnitudes", label: "magnitude histogram" },
  { key: "saturation", label: "saturation cutoffs" },
];

export default function CatalogPage() {
  const { data, loading } = useResource<StatusResp>("/api/status");
  const auth = useResource<AuthStatus>("/auth/status");
  const [view, setView] = useState<View>("positions");

  const [user, setUser] = useState("");
  const [pwd, setPwd] = useState("");
  const [authNote, setAuthNote] = useState<{ ok: boolean; text: string } | null>(null);
  const [authBusy, setAuthBusy] = useState(false);

  const cat = data?.catalog;
  const sum = cat?.summary;

  async function login() {
    setAuthBusy(true);
    setAuthNote(null);
    try {
      const r = await postForm<AuthResp>("/auth/login", { username: user, password: pwd });
      if (r.ok) {
        setPwd("");
        setAuthNote({ ok: true, text: `Logged in${r.user ? ` as ${r.user}` : ""}.` });
        auth.reload();
      } else {
        setAuthNote({ ok: false, text: r.error || "login failed" });
      }
    } catch (e) {
      setAuthNote({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setAuthBusy(false);
    }
  }

  async function logout() {
    setAuthBusy(true);
    try {
      await postForm("/auth/logout");
    } catch {
      /* ignore — logout is best-effort */
    } finally {
      setAuthNote(null);
      setAuthBusy(false);
      auth.reload();
    }
  }

  return (
    <Page>
      <PageHead
        eyebrow="data · catalog"
        title="Catalog"
        sub="The FASRC star catalog — build it with a query, then inspect its coverage and photometry."
      />

      <div className="grid" style={{ gridTemplateColumns: "1fr", gap: "var(--s4)" }}>
        <Card>
          <CardHead
            title="Summary"
            sub={cat?.path ? <code className="mono">{cat.path}</code> : "the FASRC (netscratch) catalog"}
          />
          <CardBody>
            {loading ? (
              <Empty><Spinner /> loading…</Empty>
            ) : !cat?.present || !sum ? (
              <Empty>
                No <code>stars.csv</code> on disk yet — run the{" "}
                <b>euclid_query</b> step below to seed the catalog.
              </Empty>
            ) : (
              <DefList
                items={[
                  ["total stars", sum.total],
                  ["valid (any band)", sum.valid],
                  ["pending", sum.pending],
                  ["corrupted", sum.corrupted],
                  ["failed", sum.failed],
                  [
                    "magnitude range",
                    sum.mag_min != null && sum.mag_max != null
                      ? `${sum.mag_min.toFixed(2)} — ${sum.mag_max.toFixed(2)}`
                      : "—",
                  ],
                  ["remote path", cat.path ? <code className="mono">{cat.path}</code> : "—"],
                ]}
              />
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHead
            title="Catalog views"
            sub="server-rendered from the cached catalog copy"
          />
          <CardBody>
            <PngFigure
              srcFor={(v) => `/view/catalog?view=${v ?? "positions"}`}
              downloadSrc={(v) => `/view/catalog?view=${v ?? "positions"}&dpi=300`}
              toolbar={VIEWS}
              active={view}
              onActive={(k) => setView(k as View)}
              alt={`catalog ${view}`}
              minHeight={360}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHead
            title="Build catalog · euclid_query"
            sub="SLURM job that queries the Euclid archive and builds the catalog"
          />
          <CardBody>
            <StepById stepId="euclid_query" />
          </CardBody>
        </Card>

        <StepById stepId="euclid_verify_photometry" />

        <Card>
          <CardHead
            title="Euclid archive"
            sub="log in to the Euclid Science Archive for this WebUI session"
          />
          <CardBody>
            {auth.data?.authenticated ? (
              <div className="row" style={{ gap: "var(--s3)", alignItems: "center" }}>
                <span className="muted">
                  Logged in to the Euclid archive
                  {auth.data.user ? ` as ${auth.data.user}` : ""}.
                </span>
                <Button onClick={logout} disabled={authBusy}>Logout</Button>
              </div>
            ) : (
              <div style={{ maxWidth: 360 }}>
                <Field label="Username">
                  <Input value={user} onChange={setUser} placeholder="username" />
                </Field>
                <Field label="Password">
                  <Input
                    value={pwd}
                    onChange={setPwd}
                    type="password"
                    placeholder="password"
                    onEnter={login}
                  />
                </Field>
                <div className="row" style={{ marginTop: "var(--s3)" }}>
                  <Button
                    variant="primary"
                    onClick={login}
                    disabled={authBusy || !user.trim() || !pwd}
                  >
                    Login
                  </Button>
                </div>
              </div>
            )}
            {authNote && (
              <div
                className={`job-panel job-panel--${authNote.ok ? "done" : "err"}`}
                style={{ marginTop: "var(--s3)" }}
              >
                <LogTail text={authNote.text} />
              </div>
            )}
          </CardBody>
        </Card>
      </div>
    </Page>
  );
}
