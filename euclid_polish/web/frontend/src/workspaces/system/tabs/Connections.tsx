/* System › Connections (`/system/connections`): every session and credential
 * the console uses, reachable offline, as one column of cards.
 *  - FASRC: the SSH ControlMaster (connect / test / retry / disconnect with
 *    the real `last_error`, C4) and its settings (GET/POST /api/fasrc/config,
 *    dirty-only saves).
 *  - The ONE laptop-side Euclid archive session (/auth/*) every local archive
 *    query reads, with the pages that need it.
 *  - FASRC-side Euclid credentials (/euclid-auth/*, read by the cutout
 *    download on FASRC) and the TNG API token (/tng-auth/*). Passwords and
 *    tokens are sent once and never shown again. */
import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { ApiError, apiPost, isFasrcOffline } from "../../../api/client";
import { invalidate, useResource } from "../../../api/query";
import { usePageActions } from "../../../app/palette";
import { FASRC_STATUS_URL, useFasrcStatus } from "../../../app/status";
import { useUrlState } from "../../../hooks/useUrlState";
import { formatDateTime, formatRelative } from "../../../format";
import {
  Badge, Button, Callout, Caption, Card, CardBody, CardHead, DefList, Details, Field, Input, NumberField, Page, Section,
  Skeleton, confirm, toast,
} from "../../../ui";
import { dirtyFields, toForm, type FormState } from "../configModel";
import { PageLead } from "../../shared/PageLead";
import "../system.css";

type Reply = { ok?: boolean; error?: string } & Record<string, unknown>;

const errText = (e: unknown) => (e instanceof ApiError || e instanceof Error ? e.message : String(e));

/** POST and toast; returns the reply (or null on a failure, already toasted). */
async function post(url: string, data: Record<string, string>, label: string): Promise<Reply | null> {
  try {
    const r = await apiPost<Reply>(url, data);
    if (r?.ok === false) { toast.error(`${label} failed`, { description: r.error ?? "refused" }); return null; }
    return r ?? {};
  } catch (e) {
    toast.error(`${label} failed`, {
      description: isFasrcOffline(e) ? "FASRC is not connected — connect first." : errText(e),
    });
    return null;
  }
}

/* ── FASRC ───────────────────────────────────────────────────────────────── */

type FasrcField = { name: string; label: string; hint: string; kind?: "int"; group: "ssh" | "paths" | "local" };
const FASRC_FIELDS: FasrcField[] = [
  { name: "ssh_user", label: "SSH user", group: "ssh", hint: "Your FASRC username. Authentication is public-key only: install your key once with ssh-copy-id." },
  { name: "ssh_host", label: "Login host", group: "ssh", hint: "The FASRC login node the ControlMaster connects to." },
  { name: "control_socket", label: "Control socket", group: "ssh", hint: "Local path of the SSH ControlMaster socket every command reuses." },
  { name: "control_persist", label: "Control persist", group: "ssh", hint: "How long the master stays up after the last command (ssh ControlPersist, e.g. 8h)." },
  { name: "repo_path", label: "Repo checkout", group: "paths", hint: "EuclidPolish checkout on holylabs that jobs run from." },
  { name: "conda_env_path", label: "Conda env", group: "paths", hint: "Prefix of the conda environment jobs activate." },
  { name: "data_dir", label: "Data dir", group: "paths", hint: "Remote data root (netscratch); jobs read and write under it." },
  { name: "ckpt_dir", label: "Checkpoint dir", group: "paths", hint: "Remote checkpoint root (the ensemble lives next to it)." },
  { name: "logs_subdir", label: "Logs subdir", group: "paths", hint: "Job logs, relative to the repo checkout." },
  { name: "tracking_remote_dir", label: "Tracking mirror", group: "paths", hint: "Persistent holylabs mirror of ./tracking. Blank = <repo>/tracking (never netscratch: it is purged)." },
  { name: "local_ckpt_mirror", label: "Local checkpoint mirror", group: "local", hint: "Where pulled checkpoints land on this laptop. Blank = the default checkpoint dir." },
];
const GROUP_TITLE = { ssh: "SSH", paths: "Remote paths", local: "This laptop" } as const;

function FasrcSettings({ onSaved }: { onSaved: (connect: boolean) => void }) {
  const cfg = useResource<Record<string, string | number>>("/api/fasrc/config", [], { ttl: 0 });
  const [loaded, setLoaded] = useState<FormState | null>(null);
  const [form, setForm] = useState<FormState | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (cfg.data && !loaded) { const f = toForm(cfg.data); setLoaded(f); setForm(f); }
  }, [cfg.data, loaded]);
  if (cfg.loading && !form) return <Skeleton lines={4} />;
  if (cfg.error && !form) return <Callout tone="bad" title="Could not read the FASRC settings">{cfg.error.message}</Callout>;
  if (!form || !loaded) return null;
  const dirty = dirtyFields(form, loaded);

  async function save(connect: boolean) {
    if (!form || !loaded) return;
    setBusy(true);
    const body: Record<string, string> = {};
    for (const k of dirty) body[k] = form[k];
    const r = await post("/api/fasrc/config", body, "Saving the FASRC settings");
    setBusy(false);
    if (!r) return;
    const next = toForm(r as Record<string, string>);
    setLoaded(next); setForm(next);
    toast.success(dirty.length ? `Saved ${dirty.length} FASRC setting${dirty.length === 1 ? "" : "s"}` : "FASRC settings unchanged");
    onSaved(connect);
  }

  const groups = (["ssh", "paths", "local"] as const).map((g) => ({ g, fields: FASRC_FIELDS.filter((f) => f.group === g) }));
  const known = new Set(FASRC_FIELDS.map((f) => f.name));
  const extra = Object.keys(form).filter((k) => !known.has(k));
  return (
    <form className="sys-stack" onSubmit={(e: FormEvent) => { e.preventDefault(); void save(false); }}>
      {groups.map(({ g, fields }) => (
        <div key={g} className="conn-form">
          <h4 className="conn-section-title">{GROUP_TITLE[g]}</h4>
          {fields.filter((f) => f.name in form).map((f) => (
            <Field key={f.name} label={f.label} hint={f.hint}>
              {f.kind === "int"
                ? <NumberField value={form[f.name]} onChange={(v) => setForm({ ...form, [f.name]: v })} min={0} aria-label={f.label} />
                : <Input value={form[f.name]} onChange={(v) => setForm({ ...form, [f.name]: v })} aria-label={f.label}
                    spellCheck={false} autoComplete="off" />}
            </Field>
          ))}
        </div>
      ))}
      {extra.length > 0 && (
        <div className="conn-form">
          <h4 className="conn-section-title">Other</h4>
          {extra.map((k) => (
            <Field key={k} label={k}><Input value={form[k]} onChange={(v) => setForm({ ...form, [k]: v })} aria-label={k} /></Field>
          ))}
        </div>
      )}
      <div className="sys-row">
        {dirty.length > 0 && <Badge tone="warn">{dirty.length} unsaved</Badge>}
        <span className="sys-bar__spacer" />
        <Button size="sm" variant="ghost" disabled={!dirty.length} onClick={() => setForm(loaded)}>Discard</Button>
        <Button size="sm" type="submit" loading={busy} disabled={!dirty.length}>Save settings</Button>
        <Button size="sm" variant="primary" loading={busy} onClick={() => void save(true)}>Save & connect</Button>
      </div>
    </form>
  );
}

function FasrcCard() {
  const status = useFasrcStatus();
  const s = status.data;
  const connected = !!s?.ssh_connected;
  const [busy, setBusy] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useUrlState("ssh", false);

  async function act(url: string, label: string, success: string) {
    setBusy(url);
    const r = await post(url, {}, label);
    setBusy(null);
    void invalidate(FASRC_STATUS_URL);
    if (r) toast.success(success);
  }
  const connect = () => act("/api/fasrc/connect", "Connecting to FASRC", "FASRC connected");
  const disconnect = async () => {
    if (!(await confirm({ title: "Disconnect from FASRC?", message: "Cluster views and FASRC jobs stop updating until you reconnect; local pages keep working.", confirmLabel: "Disconnect" }))) return;
    await act("/api/fasrc/disconnect", "Disconnecting", "FASRC disconnected");
  };
  usePageActions([
    { id: "conn:connect", label: connected ? "Test the FASRC connection" : "Connect to FASRC", group: "Connections",
      keywords: ["ssh", "fasrc", "connect"], run: () => { void connect(); } },
    { id: "conn:disconnect", label: "Disconnect from FASRC", group: "Connections", disabled: !connected, run: () => { void disconnect(); } },
    { id: "conn:ssh-settings", label: "Edit the FASRC SSH settings", group: "Connections", keywords: ["ssh", "user", "host", "paths"],
      run: () => setShowSettings(true) },
  ]);

  return (
    <Card>
      <CardHead title="FASRC" sub="SSH ControlMaster to the login node"
        right={status.loading && !s ? <Badge>…</Badge> : !connected ? <Badge tone="bad" dot>Offline</Badge> : undefined} />
      <CardBody>
        <div className="sys-stack">
          {s && (
            <Caption>
              {connected
                ? <span title={s.connected_at ? formatDateTime(s.connected_at) : undefined}>
                  {s.connected_at ? `Connected ${formatRelative(s.connected_at)}.` : "Connected."}
                </span>
                : "Not connected: local pages keep working; cluster views and FASRC jobs wait for it."}
            </Caption>
          )}
          {s?.last_error && !connected && (
            <Callout tone="bad" title="Last connection error">
              <pre className="conn-error">{s.last_error}</pre>
            </Callout>
          )}
          {status.error && !s && <Callout tone="bad" title="Could not read the connection state">{status.error.message}</Callout>}
          <div className="sys-row">
            <Button variant={connected ? "default" : "primary"} size="sm" loading={busy === "/api/fasrc/connect"} onClick={() => void connect()}>
              {connected ? "Test connection" : "Connect"}
            </Button>
            {!connected && (
              <Button size="sm" loading={busy === "/api/connection/retry"}
                onClick={() => void act("/api/connection/retry", "Retrying the startup connect", "FASRC connected")}>
                Retry startup connect
              </Button>
            )}
            {connected && (
              <Button size="sm" variant="ghost" loading={busy === "/api/fasrc/disconnect"} onClick={() => void disconnect()}>Disconnect</Button>
            )}
          </div>
          <Section title="SSH settings" collapsible open={showSettings} onOpenChange={setShowSettings}
            sub="~/.euclid_polish/fasrc.json">
            <FasrcSettings onSaved={(andConnect) => { if (andConnect) void connect(); }} />
          </Section>
          {s?.socket && (
            <Details summary="Control socket"><code className="mono">{s.socket}</code></Details>
          )}
        </div>
      </CardBody>
    </Card>
  );
}

/* ── the one laptop-side Euclid archive session ──────────────────────────── */

type AuthStatus = {
  authenticated: boolean; user: string | null; logged_in_at?: string | null;
  used_by?: { id: string; label: string; to: string }[];
};

function EuclidSessionCard() {
  const auth = useResource<AuthStatus>("/auth/status", [], { ttl: 30_000 });
  const a = auth.data;
  const [user, setUser] = useState("");
  const [pwd, setPwd] = useState("");
  const [busy, setBusy] = useState(false);
  const userRef = useRef<HTMLInputElement | null>(null);

  async function login(e?: FormEvent) {
    e?.preventDefault();
    if (!user.trim() || !pwd) { toast.warning("Enter the archive username and password"); return; }
    setBusy(true);
    const r = await post("/auth/login", { username: user.trim(), password: pwd }, "Euclid archive login");
    setBusy(false);
    setPwd("");
    void invalidate("/auth/status");
    if (r) toast.success(`Logged in to the Euclid archive as ${user.trim()}`);
  }
  async function logout() {
    setBusy(true);
    await post("/auth/logout", {}, "Euclid archive logout");
    setBusy(false);
    void invalidate("/auth/status");
    toast.info("Logged out of the Euclid archive");
  }
  usePageActions([
    { id: "conn:euclid-login", label: "Log in to the Euclid archive", group: "Connections", keywords: ["euclid", "archive", "login"],
      disabled: !!a?.authenticated, run: () => userRef.current?.focus() },
    { id: "conn:euclid-logout", label: "Log out of the Euclid archive", group: "Connections", disabled: !a?.authenticated,
      run: () => { void logout(); } },
  ]);

  return (
    <Card>
      <CardHead title="Euclid archive · this laptop" sub="the one session every local archive query uses"
        right={a ? <Badge tone={a.authenticated ? "good" : "neutral"} dot>{a.authenticated ? a.user ?? "logged in" : "logged out"}</Badge> : undefined} />
      <CardBody>
        <div className="sys-stack">
          {auth.loading && !a && <Skeleton lines={2} />}
          {auth.error && !a && <Callout tone="bad" title="Could not read the archive session">{auth.error.message}</Callout>}
          {a?.authenticated ? (
            <>
              <DefList dense items={[
                ["user", <code className="mono">{a.user}</code>],
                a.logged_in_at ? ["since", `${formatDateTime(a.logged_in_at)} (${formatRelative(a.logged_in_at)})`] : null,
              ]} />
              <div className="sys-row"><Button size="sm" loading={busy} onClick={() => void logout()}>Log out</Button></div>
            </>
          ) : a ? (
            <form className="conn-form" onSubmit={(e) => void login(e)}>
              <Field label="Username">
                <Input ref={userRef} value={user} onChange={setUser} autoComplete="username" spellCheck={false} />
              </Field>
              <Field label="Password">
                <Input type="password" value={pwd} onChange={setPwd} autoComplete="current-password" />
              </Field>
              <div className="conn-form__wide sys-row">
                <Button size="sm" variant="primary" type="submit" loading={busy}>Log in</Button>
                <span className="muted">The password is sent to the archive, never stored.</span>
              </div>
            </form>
          ) : null}
          {a?.used_by && a.used_by.length > 0 && (
            <div className="conn-uses" aria-label="Pages that use this session">
              {a.used_by.map((u) => <Link key={u.id} to={u.to} className="cfg-chip">{u.label}</Link>)}
            </div>
          )}
        </div>
      </CardBody>
    </Card>
  );
}

/* ── FASRC-side credentials ──────────────────────────────────────────────── */

type RemoteSecretStatus = { present: boolean; connected: boolean; user?: string | null; chars?: number };

function RemoteSecretCard({ kind }: { kind: "euclid" | "tng" }) {
  const url = kind === "euclid" ? "/euclid-auth/status" : "/tng-auth/status";
  const st = useResource<RemoteSecretStatus>(url, [], { ttl: 60_000 });
  const fasrc = useFasrcStatus();
  const online = !!fasrc.data?.ssh_connected;
  const s = st.data;
  const [user, setUser] = useState("");
  const [secret, setSecret] = useState("");
  const [busy, setBusy] = useState(false);
  // The status is "not connected" while FASRC is down: re-read it on reconnect.
  useEffect(() => { if (online) void invalidate(url); }, [online, url]);

  async function save(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    const r = kind === "euclid"
      ? await post("/euclid-auth/save", { euclid_user: user.trim(), euclid_password: secret }, "Saving the FASRC Euclid credentials")
      : await post("/tng-auth/save", { tng_token: secret.trim() }, "Saving the TNG token");
    setBusy(false);
    setSecret("");
    void invalidate(url);
    if (r) toast.success(kind === "euclid" ? "Euclid credentials written on FASRC" : `TNG token written on FASRC (${r.chars ?? "?"} chars)`);
  }

  const title = kind === "euclid" ? "Euclid credentials · FASRC" : "TNG API token · FASRC";
  const sub = kind === "euclid" ? "~/.euclid_credentials, read by the cutout download" : "read by the TNG atlas and radius jobs";
  const state = !s ? null : !s.connected ? "unknown (offline)" : s.present
    ? (kind === "euclid" ? `set for ${s.user}` : `set (${s.chars} chars)`) : "not set";
  return (
    <Card>
      <CardHead title={title} sub={sub}
        right={s ? <Badge tone={!s.connected ? "neutral" : s.present ? "good" : "warn"} dot>{state}</Badge> : undefined} />
      <CardBody>
        {!online && (
          <Callout tone="info" title="Needs FASRC">Connect to FASRC to read or write this file.</Callout>
        )}
        {st.error && !s && <Callout tone="bad" title="Could not read the status">{st.error.message}</Callout>}
        <form className="conn-form" onSubmit={(e) => void save(e)} aria-label={title}>
          {kind === "euclid" && (
            <Field label="Archive username">
              <Input value={user} onChange={setUser} disabled={!online} autoComplete="off" spellCheck={false} />
            </Field>
          )}
          <Field label={kind === "euclid" ? "Archive password" : "Token"}
            hint={kind === "euclid" ? "Written mode 600 on FASRC through the SSH channel (never in a process argument or the job DB)." : "Your TNG API key; written mode 600 on FASRC and never shown again."}>
            <Input type="password" value={secret} onChange={setSecret} disabled={!online} autoComplete="new-password" />
          </Field>
          <div className="conn-form__wide sys-row">
            <Button size="sm" type="submit" loading={busy}
              disabled={!online || !secret || (kind === "euclid" && !user.trim())}>
              {s?.present ? "Replace" : "Save"}
            </Button>
          </div>
        </form>
      </CardBody>
    </Card>
  );
}

export default function Connections() {
  return (
    <Page className="sys-page">
      <PageLead>Sessions and credentials. Local pages work with all of them down.</PageLead>
      <div className="sys-column">
        <FasrcCard />
        <EuclidSessionCard />
        <RemoteSecretCard kind="euclid" />
        <RemoteSecretCard kind="tng" />
      </div>
    </Page>
  );
}
