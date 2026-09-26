/* confirm(): a promise-based confirmation dialog replacing window.confirm.

     if (!(await confirm({ title: "Archive 3 members?", tone: "danger",
                           confirmLabel: "Archive" }))) return;

   Requests queue in a tiny store and are shown one at a time by
   <ConfirmHost/> (mounted by UiProvider, inside the router). When no host is
   mounted (a test, an isolated widget) the first call mounts one into
   document.body as a SEPARATE React root with no app context: keep `title`
   and `message` context-free (text, plain elements; no router <Link>, no
   hooks that read a provider). */
import { useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { Button } from "./Button";
import { Input } from "./controls";
import { Dialog } from "./overlays";

export type ConfirmOptions = {
  title: ReactNode;
  /** Body text. */
  message?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  /** "danger" styles the confirm button red and focuses Cancel first. */
  tone?: "default" | "danger";
  /** Require typing this exact text before confirming (e.g. a run id). */
  requireText?: string;
};

type Pending = ConfirmOptions & { id: number; resolve: (ok: boolean) => void };
type Snapshot = { queue: Pending[]; explicitHosts: number };

let state: Snapshot = { queue: [], explicitHosts: 0 };
let nextId = 1;
const listeners = new Set<() => void>();
const setState = (patch: Partial<Snapshot>) => {
  state = { ...state, ...patch };
  for (const l of listeners) l();
};
const subscribe = (l: () => void) => { listeners.add(l); return () => { listeners.delete(l); }; };
const getSnapshot = () => state;

let autoHost: { root: ReturnType<typeof createRoot>; el: HTMLElement } | null = null;

function ensureHost() {
  if (state.explicitHosts > 0 || autoHost || typeof document === "undefined") return;
  const el = document.createElement("div");
  el.setAttribute("data-ui-confirm-host", "");
  document.body.appendChild(el);
  const root = createRoot(el);
  autoHost = { root, el };
  root.render(<ConfirmHost auto />);
}

/** Ask the user to confirm; resolves true on confirm, false on cancel/Escape. */
export function confirm(opts: ConfirmOptions | string): Promise<boolean> {
  const o: ConfirmOptions = typeof opts === "string" ? { title: opts } : opts;
  return new Promise<boolean>((resolve) => {
    setState({ queue: [...state.queue, { ...o, id: nextId++, resolve }] });
    ensureHost();
  });
}

function settle(id: number, ok: boolean) {
  const item = state.queue.find((p) => p.id === id);
  if (!item) return;
  setState({ queue: state.queue.filter((p) => p.id !== id) });
  item.resolve(ok);
}

/** Test/teardown helper: cancel every pending confirmation and unmount the
 *  auto-mounted host. */
export function resetConfirm(): void {
  const pending = state.queue;
  setState({ queue: [] });
  for (const p of pending) p.resolve(false);
  if (autoHost) {
    const { root, el } = autoHost;
    autoHost = null;
    root.unmount();
    el.remove();
  }
}

/** Renders the oldest pending confirmation. Mount once (UiProvider does). */
export function ConfirmHost({ auto = false }: { auto?: boolean }) {
  const snap = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  useEffect(() => {
    if (auto) return;
    setState({ explicitHosts: state.explicitHosts + 1 });
    return () => setState({ explicitHosts: state.explicitHosts - 1 });
  }, [auto]);
  const current = snap.queue[0];
  // An explicitly mounted host supersedes the automatic one.
  if (!current || (auto && snap.explicitHosts > 0)) return null;
  return <ConfirmDialog key={current.id} item={current} />;
}

function ConfirmDialog({ item }: { item: Pending }) {
  const [typed, setTyped] = useState("");
  const okRef = useRef<HTMLButtonElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const blocked = item.requireText != null && typed !== item.requireText;
  const ok = () => { if (!blocked) settle(item.id, true); };
  // role=alertdialog, and the message is the accessible description, so a
  // screen reader announces the question and its consequence together.
  return (
    <Dialog open size="sm" title={item.title} className="ui-confirm" role="alertdialog"
      description={item.message ?? undefined}
      onOpenChange={(open) => { if (!open) settle(item.id, false); }}
      onOpenAutoFocus={(e) => {
        e.preventDefault();
        const target = item.requireText != null ? inputRef.current
          : item.tone === "danger" ? cancelRef.current : okRef.current;
        target?.focus();
      }}
      footer={<>
        <Button ref={cancelRef} variant="ghost" onClick={() => settle(item.id, false)}>
          {item.cancelLabel ?? "Cancel"}
        </Button>
        <Button ref={okRef} variant={item.tone === "danger" ? "danger" : "primary"} disabled={blocked}
          onClick={ok}>
          {item.confirmLabel ?? "Confirm"}
        </Button>
      </>}>
      {item.requireText == null ? null : (
        <label className="ui-confirm__require">
          <span>Type <code>{item.requireText}</code> to confirm</span>
          <Input ref={inputRef} value={typed} onChange={setTyped} onEnter={ok} aria-label="Confirmation text" />
        </label>
      )}
    </Dialog>
  );
}
