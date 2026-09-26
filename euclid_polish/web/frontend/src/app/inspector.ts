/* Inspector registry + URL sync (contract C8).
 *
 *   registerInspector("member", MemberInspector, { title: (id) => `Member ${id}` });
 *   openInspector({ kind: "member", id: "member_196" });   // opens the right-hand panel
 *   closeInspector();
 *
 * The state (current target, back/forward history, pins) lives in
 * `state/inspector.ts`; this module adds:
 *   - the kind → component registry (a workspace registers its kinds when its
 *     module loads; the shell registers `job`). A later registration of a kind
 *     wins until it is unregistered, then the earlier one comes back;
 *   - `useInspectorUrlSync()` (mounted once by the shell): `?inspect=kind:id`
 *     mirrors the store both ways. On load and on back/forward the URL wins; a
 *     push/replace navigation that carries the param opens it; one without it
 *     keeps the open inspector and re-adds the param, so an inspected entity
 *     survives moving between workspaces and stays shareable. Store changes
 *     (row clicks through `DataTable inspect`, `openInspector`, close) are
 *     written to the URL in place (history replace).
 */
import type { ComponentType } from "react";
import { useEffect, useRef } from "react";
import { useLocation, useNavigationType } from "react-router-dom";
import { create } from "zustand";
import { useUrlState } from "../hooks/useUrlState";
import {
  formatInspectParam, parseInspectParam, useInspector, type InspectTarget,
} from "../state/inspector";

export type { InspectTarget };

export type InspectorProps = { id: string };
export type InspectorComponent = ComponentType<InspectorProps>;

export type InspectorRegistration = {
  kind: string;
  Component: InspectorComponent;
  /** Panel title: a string, or a function of the id. */
  title?: string | ((id: string) => string);
};

type Registry = {
  /** Active registration per kind (the newest). */
  kinds: Record<string, InspectorRegistration>;
  /** Every registration per kind, oldest first. */
  stacks: Record<string, { token: number; reg: InspectorRegistration }[]>;
  reset: () => void;
};

export const useInspectorRegistry = create<Registry>()((set) => ({
  kinds: {},
  stacks: {},
  reset: () => set({ kinds: {}, stacks: {} }),
}));

let nextToken = 1;

function activeKinds(stacks: Registry["stacks"]): Registry["kinds"] {
  const kinds: Registry["kinds"] = {};
  for (const [kind, stack] of Object.entries(stacks)) {
    if (stack.length) kinds[kind] = stack[stack.length - 1].reg;
  }
  return kinds;
}

/** Register the component that inspects `kind` targets; returns the unregister. */
export function registerInspector(
  kind: string,
  Component: InspectorComponent,
  opts: { title?: InspectorRegistration["title"] } = {},
): () => void {
  const token = nextToken++;
  const reg: InspectorRegistration = { kind, Component, title: opts.title };
  const { stacks } = useInspectorRegistry.getState();
  const next = { ...stacks, [kind]: [...(stacks[kind] ?? []), { token, reg }] };
  useInspectorRegistry.setState({ stacks: next, kinds: activeKinds(next) });
  return () => {
    const cur = useInspectorRegistry.getState().stacks;
    const stack = (cur[kind] ?? []).filter((e) => e.token !== token);
    const after = { ...cur };
    if (stack.length) after[kind] = stack; else delete after[kind];
    useInspectorRegistry.setState({ stacks: after, kinds: activeKinds(after) });
  };
}

/** The registration for `kind` (re-renders when it changes). */
export function useInspectorKind(kind: string | null | undefined): InspectorRegistration | null {
  return useInspectorRegistry((s) => (kind ? s.kinds[kind] ?? null : null));
}

/** The panel title of a target: the registered title, else "kind · id". */
export function inspectorTitle(t: InspectTarget): string {
  const reg = useInspectorRegistry.getState().kinds[t.kind];
  const title = reg?.title;
  if (typeof title === "function") return title(t.id);
  if (typeof title === "string") return `${title} · ${t.id}`;
  return `${t.kind} · ${t.id}`;
}

export function openInspector(target: InspectTarget, opts?: { replace?: boolean }): void {
  useInspector.getState().show(target, opts);
}

export function closeInspector(): void {
  useInspector.getState().hide();
}

/** A link to `location` with `?inspect=` set to `target` (other params kept). */
export function inspectHref(target: InspectTarget, location: { pathname: string; search?: string; hash?: string }): string {
  const params = new URLSearchParams(location.search ?? "");
  params.set("inspect", formatInspectParam(target));
  return `${location.pathname}?${params.toString()}${location.hash ?? ""}`;
}

/** Keep `?inspect=` and the inspector store in step (mount once, in the shell). */
export function useInspectorUrlSync(): void {
  const [param, setParam] = useUrlState("inspect", "");
  const location = useLocation();
  const navType = useNavigationType();
  const open = useInspector((s) => s.open);
  const current = useInspector((s) => s.current);
  const storeValue = open && current ? formatInspectParam(current) : "";
  const seen = useRef<{ key: string | null; store: string }>({ key: null, store: storeValue });

  useEffect(() => {
    const locationChanged = seen.current.key !== location.key;
    const storeChanged = seen.current.store !== storeValue;
    seen.current = { key: location.key, store: storeValue };
    if (param === storeValue) return;
    if (storeChanged && !locationChanged) {
      setParam(storeValue);
      return;
    }
    if (navType === "POP" || param) {
      const target = parseInspectParam(param);
      if (target) useInspector.getState().show(target);
      else if (param) setParam(storeValue); // malformed: drop (or restore the open one)
      else useInspector.getState().hide();
      return;
    }
    setParam(storeValue);
  }, [location.key, storeValue, param, navType, setParam]);
}
