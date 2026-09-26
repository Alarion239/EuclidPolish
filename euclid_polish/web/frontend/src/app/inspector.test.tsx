import { act, render, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter, useLocation, useNavigate } from "react-router-dom";
import { afterEach, describe, expect, it } from "vitest";
import { useInspector } from "../state/inspector";
import {
  closeInspector, inspectHref, inspectorTitle, openInspector, registerInspector,
  useInspectorKind, useInspectorRegistry, useInspectorUrlSync,
} from "./inspector";

afterEach(() => {
  useInspector.getState().reset();
  useInspectorRegistry.getState().reset();
});

function Member({ id }: { id: string }) { return <span>member {id}</span>; }

describe("registry", () => {
  it("registers a kind with its title and unregisters", () => {
    const off = registerInspector("member", Member, { title: (id) => `Member ${id}` });
    const { result } = renderHook(() => useInspectorKind("member"));
    expect(result.current?.Component).toBe(Member);
    expect(inspectorTitle({ kind: "member", id: "m_1" })).toBe("Member m_1");
    off();
    expect(useInspectorRegistry.getState().kinds.member).toBeUndefined();
    // unknown kinds get a generic title
    expect(inspectorTitle({ kind: "fits", id: "a/b.fits" })).toBe("fits · a/b.fits");
  });

  it("a later registration of the same kind wins; unregistering it restores the earlier one", () => {
    function Other() { return null; }
    const offA = registerInspector("tile", Member);
    const offB = registerInspector("tile", Other, { title: "Tile" });
    expect(useInspectorRegistry.getState().kinds.tile.Component).toBe(Other);
    offB();
    expect(useInspectorRegistry.getState().kinds.tile.Component).toBe(Member);
    offA();
    expect(useInspectorRegistry.getState().kinds.tile).toBeUndefined();
  });
});

describe("open / close", () => {
  it("openInspector shows the target; closeInspector hides it", () => {
    openInspector({ kind: "member", id: "m_1" });
    expect(useInspector.getState()).toMatchObject({ open: true, current: { kind: "member", id: "m_1" } });
    closeInspector();
    expect(useInspector.getState().open).toBe(false);
  });

  it("builds a shareable href keeping the other params", () => {
    expect(inspectHref({ kind: "tile", id: "nexus/12" }, { pathname: "/sky/atlas", search: "?ra=1" }))
      .toBe("/sky/atlas?ra=1&inspect=tile%3Anexus%2F12");
  });
});

/* ── ?inspect= two-way sync ─────────────────────────────────────────────── */

type Probe = { loc: ReturnType<typeof useLocation>; nav: ReturnType<typeof useNavigate> };

function mount(url: string) {
  const probe = {} as Probe;
  function Sync({ children }: { children?: ReactNode }) {
    useInspectorUrlSync();
    probe.loc = useLocation();
    probe.nav = useNavigate();
    return <>{children}</>;
  }
  const r = render(
    <MemoryRouter initialEntries={[url]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><Sync /></MemoryRouter>,
  );
  return { probe, ...r };
}

const search = (p: Probe) => new URLSearchParams(p.loc.search).get("inspect");

describe("useInspectorUrlSync", () => {
  it("opens the inspector from ?inspect= on load", () => {
    mount("/ensemble/starfull/members?inspect=member:member_196");
    expect(useInspector.getState()).toMatchObject({ open: true, current: { kind: "member", id: "member_196" } });
  });

  it("writes the store to the URL (and removes it on close), keeping other params", () => {
    const { probe } = mount("/sky/atlas?ra=10");
    act(() => openInspector({ kind: "tile", id: "nexus/3" }));
    expect(search(probe)).toBe("tile:nexus/3");
    expect(probe.loc.search).toContain("ra=10");
    act(() => closeInspector());
    expect(search(probe)).toBeNull();
    expect(probe.loc.search).toBe("?ra=10");
  });

  it("follows back/forward (URL wins on POP)", () => {
    const { probe } = mount("/sky/results");
    act(() => probe.nav("/sky/results?inspect=job:local/abc"));
    expect(useInspector.getState().current).toEqual({ kind: "job", id: "local/abc" });
    act(() => probe.nav("/sky/results?inspect=job:local/def"));
    expect(useInspector.getState().current).toEqual({ kind: "job", id: "local/def" });
    act(() => probe.nav(-1));
    expect(useInspector.getState().current).toEqual({ kind: "job", id: "local/abc" });
    act(() => probe.nav(-1));
    expect(useInspector.getState().open).toBe(false);
  });

  it("keeps the inspector open across a push navigation without the param", () => {
    const { probe } = mount("/data/records?inspect=member:m_2");
    act(() => probe.nav("/ops/jobs"));
    expect(probe.loc.pathname).toBe("/ops/jobs");
    expect(search(probe)).toBe("member:m_2");
    expect(useInspector.getState().open).toBe(true);
  });

  it("drops a malformed param", () => {
    const { probe } = mount("/sky/atlas?inspect=nonsense");
    expect(useInspector.getState().open).toBe(false);
    expect(search(probe)).toBeNull();
  });

  it("a push navigation WITH a param opens that target", () => {
    const { probe } = mount("/sky/atlas?inspect=member:a");
    act(() => probe.nav("/ops/jobs?inspect=job:local/z"));
    expect(useInspector.getState().current).toEqual({ kind: "job", id: "local/z" });
  });
});
