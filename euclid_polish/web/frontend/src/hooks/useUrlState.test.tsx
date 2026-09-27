import { act, render, renderHook } from "@testing-library/react";
import { useContext, type ReactNode } from "react";
import {
  MemoryRouter, RouterProvider, UNSAFE_NavigationContext, createMemoryRouter, useLocation, useNavigationType,
} from "react-router-dom";
import { describe, expect, it } from "vitest";
import { useUrlState } from "./useUrlState";

const at = (url: string) => ({ children }: { children: ReactNode }) => (
  <MemoryRouter initialEntries={[url]}>{children}</MemoryRouter>
);

describe("useUrlState", () => {
  it("reads the param, falling back to the default", () => {
    const a = renderHook(() => useUrlState("tab", "overview"), { wrapper: at("/ensemble/starfull") });
    expect(a.result.current[0]).toBe("overview");
    const b = renderHook(() => useUrlState("tab", "overview"), { wrapper: at("/x?tab=members") });
    expect(b.result.current[0]).toBe("members");
  });

  it("writes non-defaults, removes defaults and keeps other params + hash", () => {
    const { result } = renderHook(() => ({ s: useUrlState("tab", "overview"), loc: useLocation() }), {
      wrapper: at("/sky/atlas?ra=10&tab=x#frag"),
    });
    act(() => result.current.s[1]("members"));
    expect(result.current.loc.pathname).toBe("/sky/atlas");
    expect(result.current.loc.search).toBe("?ra=10&tab=members");
    expect(result.current.loc.hash).toBe("#frag");
    expect(result.current.s[0]).toBe("members");
    act(() => result.current.s[1]("overview"));
    expect(result.current.loc.search).toBe("?ra=10");
  });

  it("infers number, boolean and list codecs from the default", () => {
    const { result } = renderHook(() => ({
      i: useUrlState("i", 0),
      bad: useUrlState("bad", 7),
      on: useUrlState("on", false),
      tiers: useUrlState<string[]>("tiers", ["lr", "sr"]),
      loc: useLocation(),
    }), { wrapper: at("/x?i=5&bad=abc&on=1&tiers=lr,hr,sr") });
    expect(result.current.i[0]).toBe(5);
    expect(result.current.bad[0]).toBe(7);
    expect(result.current.on[0]).toBe(true);
    expect(result.current.tiers[0]).toEqual(["lr", "hr", "sr"]);
    act(() => result.current.on[1](false));
    act(() => result.current.tiers[1](["lr", "sr"]));
    expect(result.current.loc.search).toBe("?i=5&bad=abc");
  });

  it("keeps a stable reference for unchanged list values", () => {
    const { result, rerender } = renderHook(() => useUrlState<string[]>("t", []), { wrapper: at("/x?t=a,b") });
    const first = result.current[0];
    rerender();
    expect(result.current[0]).toBe(first);
  });

  it("supports custom parse/serialize", () => {
    type View = { ra: number; dec: number };
    const parse = (raw: string): View | undefined => {
      const [ra, dec] = raw.split(",").map(Number);
      return Number.isFinite(ra) && Number.isFinite(dec) ? { ra, dec } : undefined;
    };
    const serialize = (v: View) => `${v.ra},${v.dec}`;
    const { result } = renderHook(() => ({
      v: useUrlState<View>("c", { ra: 0, dec: 0 }, { parse, serialize }),
      loc: useLocation(),
    }), { wrapper: at("/sky?c=53.16,-27.78") });
    expect(result.current.v[0]).toEqual({ ra: 53.16, dec: -27.78 });
    act(() => result.current.v[1]({ ra: 1, dec: 2 }));
    expect(result.current.loc.search).toBe("?c=1%2C2");
  });

  it("coalesces several setters called in the same tick", () => {
    const { result } = renderHook(() => ({
      a: useUrlState("a", ""),
      b: useUrlState("b", ""),
      loc: useLocation(),
    }), { wrapper: at("/x") });
    act(() => { result.current.a[1]("1"); result.current.b[1]("2"); });
    expect(result.current.loc.search).toBe("?a=1&b=2");
  });

  it("accepts functional updates", () => {
    const { result } = renderHook(() => useUrlState("n", 0), { wrapper: at("/x?n=2") });
    act(() => result.current[1]((n) => n + 1));
    act(() => result.current[1]((n) => n + 1));
    expect(result.current[0]).toBe(4);
  });

  it("replaces history by default and pushes when asked", () => {
    const replaced = renderHook(() => ({ s: useUrlState("a", ""), nav: useNavigationType() }), { wrapper: at("/x") });
    act(() => replaced.result.current.s[1]("1"));
    expect(replaced.result.current.nav).toBe("REPLACE");
    const pushed = renderHook(() => ({ s: useUrlState("a", "", { replace: false }), nav: useNavigationType() }), { wrapper: at("/x") });
    act(() => pushed.result.current.s[1]("1"));
    expect(pushed.result.current.nav).toBe("PUSH");
  });

  it("records ONE history entry for several push-mode setters in one tick", () => {
    const { result } = renderHook(() => ({
      a: useUrlState("a", 0, { replace: false }),
      b: useUrlState("b", 0, { replace: false }),
      loc: useLocation(),
      history: useContext(UNSAFE_NavigationContext).navigator as unknown as { index: number; go: (n: number) => void },
    }), { wrapper: at("/x") });
    act(() => { result.current.a[1](1); result.current.b[1](2); });
    expect(result.current.loc.search).toBe("?a=1&b=2");
    expect(result.current.history.index).toBe(1);                 // one push, not two
    act(() => result.current.history.go(-1));
    expect(result.current.loc.search).toBe("");                   // one Back returns to the start
  });

  it.each([
    ["push then replace", true],
    ["replace then push", false],
  ])("pushes one entry and one Back undoes the tick when setters mix modes (%s)", (_name, pushFirst) => {
    const { result } = renderHook(() => ({
      p: useUrlState("p", 0, { replace: false }),
      r: useUrlState("r", 0),
      loc: useLocation(),
      history: useContext(UNSAFE_NavigationContext).navigator as unknown as { index: number; go: (n: number) => void },
    }), { wrapper: at("/x?keep=1") });
    act(() => {
      if (pushFirst) { result.current.p[1](1); result.current.r[1](2); }
      else { result.current.r[1](2); result.current.p[1](1); }
    });
    expect(new URLSearchParams(result.current.loc.search).toString()).toMatch(/^(keep=1&p=1&r=2|keep=1&r=2&p=1)$/);
    expect(result.current.history.index).toBe(1);                 // one push, whatever the order
    act(() => result.current.history.go(-1));
    expect(result.current.loc.search).toBe("?keep=1");            // one Back returns to the start
  });

  it("a same-value write is a no-op even when a neighbour is not canonically encoded", () => {
    const { result } = renderHook(() => ({
      a: useUrlState("a", 0, { replace: false }),
      loc: useLocation(),
      history: useContext(UNSAFE_NavigationContext).navigator as unknown as { index: number },
    }), { wrapper: at("/x?inspect=member:m_1&a=3") });
    act(() => result.current.a[1](3));
    expect(result.current.history.index).toBe(0);                 // no duplicate entry
    expect(result.current.loc.search).toBe("?inspect=member:m_1&a=3");
  });

  it("a same-value write of a non-canonically encoded param of its own is a no-op", () => {
    const { result } = renderHook(() => ({
      t: useUrlState<string[]>("t", [], { replace: false }),
      history: useContext(UNSAFE_NavigationContext).navigator as unknown as { index: number },
      loc: useLocation(),
    }), { wrapper: at("/x?t=lr,sr") });
    act(() => result.current.t[1](["lr", "sr"]));
    expect(result.current.history.index).toBe(0);
    expect(result.current.loc.search).toBe("?t=lr,sr");
  });

  it("writing the value the hook already reads (explicit default, unparseable → default) is a no-op", () => {
    const { result } = renderHook(() => ({
      a: useUrlState("a", 0, { replace: false }),
      bad: useUrlState("bad", 7, { replace: false }),
      history: useContext(UNSAFE_NavigationContext).navigator as unknown as { index: number },
      loc: useLocation(),
    }), { wrapper: at("/x?a=0&bad=abc") });
    expect(result.current.bad[0]).toBe(7);
    act(() => { result.current.a[1](0); result.current.bad[1](7); });
    expect(result.current.history.index).toBe(0);
    expect(result.current.loc.search).toBe("?a=0&bad=abc");
    act(() => result.current.bad[1](8));                          // a real change still pushes
    expect(result.current.history.index).toBe(1);
    expect(result.current.loc.search).toBe("?a=0&bad=8");
  });

  it("keeps the other params exactly as written (no re-encoding) on a real write", () => {
    const { result } = renderHook(() => ({ a: useUrlState("a", 0), loc: useLocation() }), {
      wrapper: at("/x?inspect=member:m_1&q=a%20b&t=lr,sr&a=1"),
    });
    act(() => result.current.a[1](3));
    expect(result.current.loc.search).toBe("?inspect=member:m_1&q=a%20b&t=lr,sr&a=3");
    act(() => result.current.a[1](0));                            // default → removed
    expect(result.current.loc.search).toBe("?inspect=member:m_1&q=a%20b&t=lr,sr");
  });

  it("matches the key after decoding and writes it once (duplicates collapse)", () => {
    const { result } = renderHook(() => ({ s: useUrlState("v.main", ""), loc: useLocation() }), {
      wrapper: at("/x?v%2Emain=a&z=1&v.main=b"),
    });
    expect(result.current.s[0]).toBe("a");
    act(() => result.current.s[1]("c d"));
    expect(result.current.loc.search).toBe("?v.main=c+d&z=1");
    expect(result.current.s[0]).toBe("c d");
  });

  it("a replace-mode write keeps the entry's history state", () => {
    const { result } = renderHook(() => ({
      a: useUrlState("a", 0),
      p: useUrlState("p", 0, { replace: false }),
      loc: useLocation(),
    }), {
      wrapper: ({ children }: { children: ReactNode }) => (
        <MemoryRouter initialEntries={[{ pathname: "/x", state: { from: "palette" } }]}>{children}</MemoryRouter>
      ),
    });
    act(() => result.current.a[1](1));
    expect(result.current.loc.state).toEqual({ from: "palette" });
    act(() => result.current.p[1](1));                            // a new entry starts without state
    expect(result.current.loc.state).toBeNull();
  });

  it("starts a new entry in the next tick", async () => {
    const { result } = renderHook(() => ({
      a: useUrlState("a", 0, { replace: false }),
      history: useContext(UNSAFE_NavigationContext).navigator as unknown as { index: number },
    }), { wrapper: at("/x") });
    act(() => result.current.a[1](1));
    await act(() => new Promise((r) => setTimeout(r, 0)));
    act(() => result.current.a[1](2));
    expect(result.current.history.index).toBe(2);
  });

  it("coalesces under a data router too (the shell's router)", async () => {
    let setters: { a: (v: number) => void; b: (v: number) => void } | null = null;
    function Probe() {
      const [, a] = useUrlState("a", 0, { replace: false });
      const [, b] = useUrlState("b", 0, { replace: false });
      setters = { a, b };
      return null;
    }
    const router = createMemoryRouter([{ path: "/x", element: <Probe /> }], { initialEntries: ["/x"] });
    render(<RouterProvider router={router} />);
    act(() => { setters!.a(1); setters!.b(2); });
    expect(router.state.location.search).toBe("?a=1&b=2");
    await act(() => router.navigate(-1));
    expect(router.state.location.search).toBe("");               // a single Back
  });

  /* Setters called from different macrotasks before React re-renders (two
     viewers answering their fetches, a timer after a click) must build on the
     LATEST location, not the one the component last rendered with, or the
     second write drops the first one's param. */
  it("a setter in a later macrotask keeps an earlier write React has not rendered yet (data router)", async () => {
    let setters: { a: (v: string) => void; b: (v: string) => void } | null = null;
    function Probe() {
      const [, a] = useUrlState("a", "");
      const [, b] = useUrlState("b", "");
      setters = { a, b };
      return null;
    }
    const router = createMemoryRouter([{ path: "/x", element: <Probe /> }], {
      initialEntries: ["/x?keep=1"], future: { v7_relativeSplatPath: true },
    });
    render(<RouterProvider router={router} future={{ v7_startTransition: true }} />);
    const { a, b } = setters!;
    // Both setters fire outside act(), one macrotask apart, with the render
    // of the first write still pending (a transition).
    a("1");
    await new Promise((r) => setTimeout(r, 0));
    b("2");
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
    expect(new URLSearchParams(router.state.location.search).get("a")).toBe("1");
    expect(new URLSearchParams(router.state.location.search).get("b")).toBe("2");
    expect(new URLSearchParams(router.state.location.search).get("keep")).toBe("1");
  });

  it("builds on a navigation that happened outside the component (data router)", async () => {
    let set: ((v: string) => void) | null = null;
    function Probe() {
      const [, a] = useUrlState("a", "");
      set = a;
      return null;
    }
    const router = createMemoryRouter([{ path: "/x", element: <Probe /> }], {
      initialEntries: ["/x"], future: { v7_relativeSplatPath: true },
    });
    render(<RouterProvider router={router} future={{ v7_startTransition: true }} />);
    // e.g. the inspector sync writing ?inspect= — committed in the router,
    // not yet rendered when the setter runs.
    await router.navigate("/x?inspect=member:m_1", { replace: true });
    set!("5");
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
    expect(router.state.location.search).toBe("?inspect=member:m_1&a=5");
  });

  it("builds on the history's location under a plain MemoryRouter too", async () => {
    let probe: { a: (v: string) => void; nav: { replace: (to: string) => void } } | null = null;
    function Probe() {
      const [, a] = useUrlState("a", "");
      const { navigator } = useContext(UNSAFE_NavigationContext);
      probe = { a, nav: navigator as unknown as { replace: (to: string) => void } };
      return <LocationSpy />;
    }
    let seen = "";
    function LocationSpy() { seen = useLocation().search; return null; }
    render(<MemoryRouter initialEntries={["/x"]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><Probe /></MemoryRouter>);
    probe!.nav.replace("/x?other=1");   // history moved; React has not re-rendered
    probe!.a("2");
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
    expect(seen).toBe("?other=1&a=2");
  });

  it("does not write onto another page the router already moved to", async () => {
    let set: ((v: string) => void) | null = null;
    function Probe() {
      const [, a] = useUrlState("a", "");
      set = a;
      return null;
    }
    const router = createMemoryRouter([
      { path: "/x", element: <Probe /> }, { path: "/y", element: null },
    ], { initialEntries: ["/x"], future: { v7_relativeSplatPath: true } });
    render(<RouterProvider router={router} future={{ v7_startTransition: true }} />);
    await router.navigate("/y?z=1");
    set!("stale-write");
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
    expect(router.state.location.pathname).toBe("/y");
    expect(router.state.location.search).toBe("?z=1");
  });

  it.each([
    ["replace then push", false, "?k=1&r=2&p=1"],
    ["push then replace", true, "?k=1&p=1&r=2"],
  ])("undoes a mixed-mode tick (%s) with one Back under a data router", async (_name, pushFirst, search) => {
    let setters: { r: (v: number) => void; p: (v: number) => void } | null = null;
    function Probe() {
      const [, r] = useUrlState("r", 0);
      const [, p] = useUrlState("p", 0, { replace: false });
      setters = { r, p };
      return null;
    }
    const router = createMemoryRouter([{ path: "/x", element: <Probe /> }], { initialEntries: ["/x?k=1"] });
    render(<RouterProvider router={router} />);
    act(() => {
      if (pushFirst) { setters!.p(1); setters!.r(2); }
      else { setters!.r(2); setters!.p(1); }
    });
    expect(router.state.location.search).toBe(search);
    await act(() => router.navigate(-1));
    expect(router.state.location.search).toBe("?k=1");
  });
});
