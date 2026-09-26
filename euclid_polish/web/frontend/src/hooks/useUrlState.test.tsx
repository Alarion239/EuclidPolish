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
