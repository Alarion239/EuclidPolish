import { act, renderHook, waitFor } from "@testing-library/react";
import { focusManager, onlineManager } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "./client";
import { invalidate, queryClient, useResource } from "./query";
import { invalidateCache, useResource as compatUseResource } from "../hooks";

type Reply = { status?: number; body: unknown; delay?: number };

/** fetch mock: per-URL reply queue (the last reply repeats). */
function mockServer(routes: Record<string, Reply[]>) {
  const calls: string[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    calls.push(url);
    const queue = routes[url];
    if (!queue) return new Response(JSON.stringify({ error: "not found" }), { status: 404 });
    const reply = queue.length > 1 ? queue.shift()! : queue[0];
    if (reply.delay) await new Promise((r) => setTimeout(r, reply.delay));
    return new Response(JSON.stringify(reply.body), { status: reply.status ?? 200 });
  });
  vi.stubGlobal("fetch", fn);
  return { calls, count: (url: string) => calls.filter((c) => c === url).length };
}

beforeEach(() => { queryClient.clear(); focusManager.setFocused(undefined); });
afterEach(() => { queryClient.clear(); vi.unstubAllGlobals(); focusManager.setFocused(undefined); });

describe("useResource", () => {
  it("loads JSON with loading → data", async () => {
    mockServer({ "/api/a": [{ body: { v: 1 } }] });
    const { result } = renderHook(() => useResource<{ v: number }>("/api/a"));
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    await waitFor(() => expect(result.current.data).toEqual({ v: 1 }));
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("does nothing for a null URL", () => {
    const srv = mockServer({});
    const { result } = renderHook(() => useResource(null));
    expect(result.current).toMatchObject({ data: null, loading: false, error: null });
    expect(srv.calls).toEqual([]);
  });

  it("treats an empty-string URL as idle, like null (compat with the old hook)", async () => {
    const srv = mockServer({});
    const { result } = renderHook(() => useResource(""));
    expect(result.current).toMatchObject({ data: null, loading: false, fetching: false, error: null });
    await new Promise((r) => setTimeout(r, 20));
    expect(srv.calls).toEqual([]);
    expect(result.current.error).toBeNull();
  });

  it("an idle resource never shows another idle resource's cached state", async () => {
    mockServer({});
    queryClient.setQueryData(["GET", ""], { leaked: true });
    const a = renderHook(() => useResource(null));
    const b = renderHook(() => useResource(""));
    expect(a.result.current.data).toBeNull();
    expect(b.result.current.data).toBeNull();
  });

  it("dedupes concurrent requests and shares the result between subscribers", async () => {
    const srv = mockServer({ "/api/steps": [{ body: { steps: [1] }, delay: 20 }] });
    const a = renderHook(() => useResource("/api/steps"));
    const b = renderHook(() => useResource("/api/steps"));
    await waitFor(() => expect(b.result.current.data).toEqual({ steps: [1] }));
    expect(a.result.current.data).toEqual({ steps: [1] });
    expect(srv.count("/api/steps")).toBe(1);
  });

  it("serves a fresh cached copy instantly without refetching", async () => {
    const srv = mockServer({ "/api/a": [{ body: { v: 1 } }] });
    const first = renderHook(() => useResource("/api/a"));
    await waitFor(() => expect(first.result.current.data).toEqual({ v: 1 }));
    first.unmount();
    const again = renderHook(() => useResource("/api/a"));
    expect(again.result.current.data).toEqual({ v: 1 });
    expect(again.result.current.loading).toBe(false);
    await new Promise((r) => setTimeout(r, 10));
    expect(srv.count("/api/a")).toBe(1);
  });

  it("revalidates in the background once the ttl has passed", async () => {
    const srv = mockServer({ "/api/a": [{ body: { v: 1 } }, { body: { v: 2 } }] });
    const first = renderHook(() => useResource("/api/a", [], { ttl: 5 }));
    await waitFor(() => expect(first.result.current.data).toEqual({ v: 1 }));
    first.unmount();
    await new Promise((r) => setTimeout(r, 15));
    const again = renderHook(() => useResource("/api/a", [], { ttl: 5 }));
    expect(again.result.current.data).toEqual({ v: 1 });          // stale copy shown at once
    await waitFor(() => expect(again.result.current.data).toEqual({ v: 2 }));
    expect(srv.count("/api/a")).toBe(2);
  });

  it("refetches when deps change even though the URL is cached and fresh", async () => {
    const srv = mockServer({ "/api/history": [{ body: { n: 1 } }, { body: { n: 2 } }] });
    const { result, rerender } = renderHook(({ k }) => useResource("/api/history", [k]), {
      initialProps: { k: 0 },
    });
    await waitFor(() => expect(result.current.data).toEqual({ n: 1 }));
    rerender({ k: 1 });
    expect(result.current.data).toEqual({ n: 1 });                // kept while refetching
    await waitFor(() => expect(result.current.data).toEqual({ n: 2 }));
    expect(srv.count("/api/history")).toBe(2);
  });

  it.each([
    ["an inline object", () => [{ mode: "x", tiles: [1, 2] }], () => [{ mode: "y", tiles: [1, 2] }]],
    ["an inline array", () => [["a", "b"]], () => [["a", "c"]]],
    ["a fresh fallback array", () => [[] as string[], null], () => [["m_1"], null]],
    ["a Date + a Map + a Set", () => [new Date(5), new Map([["k", [1]]]), new Set(["s"])],
      () => [new Date(5), new Map([["k", [2]]]), new Set(["s"])]],
  ])("compares deps by value: %s rebuilt every render never loops", async (_name, same, changed) => {
    const srv = mockServer({ "/api/obj": [{ body: { n: 1 } }, { body: { n: 2 } }, { body: { n: 3 } }] });
    let useChanged = false;
    const { result, rerender } = renderHook(() => useResource<{ n: number }>("/api/obj", useChanged ? changed() : same()));
    await waitFor(() => expect(result.current.data).toEqual({ n: 1 }));
    rerender();                                                   // equal value, new identity
    await new Promise((r) => setTimeout(r, 40));
    expect(srv.count("/api/obj")).toBe(1);                        // exactly one fetch after data lands
    useChanged = true;
    rerender();                                                   // a real value change
    await waitFor(() => expect(result.current.data).toEqual({ n: 2 }));
    rerender();
    await new Promise((r) => setTimeout(r, 40));
    expect(srv.count("/api/obj")).toBe(2);                        // exactly one more
  });

  it("ignores function deps (an inline callback would otherwise refetch forever)", async () => {
    const srv = mockServer({ "/api/fn": [{ body: 1 }] });
    const { result, rerender } = renderHook(() => useResource<number>("/api/fn", [() => 1, "k"]));
    await waitFor(() => expect(result.current.data).toBe(1));
    rerender();
    await new Promise((r) => setTimeout(r, 40));
    expect(srv.count("/api/fn")).toBe(1);
  });

  it("still refetches on a primitive dep change (NaN equals itself; null → undefined is a change)", async () => {
    const srv = mockServer({ "/api/p": [{ body: 1 }, { body: 2 }, { body: 3 }] });
    const { result, rerender } = renderHook(({ k }) => useResource<number>("/api/p", [k]), {
      initialProps: { k: Number.NaN as number | null | undefined },
    });
    await waitFor(() => expect(result.current.data).toBe(1));
    rerender({ k: Number.NaN });
    await new Promise((r) => setTimeout(r, 30));
    expect(srv.count("/api/p")).toBe(1);
    rerender({ k: null });
    await waitFor(() => expect(result.current.data).toBe(2));
    rerender({ k: undefined });                                   // null → undefined is a change
    await waitFor(() => expect(result.current.data).toBe(3));
    expect(srv.count("/api/p")).toBe(3);
  });

  it("does not double-fetch when deps change together with the URL", async () => {
    const srv = mockServer({ "/api/x?m=a": [{ body: "a" }], "/api/x?m=b": [{ body: "b" }] });
    const { result, rerender } = renderHook(({ m }) => useResource(`/api/x?m=${m}`, [m]), {
      initialProps: { m: "a" },
    });
    await waitFor(() => expect(result.current.data).toBe("a"));
    rerender({ m: "b" });
    expect(result.current.data).toBeNull();                       // never shows the other URL's data
    await waitFor(() => expect(result.current.data).toBe("b"));
    rerender({ m: "a" });
    expect(result.current.data).toBe("a");                        // cached + fresh
    await new Promise((r) => setTimeout(r, 10));
    expect(srv.count("/api/x?m=a")).toBe(1);
    expect(srv.count("/api/x?m=b")).toBe(1);
  });

  it("reload() forces a refetch that every subscriber sees", async () => {
    const srv = mockServer({ "/api/a": [{ body: 1 }, { body: 2 }] });
    const a = renderHook(() => useResource<number>("/api/a"));
    const b = renderHook(() => useResource<number>("/api/a"));
    await waitFor(() => expect(a.result.current.data).toBe(1));
    act(() => a.result.current.reload());
    await waitFor(() => expect(b.result.current.data).toBe(2));
    expect(srv.count("/api/a")).toBe(2);
  });

  it("exposes an ApiError (truthy) when there is no data", async () => {
    mockServer({ "/api/bad": [{ status: 404, body: { error: "no evals yet" } }] });
    const { result } = renderHook(() => useResource("/api/bad"));
    await waitFor(() => expect(result.current.error).toBeInstanceOf(ApiError));
    expect(result.current.error?.status).toBe(404);
    expect(result.current.error?.message).toBe("no evals yet");
    expect(result.current.loading).toBe(false);
    expect(result.current.data).toBeNull();
  });

  it.each([
    [400, { error: "bad request" }],
    [404, { error: "no evals yet" }],
    [501, { error: "not implemented" }],
    [503, { ok: false, error: "FASRC not connected", code: "fasrc_offline" }],
  ])("never retries a %i (fetched exactly once)", async (status, body) => {
    const srv = mockServer({ "/api/once": [{ status, body }] });
    const { result } = renderHook(() => useResource("/api/once"));
    // A retried request only surfaces its error after every attempt failed,
    // so a single fetch at the moment the error shows proves "no retry".
    await waitFor(() => expect(result.current.error?.status).toBe(status));
    expect(srv.count("/api/once")).toBe(1);
    await new Promise((r) => setTimeout(r, 30));
    expect(srv.count("/api/once")).toBe(1);
    if (status === 503) expect(result.current.error?.code).toBe("fasrc_offline");
  });

  it("aborts the request once nothing is subscribed any more", async () => {
    const signals: AbortSignal[] = [];
    vi.stubGlobal("fetch", vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
      const signal = init?.signal;
      if (signal) signals.push(signal);
      return new Promise<Response>((_resolve, reject) => {
        signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      });
    }));
    const a = renderHook(() => useResource("/api/slow"));
    const b = renderHook(() => useResource("/api/slow"));
    await waitFor(() => expect(signals).toHaveLength(1));         // deduped: one request
    a.unmount();
    expect(signals[0].aborted).toBe(false);                       // b still wants it
    b.unmount();
    expect(signals[0].aborted).toBe(true);
  });

  it("keeps stale data (error null) when a revalidation fails", async () => {
    mockServer({ "/api/a": [{ body: { v: 1 } }, { status: 404, body: { error: "gone" } }] });
    const { result } = renderHook(() => useResource("/api/a"));
    await waitFor(() => expect(result.current.data).toEqual({ v: 1 }));
    act(() => result.current.reload());
    await waitFor(() => expect(result.current.staleError?.status).toBe(404));
    expect(result.current.data).toEqual({ v: 1 });
    expect(result.current.error).toBeNull();
  });

  it("polls while visible and pauses when the tab is hidden", async () => {
    const srv = mockServer({ "/api/poll": [{ body: 1 }] });
    const { result, unmount } = renderHook(() => useResource("/api/poll", [], { poll: 20 }));
    await waitFor(() => expect(result.current.data).toBe(1));
    await waitFor(() => expect(srv.count("/api/poll")).toBeGreaterThanOrEqual(3));
    act(() => focusManager.setFocused(false));
    await new Promise((r) => setTimeout(r, 30));
    const paused = srv.count("/api/poll");
    await new Promise((r) => setTimeout(r, 80));
    expect(srv.count("/api/poll")).toBe(paused);
    act(() => focusManager.setFocused(true));
    await waitFor(() => expect(srv.count("/api/poll")).toBeGreaterThan(paused));
    unmount();
  });
});

describe("offline-first (the browser reports offline; Flask on localhost still answers)", () => {
  let unmountClient: (() => void) | null = null;
  beforeEach(() => {
    queryClient.mount();                                          // as QueryClientProvider does
    unmountClient = () => queryClient.unmount();
    act(() => { window.dispatchEvent(new Event("offline")); });
  });
  afterEach(() => {
    act(() => { window.dispatchEvent(new Event("online")); });
    onlineManager.setOnline(true);
    unmountClient?.();
    unmountClient = null;
  });

  it("still fetches a resource after an `offline` event", async () => {
    expect(onlineManager.isOnline()).toBe(false);                 // the event did take effect
    const srv = mockServer({ "/api/noise": [{ body: { ok: true } }] });
    const { result } = renderHook(() => useResource<{ ok: boolean }>("/api/noise"));
    await waitFor(() => expect(result.current.data).toEqual({ ok: true }));
    expect(result.current.loading).toBe(false);
    expect(srv.count("/api/noise")).toBe(1);
  });

  it("keeps polling and retrying while offline", async () => {
    const srv = mockServer({ "/api/poll": [{ status: 502, body: { error: "boom" } }, { body: 1 }] });
    const { result, unmount } = renderHook(() => useResource<number>("/api/poll", [], { poll: 20 }));
    await waitFor(() => expect(result.current.data).toBe(1));     // the 502 was retried
    await waitFor(() => expect(srv.count("/api/poll")).toBeGreaterThanOrEqual(4));
    unmount();
  });
});

describe("invalidation", () => {
  it("invalidate(prefix) refetches only matching resources", async () => {
    const srv = mockServer({
      "/ensemble/status.json": [{ body: 1 }, { body: 2 }],
      "/api/other": [{ body: "x" }],
    });
    const ens = renderHook(() => useResource("/ensemble/status.json"));
    const other = renderHook(() => useResource("/api/other"));
    await waitFor(() => expect(ens.result.current.data).toBe(1));
    await waitFor(() => expect(other.result.current.data).toBe("x"));
    await act(() => invalidate("/ensemble/"));
    await waitFor(() => expect(ens.result.current.data).toBe(2));
    expect(srv.count("/api/other")).toBe(1);
  });

  it("the hooks.ts compat exports keep working", async () => {
    const srv = mockServer({ "/api/c?x=1": [{ body: 1 }, { body: 2 }] });
    const { result } = renderHook(() => compatUseResource<number>("/api/c?x=1"));
    await waitFor(() => expect(result.current.data).toBe(1));
    await act(() => invalidateCache("c?x"));
    await waitFor(() => expect(result.current.data).toBe(2));
    expect(srv.count("/api/c?x=1")).toBe(2);
  });
});
