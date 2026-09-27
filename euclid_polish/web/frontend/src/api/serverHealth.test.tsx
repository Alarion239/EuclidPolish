import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "./client";
import {
  ServerHealthTracker, isServerDownError, queryClient, serverHealth, setResourceData, useResource,
  useServerHealth,
} from "./query";

const net = () => new ApiError({ status: 0, message: "network error: Failed to fetch" });
const http = (status: number, code?: string) => new ApiError({ status, message: `HTTP ${status}`, code });

describe("isServerDownError", () => {
  it("counts no response, internal and gateway errors (Vite's dev proxy answers 500 when Flask is down)", () => {
    expect(isServerDownError(net())).toBe(true);
    for (const s of [500, 502, 504]) expect(isServerDownError(http(s))).toBe(true);
  });

  it("does not count a 5xx carrying the app's JSON envelope (Flask answered: e.g. the 502 of a failed FASRC call)", () => {
    const json = (status: number) => new ApiError({
      status, message: "remote listing failed", body: { ok: false, error: "remote listing failed" },
    });
    for (const s of [500, 502, 504]) expect(isServerDownError(json(s))).toBe(false);
    // A proxy's text body is not the app's.
    expect(isServerDownError(new ApiError({ status: 502, message: "HTTP 502", body: "Bad Gateway" }))).toBe(true);
  });

  it("does not count answers the server gave on purpose", () => {
    for (const s of [400, 403, 404, 409, 501]) expect(isServerDownError(http(s))).toBe(false);
    expect(isServerDownError(http(503, "fasrc_offline"))).toBe(false);   // the FASRC gate
    expect(isServerDownError(http(503))).toBe(false);
  });

  it("ignores aborts and errors that are not API errors", () => {
    expect(isServerDownError(new DOMException("aborted", "AbortError"))).toBe(false);
    expect(isServerDownError(new TypeError("x is undefined"))).toBe(false);
    expect(isServerDownError(null)).toBe(false);
  });
});

describe("ServerHealthTracker", () => {
  let t = 1_000;
  const make = () => new ServerHealthTracker({ now: () => t });
  beforeEach(() => { t = 1_000; });

  it("starts up, with no answer yet", () => {
    expect(make().getState()).toEqual({ down: false, lastOkAt: null, downSince: null, lastError: null });
  });

  it("a request that got no response at all means the server is down", () => {
    const h = make();
    h.answered();
    t = 5_000;
    h.failed("GET /api/version", net());
    expect(h.getState()).toMatchObject({ down: true, lastOkAt: 1_000, downSince: 5_000 });
    expect(h.getState().lastError).toMatch(/Failed to fetch/);
  });

  it("one route answering 500 is a route bug, not a dead server; two different ones are", () => {
    const h = make();
    h.failed("GET /api/a", http(500));
    h.failed("GET /api/a", http(500));
    expect(h.getState().down).toBe(false);
    t = 2_000;
    h.failed("GET /api/b", http(502));
    expect(h.getState()).toMatchObject({ down: true, downSince: 2_000 });
  });

  it("any answer clears the outage and the failure count", () => {
    const h = make();
    h.failed("a", http(500));
    h.answered();
    h.failed("b", http(500));
    expect(h.getState().down).toBe(false);          // a and b were not failing together
    h.failed("c", net());
    expect(h.getState().down).toBe(true);
    t = 9_000;
    h.answered();
    expect(h.getState()).toEqual({ down: false, lastOkAt: 9_000, downSince: null, lastError: null });
  });

  it("keeps the first moment of an outage while it lasts", () => {
    const h = make();
    h.failed("a", net());
    t = 4_000;
    h.failed("b", net());
    expect(h.getState().downSince).toBe(1_000);
  });

  it("notifies subscribers only when the state changes", () => {
    const h = make();
    const fn = vi.fn();
    const off = h.subscribe(fn);
    h.failed("a", http(500));
    expect(fn).not.toHaveBeenCalled();
    h.failed("b", http(500));
    expect(fn).toHaveBeenCalledTimes(1);
    h.failed("c", http(500));
    expect(fn).toHaveBeenCalledTimes(1);            // still down, same downSince
    h.answered();
    expect(fn).toHaveBeenCalledTimes(2);
    off();
    h.failed("d", net());
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("reset forgets everything", () => {
    const h = make();
    h.answered();
    h.failed("a", net());
    h.reset();
    expect(h.getState()).toEqual({ down: false, lastOkAt: null, downSince: null, lastError: null });
  });
});

/* ── wired to the shared query cache ─────────────────────────────────────── */

type Mode = "up" | "down";
let mode: Mode;
let calls: string[];
const DEFAULTS = queryClient.getDefaultOptions();

beforeEach(() => {
  mode = "up";
  calls = [];
  queryClient.clear();
  serverHealth.reset();
  // Retries stay on (the real policy) but without the 0.5–4 s back-off.
  queryClient.setDefaultOptions({ ...DEFAULTS, queries: { ...DEFAULTS.queries, retryDelay: 1 } });
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    calls.push(url);
    if (mode === "down") throw new TypeError("Failed to fetch");
    return new Response(JSON.stringify({ url, n: calls.length }), { status: 200 });
  }));
});
afterEach(() => {
  queryClient.clear();
  queryClient.setDefaultOptions(DEFAULTS);
  serverHealth.reset();
  vi.unstubAllGlobals();
});

describe("useServerHealth (the query cache feeds it)", () => {
  it("goes down when refreshes stop reaching the server, keeps the stale data, and recovers", async () => {
    const a = renderHook(() => useResource<{ url: string }>("/api/a"));
    const b = renderHook(() => useResource<{ url: string }>("/api/b"));
    const h = renderHook(() => useServerHealth());
    await waitFor(() => expect(a.result.current.data?.url).toBe("/api/a"));
    await waitFor(() => expect(b.result.current.data?.url).toBe("/api/b"));
    expect(h.result.current.down).toBe(false);
    expect(h.result.current.lastOkAt).not.toBeNull();

    mode = "down";
    act(() => { a.result.current.reload(); b.result.current.reload(); });
    await waitFor(() => expect(h.result.current.down).toBe(true));
    // the last data stays on screen, marked by its failed refresh
    expect(a.result.current.data?.url).toBe("/api/a");
    expect(a.result.current.staleError?.status).toBe(0);

    mode = "up";
    const before = calls.length;
    act(() => { a.result.current.reload(); });
    await waitFor(() => expect(h.result.current.down).toBe(false));
    // recovery refetches every resource whose refresh had failed (b too)
    await waitFor(() => expect(calls.slice(before)).toContain("/api/b"));
    await waitFor(() => expect(b.result.current.staleError).toBeNull());
  });

  it("an optimistic cache write is not an answer from the server", async () => {
    const a = renderHook(() => useResource("/api/a"));
    await waitFor(() => expect(a.result.current.data).not.toBeNull());
    mode = "down";
    act(() => { a.result.current.reload(); });
    await waitFor(() => expect(serverHealth.getState().down).toBe(true));
    act(() => { setResourceData("/api/a", { optimistic: true }); });
    expect(serverHealth.getState().down).toBe(true);
  });

  it("an HTTP error answer (404) counts as the server answering", async () => {
    serverHealth.failed("GET /api/x", net());
    expect(serverHealth.getState().down).toBe(true);
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ error: "nope" }), { status: 404 })));
    const r = renderHook(() => useResource("/api/missing"));
    await waitFor(() => expect(r.result.current.error?.status).toBe(404));
    expect(serverHealth.getState().down).toBe(false);
  });

  it("two different routes failing with Flask's JSON 502 (FASRC unreachable) do not mark the local server down", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(
      JSON.stringify({ ok: false, error: "remote listing failed: ssh timeout" }), { status: 502 })));
    const a = renderHook(() => useResource("/api/fasrc?path=a"));
    const b = renderHook(() => useResource("/fasrc/file/inspect?path=b"));
    await waitFor(() => expect(a.result.current.error?.status).toBe(502));
    await waitFor(() => expect(b.result.current.error?.status).toBe(502));
    expect(serverHealth.getState().down).toBe(false);
  });

  it("two different routes getting the dev proxy's empty 500 (Flask gone) do", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 500 })));
    renderHook(() => useResource("/api/a"));
    renderHook(() => useResource("/api/b"));
    await waitFor(() => expect(serverHealth.getState().down).toBe(true));
  });

  it("clearing the whole cache forgets the outage (nothing stale is left)", () => {
    serverHealth.failed("GET /api/x", net());
    queryClient.setQueryData(["GET", "/api/x"], 1);
    expect(serverHealth.getState().down).toBe(true);
    act(() => queryClient.clear());
    expect(serverHealth.getState().down).toBe(false);
  });
});
