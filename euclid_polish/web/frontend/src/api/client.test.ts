import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiGet, apiPost, isFasrcOffline } from "./client";
import { getJSON, postForm } from "../api";

type Handler = (url: string, init: RequestInit) => Response | Promise<Response>;

function mockFetch(handler: Handler) {
  const fn = vi.fn((input: RequestInfo | URL, init: RequestInit = {}) =>
    Promise.resolve(handler(String(input), init)));
  vi.stubGlobal("fetch", fn);
  return fn;
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

afterEach(() => { vi.unstubAllGlobals(); });

/** The rejection of `p` (typed: these tests only ever reject with ApiError). */
const caught = (p: Promise<unknown>) => p.then(
  () => { throw new Error("expected a rejection"); },
  (e: unknown) => e as ApiError,
);

describe("apiGet", () => {
  it("returns parsed JSON and asks for JSON", async () => {
    const fetch = mockFetch(() => json({ ok: true, n: 3 }));
    await expect(apiGet<{ n: number }>("/api/x")).resolves.toEqual({ ok: true, n: 3 });
    const init = fetch.mock.calls[0][1] as RequestInit;
    expect(new Headers(init.headers).get("Accept")).toBe("application/json");
    expect(init.credentials).toBe("same-origin");
  });

  it("passes the abort signal through", async () => {
    const fetch = mockFetch(() => json({}));
    const ctl = new AbortController();
    await apiGet("/api/x", { signal: ctl.signal });
    expect((fetch.mock.calls[0][1] as RequestInit).signal).toBe(ctl.signal);
  });

  it("throws ApiError carrying the server's error text, code and body", async () => {
    mockFetch(() => json({ ok: false, error: "no such member", code: "missing" }, 404));
    const err = await caught(apiGet("/api/x"));
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toBeInstanceOf(Error);
    expect(err.status).toBe(404);
    expect(err.message).toBe("no such member");
    expect(err.code).toBe("missing");
    expect(err.body).toEqual({ ok: false, error: "no such member", code: "missing" });
    expect(err.url).toBe("/api/x");
  });

  it("recognises the FASRC-offline gate (C4)", async () => {
    mockFetch(() => json({ ok: false, error: "FASRC not connected", code: "fasrc_offline" }, 503));
    const err = await caught(apiGet("/api/fasrc/steps/status"));
    expect(err.status).toBe(503);
    expect(isFasrcOffline(err)).toBe(true);
    expect(isFasrcOffline(new ApiError({ status: 503, message: "busy" }))).toBe(false);
    expect(isFasrcOffline(new Error("x"))).toBe(false);
  });

  it("falls back to HTTP status text for non-JSON error bodies", async () => {
    mockFetch(() => new Response("<html>boom</html>", { status: 500, statusText: "INTERNAL SERVER ERROR" }));
    const err = await caught(apiGet("/api/x"));
    expect(err.status).toBe(500);
    expect(err.message).toBe("HTTP 500 INTERNAL SERVER ERROR");
    expect(err.body).toBe("<html>boom</html>");
  });

  it("wraps network failures as status 0", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    const err = await caught(apiGet("/api/x"));
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
    expect(err.message).toMatch(/Failed to fetch/);
  });

  it("rethrows aborts untouched", async () => {
    const abort = new DOMException("aborted", "AbortError");
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(abort)));
    await expect(apiGet("/api/x")).rejects.toBe(abort);
  });

  it("rejects a 200 whose body is not JSON", async () => {
    mockFetch(() => new Response("not json", { status: 200 }));
    const err = await caught(apiGet("/api/x"));
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(200);
    expect(err.message).toMatch(/not valid JSON/);
  });
});

describe("apiPost", () => {
  it("form-encodes a record, skipping null/undefined", async () => {
    const fetch = mockFetch(() => json({ job_id: "j1" }));
    await expect(apiPost("/api/run", { a: 1, b: "x", c: true, d: null, e: undefined })).resolves.toEqual({ job_id: "j1" });
    const init = fetch.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("POST");
    const body = init.body as FormData;
    expect(body).toBeInstanceOf(FormData);
    expect([...body.entries()]).toEqual([["a", "1"], ["b", "x"], ["c", "true"]]);
  });

  it("passes FormData through unchanged", async () => {
    const fetch = mockFetch(() => json({}));
    const fd = new FormData();
    fd.set("k", "v");
    await apiPost("/api/run", fd);
    expect((fetch.mock.calls[0][1] as RequestInit).body).toBe(fd);
  });

  it("sends JSON when asked", async () => {
    const fetch = mockFetch(() => json({ ok: true }));
    await apiPost("/api/experiments", { tiles: ["nexus/1"], models: ["production"] }, { json: true });
    const init = fetch.mock.calls[0][1] as RequestInit;
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
    expect(JSON.parse(String(init.body))).toEqual({ tiles: ["nexus/1"], models: ["production"] });
  });

  it("throws ApiError on HTTP errors and returns {} for empty success bodies", async () => {
    mockFetch(() => json({ error: "bad knee" }, 400));
    await expect(apiPost("/api/x", {})).rejects.toMatchObject({ status: 400, message: "bad knee" });
    mockFetch(() => new Response("", { status: 200 }));
    await expect(apiPost("/api/x")).resolves.toEqual({});
  });

  it("returns a 200 {error} body as-is (compat with the old postForm)", async () => {
    mockFetch(() => json({ error: "soft failure" }));
    await expect(apiPost("/api/x")).resolves.toEqual({ error: "soft failure" });
  });
});

describe("compat wrappers (src/api.ts)", () => {
  it("getJSON resolves null on any failure", async () => {
    mockFetch(() => json({ error: "gone" }, 404));
    await expect(getJSON("/api/x")).resolves.toBeNull();
    mockFetch(() => json({ v: 1 }));
    await expect(getJSON("/api/x")).resolves.toEqual({ v: 1 });
  });

  it("postForm throws an Error with the server message", async () => {
    mockFetch(() => json({ error: "nope" }, 409));
    const err = await caught(postForm("/api/x", { a: 1 }));
    expect(err).toBeInstanceOf(Error);
    expect(err.message).toBe("nope");
  });
});
