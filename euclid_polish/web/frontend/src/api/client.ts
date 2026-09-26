/* Typed fetch layer (contract C8).
 *
 * `apiGet` / `apiPost` resolve with the parsed JSON body or throw an
 * `ApiError` that keeps the HTTP status, the server's `{error}` text, its
 * optional machine `code` (e.g. "fasrc_offline", contract C4) and the raw
 * body — so pages can tell "no data yet" from "server error" from "FASRC
 * offline". Same-origin in production (Flask serves the SPA); in dev Vite
 * proxies every non-page path to Flask (vite.config.ts).
 *
 * Aborts (AbortController) are rethrown untouched so TanStack Query and
 * callers can ignore them.
 */

export type FormValue = string | number | boolean | undefined | null;
export type FormRecord = Record<string, FormValue>;

export class ApiError extends Error {
  /** HTTP status; 0 for a network failure. */
  readonly status: number;
  /** Machine-readable error code from the body (`{code}`), when present. */
  readonly code?: string;
  /** Parsed JSON body, or the raw text when the body was not JSON. */
  readonly body: unknown;
  readonly url?: string;

  constructor(init: { status: number; message: string; code?: string; body?: unknown; url?: string }) {
    super(init.message);
    this.name = "ApiError";
    this.status = init.status;
    this.code = init.code;
    this.body = init.body ?? null;
    this.url = init.url;
  }
}

/** True for the C4 gate response (503 `{code: "fasrc_offline"}`). */
export function isFasrcOffline(err: unknown): boolean {
  return err instanceof ApiError && err.status === 503 && err.code === "fasrc_offline";
}

export function isAbortError(err: unknown): boolean {
  return (err instanceof DOMException || err instanceof Error) && err.name === "AbortError";
}

async function send(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, { credentials: "same-origin", ...init });
  } catch (e) {
    if (isAbortError(e)) throw e;
    const msg = e instanceof Error ? e.message : String(e);
    throw new ApiError({ status: 0, message: `network error: ${msg}`, url });
  }
}

type Parsed = { json: unknown; text: string; isJson: boolean };

async function read(r: Response): Promise<Parsed> {
  const text = await r.text();
  if (!text) return { json: null, text, isJson: false };
  try { return { json: JSON.parse(text), text, isJson: true }; } catch { return { json: null, text, isJson: false }; }
}

function httpError(r: Response, p: Parsed, url: string): ApiError {
  const body = p.isJson ? p.json : p.text || null;
  const obj = (p.isJson && p.json && typeof p.json === "object") ? p.json as Record<string, unknown> : null;
  const serverMsg = obj && typeof obj.error === "string" && obj.error ? obj.error : null;
  const code = obj && typeof obj.code === "string" ? obj.code : undefined;
  const fallback = `HTTP ${r.status}${r.statusText ? ` ${r.statusText}` : ""}`;
  return new ApiError({ status: r.status, message: serverMsg ?? fallback, code, body, url });
}

/** GET JSON. Throws `ApiError` on HTTP/network errors and non-JSON bodies. */
export async function apiGet<T = unknown>(url: string, opts: { signal?: AbortSignal } = {}): Promise<T> {
  const r = await send(url, { headers: { Accept: "application/json" }, signal: opts.signal });
  const p = await read(r);
  if (!r.ok) throw httpError(r, p, url);
  if (!p.isJson) {
    throw new ApiError({
      status: r.status, message: `response from ${url} is not valid JSON`, body: p.text || null, url,
    });
  }
  return p.json as T;
}

/** Build the form body every mutation/job endpoint expects (null/undefined
 *  values are skipped; everything else is stringified). */
export function toFormData(data: FormRecord | FormData): FormData {
  if (data instanceof FormData) return data;
  const body = new FormData();
  for (const [k, v] of Object.entries(data)) if (v != null) body.set(k, String(v));
  return body;
}

/**
 * POST form-encoded (default) or JSON (`{json: true}`) and return the parsed
 * body (`{}` when empty). Throws `ApiError` on HTTP/network errors. A 200
 * response carrying `{error}` is returned as-is (not thrown), like the old
 * `postForm` — callers such as `useJob` inspect it.
 */
export async function apiPost<T = unknown>(
  url: string,
  data: FormRecord | FormData | object = {},
  opts: { json?: boolean; signal?: AbortSignal } = {},
): Promise<T> {
  const init: RequestInit = { method: "POST", signal: opts.signal, headers: { Accept: "application/json" } };
  if (opts.json) {
    init.headers = { Accept: "application/json", "Content-Type": "application/json" };
    init.body = JSON.stringify(data);
  } else {
    init.body = toFormData(data as FormRecord | FormData);
  }
  const r = await send(url, init);
  const p = await read(r);
  if (!r.ok) throw httpError(r, p, url);
  return (p.isJson ? (p.json ?? {}) : {}) as T;
}
