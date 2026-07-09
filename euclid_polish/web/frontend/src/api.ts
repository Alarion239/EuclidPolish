/* Thin, typed fetch layer. Same-origin in production (Flask serves the SPA); in
   dev Vite proxies these prefixes to Flask. Returns null on 404 so callers can
   show an empty state without try/catch noise. */
export async function getJSON<T = unknown>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url, { headers: { Accept: "application/json" } });
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}

export type FormValue = string | number | boolean | undefined | null;

/** POST a form-encoded body (the shape every mutation/job endpoint expects) and
 *  return the parsed JSON. Throws on transport/HTTP error so callers can show a
 *  message; a JSON body of `{error}` is returned as-is (not thrown). */
export async function postForm<T = unknown>(
  url: string,
  data: Record<string, FormValue> | FormData = {},
): Promise<T> {
  let body: FormData;
  if (data instanceof FormData) {
    body = data;
  } else {
    body = new FormData();
    for (const [k, v] of Object.entries(data)) if (v != null) body.set(k, String(v));
  }
  const r = await fetch(url, { method: "POST", body, credentials: "same-origin" });
  const text = await r.text();
  let json: unknown = null;
  try { json = text ? JSON.parse(text) : null; } catch { /* non-JSON */ }
  if (!r.ok) {
    const msg = (json as { error?: string })?.error ?? `HTTP ${r.status}`;
    throw new Error(msg);
  }
  return (json ?? {}) as T;
}

/** Build a `?a=1&b=2` query string, skipping null/undefined values. */
export function qs(params: Record<string, FormValue>): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v != null) p.set(k, String(v));
  const s = p.toString();
  return s ? `?${s}` : "";
}
