/* Thin fetch layer. Same-origin in production (Flask serves the SPA); in dev
   Vite proxies these prefixes to Flask. Returns null on 404 so callers can show
   an empty state without try/catch noise. */
export async function getJSON<T = any>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url, { headers: { Accept: "application/json" } });
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}

export function qs(params: Record<string, string | number | undefined>): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v != null) p.set(k, String(v));
  const s = p.toString();
  return s ? `?${s}` : "";
}
