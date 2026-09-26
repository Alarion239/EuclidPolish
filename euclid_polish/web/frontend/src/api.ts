/* Compatibility layer over `api/client.ts` for pages written before the
   foundation rework. New code uses `apiGet` / `apiPost` / `ApiError` from
   "./api/client" (or `useResource` from "./api/query"). */
import { apiGet, apiPost, type FormRecord } from "./api/client";

export { ApiError, apiGet, apiPost, isFasrcOffline, toFormData } from "./api/client";
export type { FormRecord, FormValue } from "./api/client";

/** GET JSON, or null on ANY failure (404, 5xx, network, bad JSON). Prefer
 *  `apiGet`, which keeps the status and the server's error text. */
export async function getJSON<T = unknown>(url: string): Promise<T | null> {
  try {
    return await apiGet<T>(url);
  } catch {
    return null;
  }
}

/** POST a form-encoded body and return the parsed JSON (`{}` when empty).
 *  Throws `ApiError` (an `Error` with the server's message) on HTTP errors; a
 *  200 `{error}` body is returned as-is. */
export function postForm<T = unknown>(url: string, data: FormRecord | FormData = {}): Promise<T> {
  return apiPost<T>(url, data);
}
