/* try/catch-guarded localStorage for persisted stores. Storage can throw
   (private mode, quota, sandboxed iframes, disabled cookies); a failure must
   never break the UI — the store simply stops persisting. */
import { createJSONStorage, type StateStorage } from "zustand/middleware";

function ls(): Storage | null {
  try { return typeof window !== "undefined" ? window.localStorage ?? null : null; } catch { return null; }
}

export const safeLocalStorage: StateStorage = {
  getItem(name) {
    try { return ls()?.getItem(name) ?? null; } catch { return null; }
  },
  setItem(name, value) {
    try { ls()?.setItem(name, value); } catch { /* quota / disabled: ignore */ }
  },
  removeItem(name) {
    try { ls()?.removeItem(name); } catch { /* ignore */ }
  },
};

/** JSON storage for zustand's `persist` middleware. */
export const safeJSONStorage = createJSONStorage(() => safeLocalStorage);

/** Read a raw string key (null when missing or storage is unavailable). */
export function readStorage(key: string): string | null {
  return safeLocalStorage.getItem(key) as string | null;
}

/** Write a raw string key (silently ignored when storage is unavailable). */
export function writeStorage(key: string, value: string): void {
  void safeLocalStorage.setItem(key, value);
}
