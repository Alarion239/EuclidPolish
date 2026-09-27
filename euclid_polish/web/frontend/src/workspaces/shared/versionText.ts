/* The server / console version state in plain words, shared by Home › Server
 * and Settings › About (versionText.test.ts). `behind` (GET /api/version) is
 * set when a backend .py file the server loaded changed on disk since it
 * started; a new console build only needs a page reload. */
import type { Tone } from "../../ui";

export type CodeState = { tone: Tone; badge: string; title: string | null };

export function serverCodeText(v: { behind: boolean }, consoleUpdated: boolean): CodeState {
  if (v.behind) return { tone: "warn", badge: "code changed", title: "Backend code changed — restart the server to load it" };
  if (consoleUpdated) return { tone: "warn", badge: "new build", title: "The console build changed — reload" };
  return { tone: "good", badge: "current code", title: null };
}
