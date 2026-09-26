/* Client-side file downloads (CSV/PNG exports) and clipboard copy — the one
   place that touches object URLs and navigator.clipboard. */

/** Filesystem-safe base name: "Knee PSNR / VIS" → "knee-psnr-vis". */
export function safeFileName(name: string, fallback = "export"): string {
  const s = name.trim().toLowerCase().replace(/[^\w.-]+/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "");
  return s || fallback;
}

/** Save a Blob as `name` via a temporary object-URL link. */
export function downloadBlob(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.rel = "noopener";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    // Revoke on the next tick: some browsers start the download asynchronously.
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

export function downloadText(name: string, text: string, mime = "text/plain;charset=utf-8"): void {
  downloadBlob(name, new Blob([text], { type: mime }));
}

/** Copy text; resolves false when neither the async clipboard nor the
 *  execCommand fallback succeeded (e.g. an insecure context). */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch { /* fall through to the legacy path */ }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = typeof document.execCommand === "function" && document.execCommand("copy");
    ta.remove();
    return !!ok;
  } catch {
    return false;
  }
}
