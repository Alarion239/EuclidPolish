import { useEffect, useState } from "react";

export type Theme = "light" | "dark";
const KEY = "ep-theme";

/** The persisted theme (default light). Mirrors the pre-paint script in
 *  index.html so the first React render matches what's already on <html>. */
export function readTheme(): Theme {
  try {
    const s = localStorage.getItem(KEY);
    if (s === "dark" || s === "light") return s;
  } catch { /* localStorage unavailable — fall through */ }
  return "light";
}

/** Theme state bound to <html data-theme> + localStorage. `toggle` flips
 *  light↔dark; because the state lives at the app root, a flip re-renders the
 *  whole tree so the canvas figures redraw from the new theme's tokens. */
export function useTheme(): { theme: Theme; toggle: () => void } {
  const [theme, setTheme] = useState<Theme>(readTheme);
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem(KEY, theme); } catch { /* ignore */ }
  }, [theme]);
  return {
    theme,
    toggle: () => setTheme((t) => (t === "light" ? "dark" : "light")),
  };
}
