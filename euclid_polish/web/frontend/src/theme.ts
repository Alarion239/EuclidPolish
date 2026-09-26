/* Theme — compatibility module over the prefs store (`state/prefs.ts`) for
 * the pre-rework pages.
 *
 * The preference (light | dark | system) lives in `usePrefs`; the RESOLVED
 * theme is written to <html data-theme> by `bindPrefsToDocument()` (main.tsx)
 * and read by components through `useResolvedTheme()` / `useThemeValue()`.
 * The old `ThemeContext` / `useTheme` (fed by the removed App.tsx) are gone:
 * the theme toggle lives in the top bar, the choice in Settings › Appearance.
 */
import { resolveTheme, usePrefs, useResolvedTheme, type ResolvedTheme } from "./state/prefs";

export type Theme = ResolvedTheme;

/** Current resolved theme, for components that must recompute on a theme flip
 *  (e.g. canvas figures whose colours read live from the CSS tokens). */
export const useThemeValue = (): Theme => useResolvedTheme();

/** The resolved theme right now (outside React). */
export function readTheme(): Theme {
  return resolveTheme(usePrefs.getState().theme);
}
