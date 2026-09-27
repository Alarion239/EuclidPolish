/* The ensemble tab strip's aside slot: an element right of the tabs (left of
 * the regime switch) that a tab may portal ONE compact control into — e.g.
 * Disagreement's "Pick members" menu — so the control is reachable without
 * scrolling and costs no row above the images. Provided by ./index.tsx;
 * null outside the workspace (a tab then shows only its in-page controls). */
import { createContext, useContext } from "react";

export const TabAsideSlot = createContext<HTMLElement | null>(null);

export const useTabAside = (): HTMLElement | null => useContext(TabAsideSlot);
