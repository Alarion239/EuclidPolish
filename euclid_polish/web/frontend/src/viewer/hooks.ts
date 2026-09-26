/* React bindings of a ViewerController. */
import { createContext, useContext, useMemo } from "react";
import { useStore } from "zustand";
import { mergeDisplay, useDisplay, type DisplaySettings } from "../state/display";
import type { ViewerController, ViewerStoreState } from "./controller";

export const ViewerContext = createContext<ViewerController | null>(null);

export function useController(): ViewerController {
  const c = useContext(ViewerContext);
  if (!c) throw new Error("viewer components must be rendered inside <ImageViewer>");
  return c;
}

/** Subscribe to one slice of the viewer's store. */
export function useViewer<T>(selector: (s: ViewerStoreState) => T): T {
  return useStore(useController().store, selector);
}

/** The effective display settings (Display panel ∘ this viewer's override). */
export function useSettings(): DisplaySettings {
  const global = useDisplay();
  const override = useViewer((s) => s.override);
  return useMemo(() => mergeDisplay(global, override), [global, override]);
}
