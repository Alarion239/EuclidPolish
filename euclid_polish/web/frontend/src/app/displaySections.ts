/* Extension point of the Display panel: a workspace adds its own section
 * (the Sky atlas adds "Sky" — base HiPS colormap, stretch, cuts…) while its
 * module is loaded.
 *
 *   useEffect(() => registerDisplaySection({ id: "sky", title: "Sky", order: 10, Component: SkyDisplay }), []);
 *
 * Sections render below the built-in "Image" and "Viewers" sections, by
 * `order` then registration order. */
import type { ComponentType } from "react";
import { create } from "zustand";

export type DisplaySection = {
  id: string;
  title: string;
  /** Lower first (default 100). */
  order?: number;
  Component: ComponentType;
};

type Registry = {
  sections: DisplaySection[];
  reset: () => void;
};

export const useDisplaySections = create<Registry>()((set) => ({
  sections: [],
  reset: () => set({ sections: [] }),
}));

/** Add (or replace, by id) a section; returns the unregister function. */
export function registerDisplaySection(section: DisplaySection): () => void {
  const others = useDisplaySections.getState().sections.filter((s) => s.id !== section.id);
  const next = [...others, section].sort((a, b) => (a.order ?? 100) - (b.order ?? 100));
  useDisplaySections.setState({ sections: next });
  return () => {
    const cur = useDisplaySections.getState().sections;
    if (cur.includes(section)) useDisplaySections.setState({ sections: cur.filter((s) => s !== section) });
  };
}
