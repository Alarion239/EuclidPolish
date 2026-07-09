/* Sidebar navigation model. Each item either routes inside the SPA (`to`) or
   bridges to a legacy Flask page (`legacy`) until that tab is ported. Grouped
   into sections so the 18-tab rail is scannable. */
export type NavItem = { label: string; to?: string; legacy?: string; done?: boolean };
export type NavSection = { title: string; items: NavItem[] };

export const NAV: NavSection[] = [
  {
    title: "Setup",
    items: [
      { label: "Config", legacy: "/config" },
      { label: "Catalog", legacy: "/catalog" },
      { label: "PSFs", legacy: "/psfs" },
    ],
  },
  {
    title: "Data",
    items: [
      { label: "Sky", legacy: "/sky" },
      { label: "Cutouts", legacy: "/cutouts" },
      { label: "TNG", legacy: "/tng" },
    ],
  },
  {
    title: "Model",
    items: [
      { label: "Inference", legacy: "/inference" },
      { label: "Ensemble", to: "/ensemble", done: true },
      { label: "Evaluation", legacy: "/evaluation" },
      { label: "Lens finder", legacy: "/lensfinder" },
    ],
  },
  {
    title: "Ops",
    items: [
      { label: "Tracking", legacy: "/tracking" },
      { label: "Visualization", legacy: "/visualization" },
      { label: "FASRC", legacy: "/fasrc" },
      { label: "Git", legacy: "/git" },
    ],
  },
];
