/* Viewer-only glyphs (16-unit stroke icons, currentColor), passed to the kit's
 * IconButton as nodes. Decorative: the button carries the label. */
const PATHS = {
  pan: "M8 1.8v12.4M1.8 8h12.4M8 1.8L6.3 3.5M8 1.8l1.7 1.7M8 14.2l-1.7-1.7M8 14.2l1.7-1.7M1.8 8l1.7-1.7M1.8 8l1.7 1.7M14.2 8l-1.7-1.7M14.2 8l-1.7 1.7",
  lens: "M6.5 11a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9zM9.8 9.8L14 14M4.8 6.5h3.4M6.5 4.8v3.4",
  fit: "M2.5 5.5v-3h3M10.5 2.5h3v3M13.5 10.5v3h-3M5.5 13.5h-3v-3",
  blink: "M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8zM8 10a2 2 0 1 0 0-4 2 2 0 0 0 0 4z",
  swipe: "M8 1.5v13M2.5 3.5h11v9h-11zM5.5 8H3.8M3.8 8l1-1M3.8 8l1 1M10.5 8h1.7M12.2 8l-1-1M12.2 8l-1 1",
  histogram: "M2 13.5h12M3.5 13.5V9M6 13.5V4.5M8.5 13.5V7M11 13.5V10.5M13.2 13.5v-1.5",
  profile: "M1.5 12.5c2-.2 3-1.5 4-4.5s1.8-5 2.5-5 1.5 2 2.5 5 2 4.3 4 4.5M1.5 14.5h13",
  residual: "M3 8h4M5 6v4M9 8h4M8 2.5v11",
  palette: "M8 14.5A6.5 6.5 0 1 1 14.5 8c0 1.4-1.1 2-2.2 2H10.6a1.2 1.2 0 0 0-.9 2c.5.6.2 2.5-1.7 2.5zM4.8 8.2h.01M6.2 5.2h.01M9.6 4.8h.01M11.8 7h.01",
  unlink: "M6.5 9.5l-1.8 1.8a2.3 2.3 0 0 1-3.2-3.2l1.8-1.8M9.5 6.5l1.8-1.8a2.3 2.3 0 0 1 3.2 3.2l-1.8 1.8M3 3l10 10",
  link: "M6.8 9.2l2.4-2.4M5.5 7.5L3.9 9.1a2.3 2.3 0 0 0 3.2 3.2l1.6-1.6M10.5 8.5l1.6-1.6a2.3 2.3 0 0 0-3.2-3.2L7.3 5.3",
  record: "M8 12.5a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9z",
  stopRecord: "M4.5 4.5h7v7h-7z",
  figure: "M2.5 2.5h11v11h-11zM2.5 11l3-3 2.5 2.5 2-2 3.5 3.5M10.5 5.5h.01",
};

export type ViewerIconName = keyof typeof PATHS;

export function VIcon({ name, size = 16 }: { name: ViewerIconName; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      <path d={PATHS[name]} />
    </svg>
  );
}
