/* Open/closed state of the shell's overlays (palette, ? sheet, display
 * panel, job tray, narrow-screen rail drawer), so any part of the app — a
 * shortcut, a palette command, a page button — can open one. Session-only. */
import { create } from "zustand";

export type ShellOverlay = "palette" | "shortcuts" | "display" | "tray" | "drawer";

type ShellUi = Record<ShellOverlay, boolean> & {
  setOpen: (overlay: ShellOverlay, open: boolean) => void;
  toggle: (overlay: ShellOverlay) => void;
  /** Open one overlay and close the others (they are mutually exclusive). */
  openOnly: (overlay: ShellOverlay) => void;
  closeAll: () => void;
};

const CLOSED: Record<ShellOverlay, boolean> = {
  palette: false, shortcuts: false, display: false, tray: false, drawer: false,
};

export const useShellUi = create<ShellUi>()((set, get) => ({
  ...CLOSED,
  setOpen: (overlay, open) => set({ [overlay]: open } as Partial<ShellUi>),
  toggle: (overlay) => set({ [overlay]: !get()[overlay] } as Partial<ShellUi>),
  openOnly: (overlay) => set({ ...CLOSED, [overlay]: true }),
  closeAll: () => set({ ...CLOSED }),
}));

export const openPalette = () => useShellUi.getState().openOnly("palette");
export const openDisplayPanel = () => useShellUi.getState().openOnly("display");
export const openShortcutSheet = () => useShellUi.getState().openOnly("shortcuts");
export const openJobTray = () => useShellUi.getState().openOnly("tray");
