/* Escape inside the narrow-screen inspector sheet.
 *
 * The sheet is a Radix modal dialog: it hears Escape in the capture phase,
 * before the image viewer (a document listener) can use the key to leave
 * focus mode, unfreeze its lens, clear its profile or close its Display
 * row. Without this check one Esc both left focus mode and closed the sheet
 * (the tile card vanished under the user). */

/** Whether a viewer inside `root` is in a state it leaves on Escape. */
export function viewerTakesEscape(root: ParentNode | null | undefined): boolean {
  return !!root?.querySelector(".cv-root[data-focus], .cv-lens--frozen, .cv-body[data-dock], .cv-svg .cv-prof");
}
