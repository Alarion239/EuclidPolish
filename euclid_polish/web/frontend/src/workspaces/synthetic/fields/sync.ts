/* Fields › look: keep the real and synthetic viewers on ONE colour transfer
   while each keeps its own toolbar. A viewer reports its state (onState);
   an edit there (a colour chip, the knee or brightness slider, a Q–Y key) is
   a change against that viewer's previous report, and becomes the shared
   transfer, which is then applied to both viewers (setView). The first
   report of a viewer only records its state, so opening the page writes
   nothing. The shared transfer is the URL's: an unset knee / brightness
   means the locked default (DEFAULT_TRANSFER), sent explicitly, so a reset
   really resets both viewers and what is on screen always matches the
   URL. Pure. */
import { DEFAULT_TRANSFER } from "../../../state/display";

export type Transfer = { color: string; knee: number | null; gain: number | null };
export type Reported = { color: string; knee: number; gain: number };

/** The locked default knee (e⁻) and brightness an unset field stands for. */
export const DEFAULT_KNEE = DEFAULT_TRANSFER.knee;
export const DEFAULT_GAIN = DEFAULT_TRANSFER.gain;

const close = (a: number, b: number) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(a), Math.abs(b));

/** The explicit transfer a shared one stands for (null → the default). */
export function resolved(shared: Transfer): Reported {
  return { color: shared.color, knee: shared.knee ?? DEFAULT_KNEE, gain: shared.gain ?? DEFAULT_GAIN };
}

/** The part of `next` that differs from the previous report (null: nothing
 *  new, or this is the first report). */
export function editOf(prev: Reported | null, next: Reported): Partial<Transfer> | null {
  if (!prev) return null;
  const out: Partial<Transfer> = {};
  if (next.color !== prev.color) out.color = next.color;
  if (!close(next.knee, prev.knee)) out.knee = next.knee;
  if (!close(next.gain, prev.gain)) out.gain = next.gain;
  return Object.keys(out).length ? out : null;
}

/** Does a viewer already show `shared`? (skip a redundant setView). */
export function shows(state: Reported | null, shared: Transfer): boolean {
  if (!state) return false;
  const want = resolved(shared);
  return state.color === want.color && close(state.knee, want.knee) && close(state.gain, want.gain);
}

/** The setView patch of a shared transfer: always explicit, so the viewers
 *  hold the page's transfer (not the Display panel's) and a reset lands. */
export function viewPatch(shared: Transfer): Reported {
  return resolved(shared);
}

/** The URL value of a shared knee / brightness: 0 (unset) for the locked
 *  default, else rounded (4 / 3 significant figures) — a clean link. */
export function urlKnee(v: number): number {
  return Math.abs(v - DEFAULT_KNEE) < 1e-6 ? 0 : Number(v.toPrecision(4));
}
export function urlGain(v: number): number {
  return Math.abs(v - DEFAULT_GAIN) < 1e-6 ? 0 : Number(v.toPrecision(3));
}
